# -*- coding: utf-8 -*-
"""MASTER 骨干（三段注意力）+ 加权 MSE 损失。

结构（研报 p10 三段公式 + 官方 MASTER 源码形态）：
  门控乘法 → Linear(70→64)+正弦位置编码 → 时序注意力(4头,无温度)
  → 截面注意力(2头,温度√(d/头),逐时间步梯度checkpoint) → 末时刻query时序聚合
  → Linear(64→1) → Sigmoid（配 [0,1] 排序标签）
注意力块对齐官方：norm1→QKV(无bias)→多头注意力(矩阵dropout)→残差→norm2→两层FFN→残差。
"""
import math

import torch
import torch.nn as nn

from config import CFG, DA_LOSS, DA_W


class MultiHeadAttention(nn.Module):
    """官方 MASTER 注意力块。temp=1.0 用于时序注意力，temp=√(d/头) 用于截面注意力。"""

    def __init__(self, d, nhead, dropout, temp):
        super().__init__()
        self.h, self.dk, self.temp = nhead, d // nhead, temp
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, x):                                    # x:(B,L,d)
        z = self.norm1(x)
        B, L, d = z.shape
        q = self.q(z).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(z).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(z).view(B, L, self.h, self.dk).transpose(1, 2)
        a = self.drop(torch.softmax(q @ k.transpose(-1, -2) / self.temp, dim=-1))
        h = self.norm2(z + (a @ v).transpose(1, 2).reshape(B, L, d))
        return h + self.ffn(h)


class MasterBackbone(nn.Module):
    """接收 70 维个股输入与 70 维门控权重，输出 (0,1) 预测分数。

    截面注意力逐时间步 + 梯度 checkpoint：n≈几千时 (n,n) 注意力矩阵 × 40 步的显存分界。
    forward(x, w70)：x=(n,40,70) 个股批；w70=(70,) 市场状态门控权重（由 MarketGate 产出）。
    """

    def __init__(self):
        super().__init__()
        d = CFG['m_dmodel']
        self.feat = nn.Linear(70, d)
        pe = torch.zeros(CFG['window'], d)                    # 正弦位置编码（官方）
        pos = torch.arange(CFG['window']).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.tatt = MultiHeadAttention(d, CFG['m_tnhead'], CFG['m_dropout'], temp=1.0)
        self.satt = MultiHeadAttention(d, CFG['m_snhead'], CFG['m_dropout'],
                                       temp=math.sqrt(d / CFG['m_snhead']))
        self.wlam = nn.Linear(d, d, bias=False)               # 时序聚合 λ=softmax(z_tᵀW z_τ)
        self.dec = nn.Linear(d, 1)

    def _satt_step(self, zs):                                 # 单时间步跨股票注意力（供 checkpoint）
        return self.satt(zs.unsqueeze(0)).squeeze(0)          # (n,d) 视作 (B=1,L=n,d)

    def forward(self, x, w70):                                # x:(n,40,70) w70:(70,)
        h = self.feat(x * w70) + self.pe                      # 门控乘法作用于全部 40 步
        h = self.tatt(h)                                      # 股票内时序注意力（B=n,L=40）
        outs = [torch.utils.checkpoint.checkpoint(self._satt_step, h[:, s, :], use_reentrant=False)
                if self.training else self._satt_step(h[:, s, :])
                for s in range(h.shape[1])]                   # 逐时间步跨股票注意力
        z = torch.stack(outs, dim=1)                          # (n,40,d)
        lam = torch.softmax((self.wlam(z) @ z[:, -1, :].unsqueeze(-1)).squeeze(-1), dim=1)
        e = (lam.unsqueeze(-1) * z).sum(dim=1)                # 末时刻 query 的时序聚合
        return torch.sigmoid(self.dec(e).squeeze(-1))         # (n,) ∈(0,1)


def wmse(pred, y):
    """加权 MSE（研报最终模型口径 p16）：w=Sigmoid(γ(ŷ−0.5))，ŷ>0.5 端权重大（多头端优先）。
    权重项对 base 取 detach（防权重项产生鼓励自抬预测的伪梯度）。"""
    base = y if DA_W == 'label' else pred
    w = torch.sigmoid(CFG['m_gamma'] * (base.detach() - 0.5))
    return (w * (pred - y) ** 2).mean()


def ic_loss(pred, y):
    """截面 IC 损失 = −Pearson(ŷ, y)（研报基线口径 p4 图3「损失函数: -IC」）。
    批 = 单日截面，故此处即当日 IC。**对 ŷ 与 y 的仿射变换均不变**，故 γ 的多头加权
    在该损失下无从体现（式中根本不含 γ）。

    标签适配器 H 是否失效取决于 DA_H：
      DA_H=affine（y 域仿射）→ Pearson 不变 ⇒ H 无梯度（实测 |∂| < 1e-8），退化为恒等；
      DA_H=logit（主线，logit 域仿射）→ 在 y 空间是非线性单调变换，Pearson 并不不变
        ⇒ H 仍有梯度（实测 ∂/∂scale≈3.4e-3、∂/∂bias≈6.2e-3），DoubleAdapt 标签适配未失效。
    """
    p = pred - pred.mean()
    t = y - y.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + CFG['eps'])


def rank_wmse(pred, y):
    """秩空间加权 MSE：先把预测按截面高斯 CDF 摊平到近似均匀，再进加权 MSE。

    u = Φ((ŷ−μ)/σ)，μ/σ 取当日截面统计量，全程可微（torch.erf），O(n)。
    标签 y 本就是截面排序归一到 [0,1]（均匀），u 摊平后与之同分布 —— 两个均匀变量的
    MSE = 1/6 − 2·Cov(u,y)，故最小化该项等价于最大化 Spearman(ŷ,y)=RankIC，
    再乘 w 即为「重心压在多头端的加权 RankIC」。

    摊平的实际作用是恢复 γ 的动态范围：原始 ŷ 截面 std≈0.038，w 比值仅 2.17×；
    u 铺满 [0,1] 后 γ=5 的 w 比值达 12.06×。权重项对 u 取 detach，同 wmse。
    """
    z = (pred - pred.mean()) / (pred.std() + CFG['eps'])
    u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))          # 标准正态 CDF，可微
    base = y if DA_W == 'label' else u
    w = torch.sigmoid(CFG['m_gamma'] * (base.detach() - 0.5))
    return (w * (u - y) ** 2).mean()


def train_loss(pred, y):
    """训练用损失，由 DA_LOSS 分派（默认 wmse=定稿口径）。"""
    if DA_LOSS == 'ic':
        return ic_loss(pred, y)
    if DA_LOSS == 'rankwmse':
        return rank_wmse(pred, y)
    if DA_LOSS == 'prank':                       # A：纯头部 pairwise 排序
        return pairwise_rank_loss(pred, y)
    if DA_LOSS == 'listnet':                     # B：纯 listwise 交叉熵
        return listnet_loss(pred, y)
    if DA_LOSS == 'focal':                       # C：wMSE + focal 顶部分类辅助
        return wmse(pred, y) + CFG['focal_lambda'] * focal_aux_loss(pred, y)
    if DA_LOSS == 'hybrid':                      # D：wMSE + pairwise 混合
        return wmse(pred, y) + CFG['hb_lambda'] * pairwise_rank_loss(pred, y)
    return wmse(pred, y)


# ================================================================== ABCD 头部损失实验
# 设计背景：wMSE 拟合条件均值，对"高方差小概率"的极端头部天然往中间缩（回归稀释），
# 且权重挂在被压缩的预测值上、多头端加权几乎不激活 → 模型 G1/G2 区分度≈0。
# 以下损失只学"相对顺序"（排序/分类），不拟合数值，结构上规避均值缩。
# 统一在 z=logit(pred) 单调空间计算（排序只看顺序，logit 空间数值条件好、无 sigmoid 饱和）。

def _rank_scores(pred):
    """排序分数 z = logit(pred)，clamp 防 ±inf。"""
    return torch.logit(pred.clamp(1e-5, 1 - 1e-5))


def pairwise_rank_loss(pred, y):
    """A（DA_LOSS=prank）：头部加权 pairwise 排序损失；同时是 D（hybrid）的排序项。

    当日截面内：正样本 = 标签前 20%（y≥rk_head_q），每个正样本配 rk_k 个随机负样本
    （只保留标签更低者），L = Σ w·softplus(−(z_pos−z_neg)/τ) / Σw；
    y≥rk_topq 的正样本权重 ×(1+rk_wtop)，对应"G1 应排在 G2 前"的边界对。
    负样本索引在 CPU 采样再挪上卡：离线训练续跑恢复的是 CPU RNG，保证续跑采样一致。
    """
    z = _rank_scores(pred)
    q, k, tau = CFG['rk_head_q'], CFG['rk_k'], CFG['rk_tau']
    topq, wtop = CFG['rk_topq'], CFG['rk_wtop']
    n = y.shape[0]
    pos_mask = y >= q
    npos = int(pos_mask.sum())
    if npos == 0 or n < 2:
        return (pred - y).pow(2).mean() * 0.0                    # 兜底：零损失（保持计算图）
    z_pos, y_pos = z[pos_mask], y[pos_mask]                      # (P,)
    neg_idx = torch.randint(0, n, (npos, k)).to(y.device)        # (P,K)
    y_neg, z_neg = y[neg_idx], z[neg_idx]                        # (P,K)
    valid = (y_neg < y_pos.unsqueeze(1)).float()                 # 负样本标签必须更低
    w = (1.0 + wtop * (y_pos >= topq).float()).unsqueeze(1) * valid
    loss = (w * torch.nn.functional.softplus(-(z_pos.unsqueeze(1) - z_neg) / tau)).sum()
    return loss / w.sum().clamp(min=1.0)


def listnet_loss(pred, y):
    """B（DA_LOSS=listnet）：ListNet 式 listwise 交叉熵。

    q = softmax(y/τq)：目标分布，τq 小 → 概率质量压在头部股票上；
    p = softmax(z/τz)：模型分布；L = −Σ q·log p = H(q,p)（与 KL(q‖p) 同最优解）。
    O(n) 无需采 pair，直接优化"头部整体质量"。
    """
    z = _rank_scores(pred)
    q = torch.softmax(y / CFG['ln_tau_q'], dim=0)
    log_p = torch.log_softmax(z / CFG['ln_tau_z'], dim=0)
    # 除以 log(n)：归一化交叉熵（量纲与截面大小无关、初值≈O(1)，对 FOMAML 内层步更友好）
    return -(q * log_p).sum() / math.log(max(y.shape[0], 2))


def focal_aux_loss(pred, y):
    """C 的辅助项（DA_LOSS=focal 时按 focal_lambda 与 wMSE 相加）：
    "是否真实头部（y≥focal_topq）"的 focal 二分类，直接复用 Sigmoid 输出 pred 作概率，
    无需新增输出头；γ 聚焦难分样本、α 平衡正负类（正类≈10%）。"""
    p = pred.clamp(1e-6, 1 - 1e-6)
    b = (y >= CFG['focal_topq']).float()
    g, a = CFG['focal_gamma'], CFG['focal_alpha']
    pt = b * p + (1.0 - b) * (1.0 - p)                           # p_t
    at = b * a + (1.0 - b) * (1.0 - a)                           # α_t
    return (-at * (1.0 - pt) ** g * torch.log(pt)).mean()

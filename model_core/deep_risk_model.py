# -*- coding: utf-8 -*-
"""自建深度风险模型 DRM：隐式 GAT + 双支 GRU，K=10。

结构为 Deep Risk Model 论文(ICAIF'21)的研报改造版（窗 40 / GAT dropout 0.3 / scaled-dot 隐式 GAT）。
损失 = 多期归一化残差 R²（对 y_{t+h} 自投影）+ λ·tr((FᵀF)⁻¹) VIF 迹正则。
产物=10 维深度风险暴露，作为 Meta Master 的市场状态(120维衍生)与个股输入(后10维)来源。
"""
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from config import CFG


class ImplicitGAT(nn.Module):
    """研报字面（P4 代码面板）：Q=K=V 共享线性投影 → 缩放点积 → LeakyReLU → 截面 softmax。
    逐时间步在当日截面（batch 维）建全连接注意力，无预定义邻接矩阵。训练态用梯度检查点省显存。"""

    def __init__(self, d, dropout):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)                  # 注意力投影无偏置
        self.scale = math.sqrt(d)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)                          # 注意力矩阵 dropout(主线0.3/deck变体0.5)

    def _step(self, qs):                                         # 单时间步截面注意力
        a = qs @ qs.T / self.scale
        return self.drop(torch.softmax(self.act(a), dim=-1)) @ qs

    def forward(self, x):                                        # x: (n, 40, d)
        q = self.proj(x)
        outs = [torch.utils.checkpoint.checkpoint(self._step, q[:, s, :], use_reentrant=False)
                if self.training else self._step(q[:, s, :])
                for s in range(x.shape[1])]
        return torch.stack(outs, dim=1)                          # (n, 40, d)


class DRM(nn.Module):
    """上支：残差(输入−GAT输出)→GRU→FC 出 K/2 个特质因子；下支：原始→GRU→FC 出 K/2 个全局因子。"""

    def __init__(self):
        super().__init__()
        d, k = CFG['n_price'] + CFG['n_barra'], CFG['drm_k']
        self.gat = ImplicitGAT(d, CFG['drm_gat_dropout'])
        self.gru1 = nn.GRU(d, CFG['drm_hidden'], CFG['drm_gru_layers'], batch_first=True)
        self.gru2 = nn.GRU(d, CFG['drm_hidden'], CFG['drm_gru_layers'], batch_first=True)
        self.fc1 = nn.Linear(CFG['drm_hidden'], k // 2)
        self.fc2 = nn.Linear(CFG['drm_hidden'], k // 2)

    def forward(self, x):                                        # x: (n, 40, 15)
        resid = x - self.gat(x)                                  # 剥离截面相关后的特质信息
        f1 = self.fc1(self.gru1(resid)[0][:, -1])               # 各支取 GRU 末时刻隐状态
        f2 = self.fc2(self.gru2(x)[0][:, -1])
        return torch.cat([f1, f2], dim=1)                        # (n, 10)


def cs_zscore(f):
    """截面 z-score（"FC+Norm"的 Norm；总体标准差，与 VIF 推导"因子方差1"自洽）。训推统一。"""
    return (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True, unbiased=False) + CFG['eps'])


def drm_loss(f_raw, y):
    """多期归一化残差 R²（y_{t+h} 自投影）+ λ·平均VIF。y:(n,H) 未来H日逐日收益。"""
    F = cs_zscore(f_raw)                                         # (n,10)
    G = F.T @ F + 1e-4 * torch.eye(CFG['drm_k'], device=f_raw.device)   # 岭稳定项防奇异
    Ginv = torch.linalg.inv(G)
    B = Ginv @ (F.T @ y)                                         # (10,H) 各期回归系数（y_h 自投影）
    resid = y - F @ B                                            # (n,H)
    r2_term = ((resid ** 2).sum(0) / (y ** 2).sum(0).clamp(min=CFG['eps'])).mean()   # 非中心化残差占比
    vif_term = torch.trace(Ginv) * len(F) / CFG['drm_k']        # 平均 VIF
    return r2_term + CFG['drm_lam_vif'] * vif_term

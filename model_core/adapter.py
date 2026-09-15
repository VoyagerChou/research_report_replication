# -*- coding: utf-8 -*-
"""DoubleAdapt 数据适配器 + φ 容器 + 参数换值上下文。

三块：
  · DataAdapter —— 官方 net.py 形态：G 特征适配器 + H/H_inv 标签适配器 + strengths 门控；
    标签仿射改在 logit 域执行以匹配 MASTER 骨干的 (0,1) Sigmoid 输出。
  · MetaMasterNet —— φ 容器：MASTER 骨干 + 市场状态门控 作为一个预测模型整体。
  · ParamSwap —— 参数换值上下文，兼容截面注意力的梯度 checkpoint（functional_call 的快权重
    前向与 checkpoint 反向重算不兼容；参数换值 + FOMAML 恒等 Jacobian 数学等价且兼容）。
"""
import math

import torch
import torch.nn as nn

from config import CFG, DA_H, DA_OPT
from model_core.master_backbone import MasterBackbone
from model_core.market_gate import MarketGate


class DataAdapter(nn.Module):
    """可学习参数 ψ = {Pf, fheads×Nh, lin, Pl, weight(γ), bias(β)}。作用在 (n,40,70) 个股批上。"""

    def __init__(self):
        super().__init__()
        Nh, nf, w, hid = CFG['da_heads'], 70, CFG['window'], CFG['da_label_hid']
        self.tau = CFG['da_tau']
        # 特征适配器：逐时间步门控原型 + Nh 个 70×70 线性头
        self.Pf = nn.Parameter(torch.empty(Nh, nf))
        nn.init.kaiming_uniform_(self.Pf, a=math.sqrt(5))
        self.fheads = nn.ModuleList([nn.Linear(nf, nf) for _ in range(Nh)])
        # 标签适配器：展平 2800→32 学习投影 + 专用原型 + 每头仿射(γ, β)
        self.lin = nn.Linear(w * nf, hid, bias=False)
        self.Pl = nn.Parameter(torch.empty(Nh, hid))
        nn.init.kaiming_uniform_(self.Pl, a=math.sqrt(5))
        self.weight = nn.Parameter(torch.empty(1, Nh))        # γ
        nn.init.uniform_(self.weight, 0.75, 1.25)
        # β 初值按域选:logit 域沿用 1/Nh(σ(z+0.125) 轻微上移,无碍);仿射域必须 0——
        # 1/Nh 会在第 0 步就把 H(y)=y+0.125,顶部 12.5% 标签直接出界,近恒等起步才公平。
        self.bias = nn.Parameter(torch.zeros(1, Nh) if DA_H == 'affine'
                                 else torch.ones(1, Nh) / Nh)
        if DA_OPT == 'ft':                                    # 内层lr可学习(meta-SGD;研报p11字面
            self.log_ilr = nn.Parameter(torch.tensor(math.log(CFG['da_inner_lr'])))   # "输出初始参数以及学习率")

    def strengths(self, x):                                   # x:(n,40,70) → (n,Nh) 标签门控
        v = torch.nn.functional.normalize(self.lin(x.reshape(x.shape[0], -1)), dim=1)
        p = torch.nn.functional.normalize(self.Pl, dim=1)
        return torch.softmax((v @ p.T) / self.tau, dim=1)

    def G(self, x):                                           # 逐时间步门控残差变换，(n,40,70)→(n,40,70)
        sims = [torch.nn.functional.cosine_similarity(x, self.Pf[i], dim=-1)
                for i in range(len(self.fheads))]             # 每头 (n,40)
        s = torch.softmax(torch.stack(sims, dim=-1) / self.tau, dim=-1)
        out = x
        for i in range(len(self.fheads)):
            out = out + s[..., i:i + 1] * self.fheads[i](x)
        return out

    @staticmethod
    def _logit01(y):
        return torch.logit(y.clamp(1e-5, 1 - 1e-5))

    def H(self, y, s):
        """标签适配。DA_H 二选一（logit=定稿默认）:
        affine = 研报p10字面 h_i(y)=γ_i·y+β_i,目标允许出界,Sigmoid 头饱和 + reg 锚自稳;
        logit  = 主线,仿射作用在 logit(y) 再映回 (0,1),不出界但头部间距被压缩。"""
        if DA_H == 'affine':
            return (s * ((self.weight + 1e-9) * y[:, None] + self.bias)).sum(1)
        z = self._logit01(y)
        zt = (s * ((self.weight + 1e-9) * z[:, None] + self.bias)).sum(1)
        return torch.sigmoid(zt)                              # 始终 ∈(0,1)，与骨干 Sigmoid 输出同域

    def H_inv(self, yt, s):                                   # 逆变换，还原为排序分数
        if DA_H == 'affine':
            return (s * ((yt[:, None] - self.bias) / (self.weight + 1e-9))).sum(1)
        zt = self._logit01(yt)
        z = (s * ((zt[:, None] - self.bias) / (self.weight + 1e-9))).sum(1)
        return torch.sigmoid(z)


class MetaMasterNet(nn.Module):
    """φ 容器：MASTER 骨干 + 门控作为一个预测模型整体（模型适应器 MAML 作用于整个 φ）。"""

    def __init__(self):
        super().__init__()
        self.net, self.gate = MasterBackbone(), MarketGate(mode=CFG['gate_mode'])

    def forward(self, x, mz):                                 # x:(n,40,70) mz:(183,) robust-z 市场状态
        return self.net(x, self.gate(mz))


class ParamSwap:
    """参数换值上下文：p.data ← p.data − delta，退出时逐位还原。换值期间前向/反传所得 φ 梯度
    即外层元梯度（FOMAML 下 fast=φ−常数，Jacobian 恒等）。用参数换值而非 functional_call，
    因为 functional_call 的临时替换在 checkpoint 反向重算阶段已退出、会读回真实参数算错梯度。"""

    def __init__(self, phi, delta):
        self.phi, self.delta = phi, delta

    def __enter__(self):
        self.bak = {k: p.data.clone() for k, p in self.phi.named_parameters()}
        for k, p in self.phi.named_parameters():
            p.data = p.data - self.delta[k]

    def __exit__(self, *a):
        for k, p in self.phi.named_parameters():
            p.data = self.bak[k]

# -*- coding: utf-8 -*-
"""市场状态门控（183 维 = 63 指数 + 120 DRM 衍生）+ 门控标准化器。

门控输入为 183 维市场状态；标准化为训练窗逐列 robust-z（逐列统计量保留各维量纲与
"当日水平"信息），与官方 MASTER 数据管道的 RobustZScoreNorm 口径一致。
"""
import numpy as np
import torch
import torch.nn as nn

from config import CFG


class GateNormalizer:
    """183 维市场状态的训练窗逐列 robust 标准化（PIT）：统计量只在训练窗拟合，随 checkpoint 保存。
    (x−median)/(1.4826·MAD)，截 ±gate_clip（amp 重尾规格）。med/mad 以 torch tensor 入盘
    （torch≥2.13 默认 weights_only=True，拒绝 numpy 对象）。"""

    def fit(self, m_matrix):                                  # m_matrix:(n_train,183) 训练窗市场状态
        _m = np.asarray(m_matrix, dtype='float32')
        self.med = np.nanmedian(_m, axis=0).astype('float32')
        self.mad = (np.nanmedian(np.abs(_m - self.med), axis=0) * 1.4826 + 1e-9).astype('float32')
        return self

    def transform(self, m):                                   # m:(183,) → robust-z 后 torch 张量
        z = (np.asarray(m, dtype='float32') - self.med) / self.mad
        return torch.from_numpy(np.clip(z, -CFG['gate_clip'], CFG['gate_clip']))

    def state_dict(self):
        return {'med': torch.from_numpy(self.med.copy()), 'mad': torch.from_numpy(self.mad.copy())}

    def load_state_dict(self, d):
        self.med = np.asarray(d['med'], dtype='float32')
        self.mad = np.asarray(d['mad'], dtype='float32')
        return self


class MarketGate(nn.Module):
    """市场状态门控（与骨干分离）。输入 robust-z 后的 183 维，输出 70 维特征权重（和=70）。
    mode: dual=研报文字双门控（老门 63→70 + 新门 120→70 + FC 融合，各 softmax×70，均值 1 信号）
        | single=单门控消融 | none=恒等权重基线。β=softmax 温度。
    """

    def __init__(self, d_feat=70, mode='dual'):
        super().__init__()
        self.d_feat, self.mode, self.beta = d_feat, mode, CFG['gate_beta']
        if mode == 'dual':
            self.g_idx = nn.Linear(63, d_feat)               # 老门：指数量价状态
            self.g_risk = nn.Linear(120, d_feat)             # 新门：DRM 风险状态
            self.fuse = nn.Linear(2 * d_feat, d_feat)        # 新老权重全连接融合
        elif mode == 'single':
            self.g_all = nn.Linear(183, d_feat)

    def forward(self, m):                                    # m:(183,) robust-z 后
        if self.mode == 'none':
            return torch.ones(self.d_feat, device=m.device)
        if self.mode == 'single':
            return torch.softmax(self.g_all(m) / self.beta, dim=-1) * self.d_feat
        w1 = torch.softmax(self.g_idx(m[:63]) / self.beta, dim=-1) * self.d_feat
        w2 = torch.softmax(self.g_risk(m[63:]) / self.beta, dim=-1) * self.d_feat
        w = torch.softmax(self.fuse(torch.cat([w1, w2])) / self.beta, dim=-1)
        return w * self.d_feat                               # (70,) 权重和=70，乘法作用于特征

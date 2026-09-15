# -*- coding: utf-8 -*-
"""输入层清理：市场状态（183 维）进门控前的降噪（DA_MCLEAN=1 启用）。

背景（Step3「修传感器」）：
  183 维市场状态中 120 维是「每日涨幅前 10% 股票 × 10 个 DRM 隐暴露 × {5,10,20,40} 窗 ×
  {均值, std, 放大性}」的二阶衍生量——群体每日换血 + 多重窗展开，噪声偏大；主线用钝门控
  （β=5）防噪声，代价是切换期没有刹车力。本模块在门控之前做两级清理：

  1) 状态依赖半衰期 EWMA（因果、逐列）：
     平时半衰期 mclean_hl（默认 5 交易日，平滑噪声）；报警期（读 switch_alarm_daily.csv 的
     state=='ALARM'）半衰期缩短为 mclean_hl_alarm（默认 1.5 交易日，快速跟随）。
     ——「平时钝、断点快」从门控输入层面实现。
  2) 120 维块 PC 去噪：以训练窗（<= da_off_end）的 120 维块拟合主成分，投影保留前
     mclean_pc_k 个主方向后重构（去尾部噪声子空间；保持 120 维形状不变，下游零改动）。

时点纪律：EWMA 只用当日及以前；PC 基只用训练窗拟合（固定线性投影，离线/在线两个进程
确定性一致）。NaN 行保持原样（不改变 MM_DAYS 可训日集合）。
"""
import numpy as np
import pandas as pd

from config import CFG
from model_core.event_schedule import load_alarm_series


def _ewma_state(M, halflife):
    """逐列因果 EWMA（半衰期按日给定）。NaN 行保持 NaN，且不打断状态递推。"""
    T, C = M.shape
    out = np.array(M, dtype='float64', copy=True)
    alpha = 1.0 - np.exp(np.log(0.5) / np.maximum(np.asarray(halflife, dtype='float64'), 1e-6))  # (T,)
    for c in range(C):
        x = M[:, c].astype('float64')
        s = np.nan
        for t in range(T):
            v = x[t]
            if not np.isfinite(v):
                out[t, c] = v
                continue
            s = v if not np.isfinite(s) else alpha[t] * v + (1.0 - alpha[t]) * s
            out[t, c] = s
    return out


def clean_mstate(MSTATE, dates):
    """返回清理后的 MSTATE（同形状 float32）。dates 为对应 DatetimeIndex。"""
    M = np.asarray(MSTATE, dtype='float64')
    dates = pd.DatetimeIndex(dates)

    # ---- 1) 状态依赖半衰期 EWMA ----
    hl = np.full(len(dates), float(CFG['mclean_hl']))
    alarm = load_alarm_series()
    if alarm is not None:
        st = alarm.set_index('date')['state'].reindex(dates)
        hl[np.asarray(st.values == 'ALARM')] = float(CFG['mclean_hl_alarm'])
    else:
        print('  [mclean] 未找到 switch_alarm_daily.csv，EWMA 使用恒定半衰期', flush=True)
    M = _ewma_state(M, hl)

    # ---- 2) 120 维块 PC 去噪（基只用训练窗拟合）----
    blk = M[:, 63:]
    fit = (dates.year <= CFG['da_off_end']) & np.isfinite(blk).all(axis=1)
    if int(fit.sum()) >= 100:
        X = blk[fit]
        mu = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        k = max(1, min(int(CFG['mclean_pc_k']), vt.shape[0]))
        P = vt[:k].T                                       # (120,k)
        ok = np.isfinite(blk).all(axis=1)
        Z = blk[ok] - mu
        M[ok, 63:] = Z @ P @ P.T + mu                      # 投影到前 k 个主方向后重构
    else:
        print('  [mclean] 训练窗样本不足，跳过 PC 去噪', flush=True)
    return M.astype('float32')

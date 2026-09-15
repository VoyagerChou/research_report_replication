# -*- coding: utf-8 -*-
"""标签层：未来10日 vwap 收益 → 全A 截面排序 [0,1]。

排序池 = 全A(VALID∩标签可实现)：训练标签在全市场内排序，指数成分池只进评价层。
产物 DERIVED/labels.npz：
  Y_RAW(T,J)  未来10日 vwap 原始收益 = VWAP_H[t+11]/VWAP_H[t+1]−1（后复权，池无关）
  LBL_OK(T,J) 标签可实现 = 建仓日(t+1)与平仓日(t+11)均真实成交
  Y_RANK(T,J) 全A 内 [0,1] 截面排序标签 =（rank−1）/(n−1)，仅 VALID∩LBL_OK 处有值
依赖：base.npz(PRICE/VALID/DATES/CODES) + tech_arrays.npz(VWAP_H)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CFG, DERIVED, LABELS_PATH
from input_data_update.build_base import listed_ok


def build_labels(force=False):
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = LABELS_PATH
    if out.exists() and not force:
        print(f'{out.name} 已存在，跳过。', flush=True)
        return out
    z = np.load(DERIVED / 'base.npz', allow_pickle=True)
    PRICE, VALID, OBS = z['PRICE'], z['VALID'], z['OBS']
    DATES, CODES = pd.DatetimeIndex(z['DATES']), z['CODES']
    VWAP_H = np.load(DERIVED / 'tech_arrays.npz')['VWAP_H']
    T, J = VALID.shape
    H = CFG['label_h']                                           # 11：t+1 建仓 → t+11 平仓（持有10日）

    TRD = PRICE[:, :, 4] > 0                                     # 当日有成交（volume>0）
    Y_RAW = np.full((T, J), np.nan, 'float32')
    Y_RAW[:T - H] = VWAP_H[H:] / VWAP_H[1:T - H + 1] - 1
    LBL_OK = np.zeros((T, J), bool)
    LBL_OK[:T - H] = TRD[1:T - H + 1] & TRD[H:]                  # 建仓/平仓日均须真实成交

    NEW_OK = listed_ok(OBS)                                      # 剔新股：上市满 min_listed_days 个交易日
    POOL_ALL = VALID & LBL_OK & NEW_OK                           # 排序池基底（全A）
    y_masked = pd.DataFrame(np.where(POOL_ALL, Y_RAW, np.nan), index=DATES, columns=CODES)
    rk = y_masked.rank(axis=1)
    Y_RANK = ((rk - 1).div(rk.max(axis=1) - 1, axis=0)).values.astype('float32')   # [0,1]

    # 防护：手工复算一条 + 值域
    _ok_days = np.where(POOL_ALL.sum(1) > 0)[0]
    _t = int(_ok_days[len(_ok_days) // 2])
    _j = int(np.where(POOL_ALL[_t])[0][0])
    assert abs(float(VWAP_H[_t + H, _j] / VWAP_H[_t + 1, _j] - 1) - float(Y_RAW[_t, _j])) < 1e-6
    assert 0 <= np.nanmin(Y_RANK) and np.nanmax(Y_RANK) <= 1
    assert not np.any(np.isfinite(Y_RANK) & ~POOL_ALL), '排序池越界'

    np.savez(out, Y_RAW=Y_RAW, LBL_OK=LBL_OK, Y_RANK=Y_RANK)
    _n0 = (VALID & LBL_OK).sum(1); _n1 = POOL_ALL.sum(1)
    print(f'{out.name} OK(全A排序): 日均有效标签 {int(_n1[_n1 > 0].mean())} 只/日 '
          f'(排序池=VALID∩可实现∩上市满{CFG["min_listed_days"]}日；剔新股前 {int(_n0[_n0 > 0].mean())} 只/日，'
          f'剔掉 {100 * (1 - _n1.sum() / _n0.sum()):.1f}%) -> {out}', flush=True)
    return out


if __name__ == '__main__':
    build_labels(force='--force' in sys.argv)

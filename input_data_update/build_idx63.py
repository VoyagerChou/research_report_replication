# -*- coding: utf-8 -*-
"""63 维指数市场状态。

3 个指数(沪深300/中证500/中证1000)× 每指数 21 维 = 63，是 183 维市场状态的前 63 列。
指数集本就含 500/1000，故三池共享同一份（市场级向量，与股票池无关）。
每指数 21 维 = 当日收益(1) + {5,10,20,30,60}窗 × {ret均值, ret标准差, 成交额均值/当日额, 成交额标准差/当日额}(4×5=20)。
产物 DERIVED/idx63.npz: STATE63 (T,63) + cols(列名)。数据源 = index_daily_all.pq。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CFG, DERIVED, RAW_TABLES

MM_INPUT = RAW_TABLES
GATE_INDEXES = ['000300.SH', '000905.SH', '000852.SH']            # 沪深300 / 中证500 / 中证1000


def _index_state21(g):
    """单指数 21 维（研报图10字面；比值分母=当日成交额）。"""
    g = g.sort_values('date').set_index('date')
    r = g['pct_chg'] / 100.0                                      # Wind 涨跌幅为百分数
    a = g['amount']
    out = {'ret': r}
    for n in [5, 10, 20, 30, 60]:
        out[f'ret_avg{n}'] = r.rolling(n).mean()
        out[f'ret_std{n}'] = r.rolling(n).std()
        out[f'amt_avg{n}'] = a.rolling(n).mean() / (a + CFG['eps'])
        out[f'amt_std{n}'] = a.rolling(n).std() / (a + CFG['eps'])
    return pd.DataFrame(out)


def build_idx63(force=False):
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / 'idx63.npz'
    if out.exists() and not force:
        print('idx63.npz 已存在，跳过。', flush=True)
        return out
    DATES = pd.DatetimeIndex(np.load(DERIVED / 'base.npz', allow_pickle=True)['DATES'])
    idx = pd.read_parquet(MM_INPUT / 'index_daily' / 'index_daily_all.pq')
    state63 = pd.concat({code.split('.')[0]: _index_state21(idx[idx['code'] == code])
                         for code in GATE_INDEXES}, axis=1)
    state63.columns = [f'idx63_{i}_{c}' for i, c in state63.columns]
    state63 = state63.reindex(DATES)                              # 对齐全局交易日轴
    assert state63.shape[1] == 63, f'指数状态维度 {state63.shape[1]}≠63'
    np.savez(out, STATE63=state63.values.astype('float32'), cols=np.asarray(state63.columns))
    print(f'idx63.npz OK: (T,63)=({len(DATES)},63) 指数={GATE_INDEXES} -> {out}', flush=True)
    return out


if __name__ == '__main__':
    build_idx63(force='--force' in sys.argv)

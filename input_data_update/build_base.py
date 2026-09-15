# -*- coding: utf-8 -*-
"""数据层：从原始行情/Barra/ST 构造全市场对齐基础矩阵。

全市场（不瘦身：全A 训练 + DRM 全市场共用）。产物 DERIVED/base.npz：
  PRICE(T,J,5) 后复权 OHLC + 原始 volume（特征是"÷窗末值"的比值，前/后复权在比值下等价）
  BARRA(T,J,10) CNE5 暴露（时间轴前向填充，暴露日频缓变）
  RET(T,J)   单日收益 = 前向填充后复权 close 的日环比 − 1（停牌=0）
  VALID(T,J) 样本准入 = 非ST ∧ 当日volume>0 ∧ 40日输入窗行情完整 ∧ 当日Barra新鲜
  LABEL_OK(T,J) 未来 horizon 日行情完整（DRM 20日标签可用；推理不需要）
  OBS(T,J)   原始有行情行（停牌日也有行，缺行=空洞/未上市/退市）
数据源 = data/raw_tables/（随交付体自带）。
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CFG, DERIVED, RAW_TABLES

MM_INPUT = RAW_TABLES


def _year_ge(path, y0):
    return int(os.path.basename(path).split('_')[-1].split('.')[0]) >= y0


def listed_ok(OBS, min_days=None):
    """新股过滤 (T,J) bool：上市满 min_days 个交易日才准入。

    研报 p6 明确「剔除新股, ST, 涨停股票后」计多头组合收益；ST 由 VALID 剔、涨停由 BUYABLE 剔，
    新股按上市满 60 个交易日准入。OBS=原始有行情行（停牌日也有行），
    其沿时间轴累计数即「上市以来的交易日数」，故无需额外的上市日期表。
    只作用于 alpha 池(MASTER_POOL)与标签排序池；**不改 VALID**，DRM 风险模型仍用最宽截面、无需重训。
    """
    md = CFG['min_listed_days'] if min_days is None else min_days
    return np.cumsum(OBS, axis=0) > md


def build_base(force=False):
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / 'base.npz'
    if out.exists() and not force:
        print('base.npz 已存在，跳过。如需重建加 force=True。', flush=True)
        return out
    ds = CFG['data_start']

    # ---- 行情：后复权 OHLC + 原始 volume ----
    sd = pd.concat([pd.read_parquet(p, columns=['date', 'code', 'open', 'high', 'low', 'close',
                                                'volume', 'adj_factor', 'tradestatus'])
                    for p in sorted(glob.glob(str(MM_INPUT / 'stock_daily' / 'stock_daily_2*.pq')))
                    if _year_ge(p, ds)], ignore_index=True)
    piv = lambda c: sd.pivot_table(index='date', columns='code', values=c, aggfunc='first')
    adj = piv('adj_factor')
    mats = {c: piv(c) * adj for c in ['open', 'high', 'low', 'close']}   # 后复权连续价
    mats['volume'] = piv('volume')
    DATES, CODES = mats['close'].index, mats['close'].columns.values
    OBS = (~mats['close'].isna()).values                                # 原始有行情行
    close_ff = mats['close'].ffill()
    RET = (close_ff / close_ff.shift(1) - 1).fillna(0).values.astype('float32')
    PRICE = np.stack([mats[c].ffill().values if c != 'volume' else mats[c].fillna(0).values
                      for c in ['open', 'high', 'low', 'close', 'volume']], axis=-1).astype('float32')

    # ---- Barra CNE5 十风格暴露（时间轴限时前向填充）----
    bar = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(MM_INPUT / 'barra_exposure' / '*_2*.pq')))
                     if _year_ge(p, ds)], ignore_index=True)
    fcols = ['beta', 'btop', 'earnyild', 'growth', 'leverage', 'liquidty', 'momentum', 'resvol', 'size', 'sizenl']
    BARRA = np.stack([bar.pivot_table(index='date', columns='code', values=c, aggfunc='first')
                         .reindex(index=DATES, columns=CODES)
                         .ffill(limit=CFG['barra_ffill_limit']).values for c in fcols],
                     axis=-1).astype('float32')                         # 当日非NaN即代表≤5日内有真值

    trade = (mats['volume'].fillna(0).values > 0)                       # 可交易=当日有成交

    # ---- 窗口完整性（以 OBS 原始缺行为准，不被 ffill 伪造）----
    T, J = RET.shape
    cum = np.zeros((T + 1, J), dtype='int32')
    cum[1:] = np.cumsum(OBS, axis=0)
    W, H = CFG['window'], CFG['horizon']
    in_ok = np.zeros_like(trade)
    in_ok[W - 1:] = (cum[W:] - cum[:-W]) == W                           # [t-39,t] 40日全有行情
    LABEL_OK = np.zeros_like(trade)
    LABEL_OK[:T - H] = (cum[H + 1: T + 1] - cum[1: T - H + 1]) == H      # (t,t+H] 全有行情

    # ---- ST 区间矩阵（PIT [进入,撤销) 左闭右开）----
    IS_ST = np.zeros_like(trade)
    if CFG['excl_st']:
        stt = pd.read_parquet(MM_INPUT / 'stock_st' / 'stock_st_all.pq')
        c2j = {c: j for j, c in enumerate(CODES)}
        for code_, e_, r_ in zip(stt['code'], stt['ENTRY_DT'], stt['REMOVE_DT']):
            j = c2j.get(code_)
            if j is None:
                continue
            i0 = DATES.searchsorted(pd.to_datetime(str(e_), format='%Y%m%d'))
            i1 = DATES.searchsorted(pd.to_datetime(str(r_), format='%Y%m%d')) if pd.notna(r_) else T
            IS_ST[i0:i1, j] = True

    VALID = (trade & in_ok & ~np.isnan(BARRA).any(-1) & ~IS_ST)
    BARRA = np.nan_to_num(BARRA, nan=0.0)                               # 残缺暴露=截面中性0

    np.savez(out, PRICE=PRICE, BARRA=BARRA, RET=RET, VALID=VALID, LABEL_OK=LABEL_OK, OBS=OBS,
             DATES=DATES.values, CODES=CODES)
    print(f'base.npz OK: T={T} 全市场J={J} 全A日均VALID={int(VALID.sum(1).mean())} '
          f'起={pd.Timestamp(DATES[0]).date()} 止={pd.Timestamp(DATES[-1]).date()} -> {out}', flush=True)
    return out


if __name__ == '__main__':
    build_base(force='--force' in sys.argv)

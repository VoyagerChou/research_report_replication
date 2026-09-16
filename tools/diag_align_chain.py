# -*- coding: utf-8 -*-
"""诊断：周期链（简化评估器口径）复算，与 README 锚点、summary.csv、日频模拟对账。

用途：验证 evaluate/report_align.py 的数据映射与口径理解正确（链复算应与
简化评估器 summary.csv 逐年一致）。用法：python tools/diag_align_chain.py [tag]
（tag 对应 results/report_align_{tag}/daily_series.csv，默认 nofee_gate）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else 'nofee_gate'
SER_DIR = ROOT / ('results/report_align' + (f'_{TAG}' if TAG else ''))
panel = np.load(ROOT / 'data/derived/panel.npz', allow_pickle=True)
aux = np.load(ROOT / 'data/derived/backtest_aux.npz', allow_pickle=True)
dates = pd.DatetimeIndex(panel['DATES'])
codes = pd.Index(panel['CODES'])
vwap = panel['VWAP_H'].astype(np.float64)
close = panel['PRICE'][..., 3].astype(np.float64)
buy = aux['BUYABLE']
sell = aux['SELLABLE']
bench = aux['BENCH_allmkt'].astype(np.float64)
d2i = {d: i for i, d in enumerate(dates)}

sc = pd.concat([pd.read_parquet(f, columns=['date', 'code', 'score'])
                for f in sorted((ROOT / 'data/factor_values/meta_master_score').glob('*.parquet'))],
               ignore_index=True)
sc['date'] = pd.to_datetime(sc['date'])
smap = {}
for d, g in sc.groupby('date'):
    t = d2i.get(d)
    if t is None:
        continue
    jj = codes.get_indexer(g['code'].values)
    ok = jj >= 0
    smap[t] = (jj[ok], g['score'].to_numpy(np.float64)[ok])

sig = []
for y in range(2019, 2027):
    fr = pd.date_range(f'{y}-01-01', f'{y+1}-01-01', freq='2W-FRI')
    sig += [d for d in fr if d in set(dates)]
sig = [d2i[d] for d in sorted(set(sig)) if d2i[d] in smap]
T = vwap.shape[0]

rows = []
for t in sig:
    if t + 11 >= T:
        continue
    jj, s = smap[t]
    r10 = vwap[t + 11][jj] / vwap[t + 1][jj] - 1.0
    order = np.argsort(-s, kind='stable')
    n1 = max(1, int(len(order) * 0.10))
    g1 = order[:n1]
    fin = np.isfinite(r10[g1])
    r_nog = float(np.mean(r10[g1][fin])) if fin.any() else np.nan
    gb = g1[buy[t + 1][jj[g1]] & fin]
    r_gate = float(np.mean(r10[gb])) if len(gb) else np.nan
    # 卖出腿(与链无关, 仅统计)
    rows.append((dates[t], r_nog, r_gate, len(g1), int(buy[t + 1][jj[g1]].sum())))
df = pd.DataFrame(rows, columns=['date', 'r_nog', 'r_gate', 'n_g1', 'n_buyable'])
df['year'] = df['date'].dt.year

print('==== 周期链（vwap(t+1)->vwap(t+11), 每期等权）====')
print('year  K   chain_nogate  chain_gate')
for y, g in df.groupby('year'):
    a = np.prod(1 + g['r_nog'].fillna(0)) ** (26.0 / len(g)) - 1
    b = np.prod(1 + g['r_gate'].fillna(0)) ** (26.0 / len(g)) - 1
    print(f'{y}  {len(g):3d}   {a:+.4f}      {b:+.4f}')

ref = {2019: 0.3773, 2020: 0.3052, 2021: 0.5055, 2022: 0.0792, 2023: 0.3102,
       2024: 0.0502, 2025: 0.9137, 2026: 0.0612}
print('\n==== summary.csv 的 g1_ann（简化评估器输出）====')
for y, v in ref.items():
    print(y, f'{v:+.4f}')

# 我的日频模拟
ser = pd.read_csv(SER_DIR / 'daily_series.csv', parse_dates=['date'])
ser['year'] = ser['date'].dt.year
print('\n==== 我的日频模拟（含费/拆腿）====')
for y, g in ser.groupby('year'):
    cum = float(np.prod(1 + g['r_p']) - 1)
    ex = float(np.prod(1 + g['excess']) - 1)
    print(f'{y}  n={len(g):3d}  组合={cum:+.4f}  超额={ex:+.4f}')

print('\n==== 逐期对照（前12期: 链 gate vs 日频期段）====')
# 日频: 每个信号 t 的执行日 t+1 到下一执行日 t+11 的累计收益
for i, t in enumerate(sig[:12]):
    if t + 11 >= T:
        break
    m = (ser['date'] >= dates[t + 1]) & (ser['date'] < dates[t + 11])
    seg = ser[m]
    cum = float(np.prod(1 + seg['r_p']) - 1) if len(seg) else np.nan
    print(f'{dates[t].date()}  链gate={df.iloc[i]["r_gate"]:+.4f}  链nog={df.iloc[i]["r_nog"]:+.4f}  日频段={cum:+.4f}')

# 全区间链
a_all = np.prod(1 + df['r_nog'].fillna(0)) ** (26.0 / len(df)) - 1
b_all = np.prod(1 + df['r_gate'].fillna(0)) ** (26.0 / len(df)) - 1
print(f'\n全区间链(2019-2026): nogate 年化={a_all:+.4f}  gate 年化={b_all:+.4f}  (K={len(df)})')

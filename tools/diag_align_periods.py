# -*- coding: utf-8 -*-
"""诊断：逐期对账——模拟器换仓时点价值（vwap 口径）vs 周期链。

用途：定位日频模拟与周期链在个别期的差异（如节假日持仓结构、涨跌停冻结）。
用法：python tools/diag_align_periods.py [tag]
（tag 对应 results/report_align_{tag}/period_values.csv，默认 nofee_gate）
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
dates = pd.DatetimeIndex(panel['DATES'])
codes = pd.Index(panel['CODES'])
vwap = panel['VWAP_H'].astype(np.float64)
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

chain = []
for t in sig:
    if t + 11 >= T:
        continue
    jj, s = smap[t]
    r10 = vwap[t + 11][jj] / vwap[t + 1][jj] - 1.0
    order = np.argsort(-s, kind='stable')
    n1 = max(1, int(len(order) * 0.10))
    g1 = order[:n1]
    fin = np.isfinite(r10[g1])
    chain.append((dates[t], float(np.mean(r10[g1][fin]))))
ch = pd.DataFrame(chain, columns=['signal_date', 'r_chain'])

pv = pd.read_csv(SER_DIR / 'period_values.csv', parse_dates=['date'])
# period k return = V[k+1]/V[k] - 1; V 序列对应信号 k 的执行日（首期 V=0，跳过）
r_sim = pv['v_vwap'].shift(-1) / pv['v_vwap'] - 1.0
cmp = pd.DataFrame({
    'signal_date': ch['signal_date'].iloc[1:len(pv) - 1].values,
    'r_chain': ch['r_chain'].iloc[1:len(pv) - 1].values,
    'r_sim': r_sim.iloc[1:len(pv) - 1].values,
})
cmp['diff'] = cmp['r_sim'] - cmp['r_chain']
print('前 14 期:')
print(cmp.head(14).to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
print('\n差异统计: mean=%.4f median=%.4f |max|=%.4f' % (cmp['diff'].mean(), cmp['diff'].median(), cmp['diff'].abs().max()))
print('\n最大差异的 10 期:')
print(cmp.reindex(cmp['diff'].abs().sort_values(ascending=False).index).head(10).to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
p1 = float(np.prod(1 + cmp['r_chain']))
p2 = float(np.prod(1 + cmp['r_sim']))
print('\n全期累计: chain=%.4f  sim=%.4f  比值=%.4f' % (p1 - 1, p2 - 1, p2 / p1))

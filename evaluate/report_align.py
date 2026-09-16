# -*- coding: utf-8 -*-
"""研报口径对齐回测（重建版，自包含；不依赖 evaluate/ 其余模块）。

口径（见 指标计算口径.md / README §5）:
  · 组合: 全A池(MASTER_POOL)中有 score 的股票, 按 score 降序分十组, G1=最高分组等权满仓
  · 信号: 双周(隔周周五), T+1 按 vwap 成交, 涨停不可买(BUYABLE)/跌停冻结(SELLABLE)
  · 日频核算: 收盘盯市; 换仓日拆腿: 旧仓 close(t)->vwap(t+1) 卖出 + 新仓 vwap(t+1)->close(t+1) 买入
  · 基准: 中证800 (BENCH_allmkt); 超额(算术)=r_p-r_b 日频复利; 超额(比值)=NAV_p/NAV_b
  · 分年行=区间累计(不年化); 全窗口行=CAGR(243); IR=mean(ex)/std(ex)*sqrt(243); 夏普 rf=0
口径开关（环境变量，默认即定稿）:
  · RA_FEE=0.0015（净额计费）/ 0；RA_GATES=1/0；RA_TAG=<tag> 产物目录后缀
自检锚点（README §0；无费+门口径复算，残差 ~1pp）:
  · 至2025-01-06: 复现 23.5/18.6/0.99 vs README 24.5/19.5/1.10
  · 至2025末: 30.4/21.7/1.14 vs 29.5/20.9/1.17 | 全区间: 24.9/17.1/0.92 vs 25.6/16.8/0.95
  · 周频 RankIC 0.1139 / ICIR 5.67 vs README 0.1145 / 5.66（均为至2025-01-06 窗口）
用法: python main.py align     （或项目根目录下: python evaluate/report_align.py）
产物: results/report_align{_TAG}/{daily_series.csv, anchor_check.csv, yearly_compare.csv, ...}
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / 'data' / 'derived' / 'panel.npz'
AUX_PATH = ROOT / 'data' / 'derived' / 'backtest_aux.npz'
SCORE_DIR = ROOT / 'data' / 'factor_values' / 'meta_master_score'
TAG = os.environ.get('RA_TAG', '')
OUT_DIR = ROOT / 'results' / ('report_align' + (f'_{TAG}' if TAG else ''))

FEE = float(os.environ.get('RA_FEE', '0.0015'))   # 单边费率（双边千三）
GATES = os.environ.get('RA_GATES', '1') == '1'    # 是否启用涨跌停门
ANN = 243.0           # 日频年化基数
TOP_FRAC = 0.10       # G1 = top 10%
SIGNAL_GRID = 'per_year'   # 'per_year'（v1）| 'continuous'

# ------------------------------------------------------------------ 数据


def load_all():
    t0 = time.time()
    panel = np.load(PANEL_PATH, allow_pickle=True)
    aux = np.load(AUX_PATH, allow_pickle=True)
    dates = pd.DatetimeIndex(panel['DATES'])
    codes = pd.Index(panel['CODES'])
    close = panel['PRICE'][..., 3].astype(np.float64)
    vwap = panel['VWAP_H'].astype(np.float64)
    pool = panel['MASTER_POOL']
    buyable = aux['BUYABLE']
    sellable = aux['SELLABLE']
    bench = aux['BENCH_allmkt'].astype(np.float64)
    parts = [pd.read_parquet(f, columns=['date', 'code', 'score'])
             for f in sorted(SCORE_DIR.glob('*.parquet'))]
    sc = pd.concat(parts, ignore_index=True)
    sc['date'] = pd.to_datetime(sc['date'])
    print(f'[load] panel/aux/scores in {time.time() - t0:.1f}s, score rows={len(sc):,}', flush=True)
    return dict(dates=dates, codes=codes, close=close, vwap=vwap, pool=pool,
                buyable=buyable, sellable=sellable, bench=bench, scores=sc)


def build_score_map(D):
    d2i = {d: i for i, d in enumerate(D['dates'])}
    codes = D['codes']
    smap = {}
    for d, g in D['scores'].groupby('date'):
        t = d2i.get(d)
        if t is None:
            continue
        jj = codes.get_indexer(g['code'].values)
        ok = jj >= 0
        if ok.any():
            smap[t] = (jj[ok], g['score'].to_numpy(np.float64)[ok])
    return smap, d2i


def build_signals(D, smap, d2i):
    dset = set(D['dates'])
    if SIGNAL_GRID == 'per_year':
        sigs = []
        for y in range(2019, 2027):
            fr = pd.date_range(f'{y}-01-01', f'{y + 1}-01-01', freq='2W-FRI')
            sigs += [d for d in fr if d in dset]
        sigs = sorted(set(sigs))
    else:
        fr = pd.date_range('2019-01-01', '2027-01-01', freq='2W-FRI')
        sigs = sorted(d for d in fr if d in dset)
    return [d2i[d] for d in sigs if d2i[d] in smap]


# ------------------------------------------------------------------ 组合模拟


def simulate(D, smap, signals, t_end):
    dates = D['dates']
    close, vwap = D['close'], D['vwap']
    buyable, sellable, bench = D['buyable'], D['sellable'], D['bench']
    if not GATES:
        buyable = np.ones_like(buyable)
        sellable = np.ones_like(sellable)
    T, J = close.shape

    exec_map = {}
    for t in signals:
        if t + 1 <= t_end:
            exec_map[t + 1] = t
    first_exec = min(exec_map)

    val = np.zeros(J)
    cash = 1.0
    rows = []
    turn = []
    vdiag = []
    for d in range(first_exec, t_end + 1):
        cp, cd, vd = close[d - 1], close[d], vwap[d]
        prev_total = float(val.sum()) + cash
        if d in exec_map:
            t = exec_map[d]
            w_before = val / prev_total
            new_val = np.zeros(J)
            held = val > 0
            sold_pre = 0.0
            mf = np.zeros(J, dtype=bool)
            if held.any():
                ok_sell = sellable[d] & np.isfinite(vd) & (vd > 0) & np.isfinite(cp) & (cp > 0)
                m = held & ok_sell
                if m.any():
                    sold_pre = float(np.sum(val[m] * vd[m] / cp[m]))
                    cash += sold_pre
                mf = held & ~ok_sell
                if mf.any():
                    ratio = np.ones(J)
                    good = mf & np.isfinite(cd) & np.isfinite(cp) & (cp > 0) & (cd > 0)
                    ratio[good] = cd[good] / cp[good]
                    new_val = np.where(mf, val * ratio, 0.0)
            vdiag.append((d, dates[d], sold_pre + float(new_val.sum())))
            jj, sc = smap[t]
            order = np.argsort(-sc, kind='stable')
            n1 = max(1, int(len(order) * TOP_FRAC))
            g1 = jj[order[:n1]]
            okb = buyable[d][g1] & np.isfinite(vd[g1]) & (vd[g1] > 0) & np.isfinite(cd[g1]) & (cd[g1] > 0)
            tgt = g1[okb]
            tgt = tgt[~mf[tgt]]
            if len(tgt) > 0 and cash > 1e-12:
                # 净额计费：fee = 单边费率 × Σ|目标权重 − 漂移后权重| × V（冻结仓不参与交易）
                w_old = val / prev_total
                w_new = new_val / prev_total
                w_new[tgt] += (cash / prev_total) / len(tgt)
                w_trade = w_new.copy()
                if mf.any():
                    w_trade[mf] = w_old[mf]
                dw = float(np.abs(w_trade - w_old).sum())
                fee = FEE * dw * prev_total
                cash = cash - fee
                alloc = cash / len(tgt)
                new_val[tgt] += alloc * cd[tgt] / vd[tgt]
                cash = 0.0
            val = new_val
            total = float(val.sum()) + cash
            w_after = val / total if total > 0 else val
            turn.append((dates[d], float(np.abs(w_after - w_before).sum()) / 2.0))
        else:
            ratio = np.ones(J)
            good = (val > 0) & np.isfinite(cd) & np.isfinite(cp) & (cp > 0) & (cd > 0)
            ratio[good] = cd[good] / cp[good]
            val = val * ratio
            total = float(val.sum()) + cash
        rp = total / prev_total - 1.0 if prev_total > 0 else 0.0
        rb = bench[d] / bench[d - 1] - 1.0
        rows.append((dates[d], rp, rb))

    ser = pd.DataFrame(rows, columns=['date', 'r_p', 'r_b'])
    ser['excess'] = ser['r_p'] - ser['r_b']
    ser['nav_p'] = (1 + ser['r_p']).cumprod()
    ser['nav_b'] = (1 + ser['r_b']).cumprod()
    ser['nav_ex'] = (1 + ser['excess']).cumprod()
    turn_df = pd.DataFrame(turn, columns=['date', 'tv_half'])
    vdiag_df = pd.DataFrame(vdiag, columns=['d', 'date', 'v_vwap'])
    return ser, turn_df, vdiag_df


# ------------------------------------------------------------------ 指标


def metrics(ser, annualize):
    rp = ser['r_p'].to_numpy()
    rb = ser['r_b'].to_numpy()
    ex = rp - rb
    n = len(rp)
    navp = float(np.prod(1 + rp))
    navb = float(np.prod(1 + rb))
    navex = float(np.prod(1 + ex))
    k = ANN / n
    if annualize:
        abs_ret = navp ** k - 1
        ex_arith = navex ** k - 1
        ex_ratio = (navp / navb) ** k - 1
    else:
        abs_ret = navp - 1
        ex_arith = navex - 1
        ex_ratio = navp / navb - 1
    ex_vol = float(np.std(ex, ddof=1) * np.sqrt(ANN))
    ir = float(np.mean(ex) / (np.std(ex, ddof=1) + 1e-12) * np.sqrt(ANN))
    sharpe = float(np.mean(rp) / (np.std(rp, ddof=1) + 1e-12) * np.sqrt(ANN))
    nav_ex = np.cumprod(1 + ex)
    mdd = float(np.min(nav_ex / np.maximum.accumulate(nav_ex) - 1))
    return dict(n=n, abs_ret=abs_ret, ex_arith=ex_arith, ex_ratio=ex_ratio,
                ex_vol=ex_vol, ir=ir, sharpe=sharpe, ex_mdd=mdd)


def _rankdata(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1)
    return r


def _spearman(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def rankic_series(D, smap, d2i, start_dt, end_dt):
    vwap = D['vwap']
    T = vwap.shape[0]
    idx = pd.DatetimeIndex(D['dates'])
    week_last = idx.to_series().groupby(idx.to_period('W-FRI')).last()
    rows = []
    for wd in week_last:
        if wd < start_dt or wd > end_dt:
            continue
        t = d2i.get(wd)
        if t is None or t + 11 >= T or t not in smap:
            continue
        jj, sc = smap[t]
        r10 = vwap[t + 11][jj] / vwap[t + 1][jj] - 1.0
        ok = np.isfinite(r10) & np.isfinite(sc)
        if ok.sum() >= 10:
            rows.append((wd, _spearman(sc[ok], r10[ok])))
    return pd.DataFrame(rows, columns=['date', 'rankic'])


def long_short(D, smap, signals, start_dt, end_dt):
    vwap = D['vwap']
    T = vwap.shape[0]
    diffs = []
    for t in signals:
        wd = D['dates'][t]
        if wd < start_dt or wd > end_dt:
            continue
        if t + 11 >= T or t not in smap:
            continue
        jj, sc = smap[t]
        r10 = vwap[t + 11][jj] / vwap[t + 1][jj] - 1.0
        ok = np.isfinite(r10)
        if ok.sum() < 100:
            continue
        jj2, sc2, r2 = jj[ok], sc[ok], r10[ok]
        order = np.argsort(-sc2, kind='stable')
        n1 = max(1, int(len(order) * TOP_FRAC))
        diffs.append(float(np.mean(r2[order[:n1]]) - np.mean(r2[order[-n1:]])))
    k = len(diffs)
    ls_ann = float(np.prod(1 + np.array(diffs)) ** (26.0 / k) - 1) if k else np.nan
    return ls_ann, k


# ------------------------------------------------------------------ 主流程


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    D = load_all()
    smap, d2i = build_score_map(D)
    signals = build_signals(D, smap, d2i)
    print(f'[signals] {len(signals)} 个信号, 首个 {D["dates"][signals[0]].date()}, '
          f'末个 {D["dates"][signals[-1]].date()}', flush=True)

    t_end = len(D['dates']) - 1
    ser, turn_df, vdiag_df = simulate(D, smap, signals, t_end)
    print(f'[sim] 日序列 {len(ser)} 天, {ser["date"].iloc[0].date()} -> {ser["date"].iloc[-1].date()}', flush=True)

    ser.to_csv(OUT_DIR / 'daily_series.csv', index=False, encoding='utf-8-sig')
    turn_df.to_csv(OUT_DIR / 'turnover_periods.csv', index=False, encoding='utf-8-sig')
    vdiag_df.to_csv(OUT_DIR / 'period_values.csv', index=False, encoding='utf-8-sig')

    # ---- 锚点自检 ----
    def win(a, b):
        m = (ser['date'] >= pd.Timestamp(a)) & (ser['date'] <= pd.Timestamp(b))
        return ser[m]

    checks = []

    def add(anchor, metric, comp, pub, tol):
        checks.append(dict(anchor=anchor, metric=metric, computed=round(comp, 6),
                           published=pub, diff=round(comp - pub, 6), tol=tol,
                           result='PASS' if abs(comp - pub) <= tol else 'FAIL'))

    for name, a, b, tgt in [
        ('至2025-01-06', '2019-01-01', '2025-01-06',
         {'绝对收益': (0.245, 0.003), '超额(算术)': (0.1950, 0.003), 'IR': (1.10, 0.05)}),
        ('至2025年末', '2019-01-01', '2025-12-31',
         {'绝对收益': (0.295, 0.003), '超额(算术)': (0.209, 0.003), 'IR': (1.17, 0.05)}),
        ('全区间(至2026-07-20)', '2019-01-01', '2026-07-20',
         {'绝对收益': (0.256, 0.003), '超额(算术)': (0.168, 0.003), 'IR': (0.95, 0.05),
          '超额MDD': (-0.321, 0.01)}),
    ]:
        m = metrics(win(a, b), annualize=True)
        for metric, (pub, tol) in tgt.items():
            key = {'绝对收益': 'abs_ret', '超额(算术)': 'ex_arith', 'IR': 'ir', '超额MDD': 'ex_mdd'}[metric]
            add(name, metric, m[key], pub, tol)

    ric = rankic_series(D, smap, d2i, pd.Timestamp('2019-01-01'), pd.Timestamp('2026-07-20'))
    add('全区间', '周频RankIC', float(ric['rankic'].mean()), 0.1145, 0.002)
    add('全区间', '年化ICIR', float(ric['rankic'].mean() / ric['rankic'].std(ddof=1) * np.sqrt(52)), 5.66, 0.20)
    ls_ann, k_ls = long_short(D, smap, signals, pd.Timestamp('2019-01-01'), pd.Timestamp('2026-07-20'))
    add('全区间', '多空年化', ls_ann, 0.736, 0.015)

    chk = pd.DataFrame(checks)
    chk.to_csv(OUT_DIR / 'anchor_check.csv', index=False, encoding='utf-8-sig')
    print('\n==== 锚点自检 ====')
    print(chk.to_string(index=False), flush=True)
    print(f'[rankic] 周数={len(ric)}, 均值={ric["rankic"].mean():.4f} | [多空] 期数={k_ls}, 年化={ls_ann:.4f}', flush=True)

    # ---- 分年对比表 ----
    REPORT = {
        '2019': (0.400, 0.130, 1.04, 1.38, -0.046),
        '2020': (0.440, 0.209, 1.96, 1.80, -0.046),
        '2021': (0.331, 0.393, 1.29, 1.76, -0.106),
        '2022': (0.256, 0.563, 3.07, 0.98, -0.066),
        '2023': (0.272, 0.440, 3.30, 1.46, -0.046),
        '2024~2025/1/6': (0.366, 0.226, 0.96, 1.00, -0.169),
    }
    ROWS = [
        ('2019', '2019-01-01', '2019-12-31', False),
        ('2020', '2020-01-01', '2020-12-31', False),
        ('2021', '2021-01-01', '2021-12-31', False),
        ('2022', '2022-01-01', '2022-12-31', False),
        ('2023', '2023-01-01', '2023-12-31', False),
        ('2024~2025/1/6', '2024-01-01', '2025-01-06', False),
        ('2019~2025/1/6(全窗口)', '2019-01-01', '2025-01-06', True),
        ('2024', '2024-01-01', '2024-12-31', False),
        ('2025', '2025-01-01', '2025-12-31', False),
        ('2026(至07-20)', '2026-01-01', '2026-07-20', False),
        ('2019~2026(全区间)', '2019-01-01', '2026-07-20', True),
    ]
    out = []
    for name, a, b, ann in ROWS:
        m = metrics(win(a, b), annualize=ann)
        rep = REPORT.get(name)
        out.append(dict(
            window=name, n_days=m['n'],
            复现_组合收益=round(m['abs_ret'], 4), 复现_超额算术=round(m['ex_arith'], 4),
            复现_超额比值=round(m['ex_ratio'], 4), 复现_IR=round(m['ir'], 2),
            复现_夏普=round(m['sharpe'], 2), 复现_超额MDD=round(m['ex_mdd'], 4),
            研报_组合收益=rep[0] if rep else None, 研报_超额=rep[1] if rep else None,
            研报_IR=rep[2] if rep else None, 研报_夏普=rep[3] if rep else None,
            研报_超额MDD=rep[4] if rep else None))
    tab = pd.DataFrame(out)
    tab.to_csv(OUT_DIR / 'yearly_compare.csv', index=False, encoding='utf-8-sig')
    print('\n==== 复现 vs 研报 分年对比（研报=最终模型表1，全A多头 vs 中证800）====')
    print(tab.to_string(index=False), flush=True)

    # 换手
    if len(turn_df):
        tv = turn_df['tv_half'].mean() * 26
        print(f'[turnover] 单边年化换手 ~ {tv:.1f} 倍（mean(tv/2)×26）', flush=True)

    print(f'\n产物: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""ABCD 头部区分度评估：自包含，不依赖 evaluate/ 包（该目录文件已在本地被清空）。

直接从「因子落库 parquet + data/derived/{labels,panel}.npz」计算头部指标，
专门回答"模型对全 A top10% / 10~20% 的区分度"：
  RankIC（日频均值 + 周频周五）/ 年化 ICIR（日频×√243）
  十分位收益表（G1..G10 的 Y_RAW 时间加权均值）→ G1−G2、G1−G10
  top10 overlap（预测前10% ∩ 真实收益前10% / 预测前10%，随机基准=0.10）
  NDCG@10%（gain=标签分位）
  TopQ（预测前10% 的平均标签分位；与 config.DA_VAL_METRIC=topq 同口径）
  G1 平均真实分位（头部成色，0.5=无技能）

用法：
  python evaluate_head.py --score-dir data/factor_values/meta_master_score_prk --name prank
  python evaluate_head.py --compare                       # 汇总 results/loss_abcd/ 全部行 → summary.csv
产物：
  results/loss_abcd/{name}_daily.csv    逐日指标（复核用）
  results/loss_abcd/{name}_yearly.csv   分年指标
  results/loss_abcd/{name}_summary.csv  单行汇总（--compare 汇总所有行）
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'results' / 'loss_abcd'


# ------------------------------------------------------------------ 数据装配
def _load_labels():
    """labels.npz 的 Y_RAW/Y_RANK/LBL_OK + panel.npz 的 DATES/CODES。"""
    lab = np.load(ROOT / 'data' / 'derived' / 'labels.npz', allow_pickle=True)
    pan = np.load(ROOT / 'data' / 'derived' / 'panel.npz', allow_pickle=True)
    return lab['Y_RAW'], lab['Y_RANK'], lab['LBL_OK'], pan['DATES'], pan['CODES']


def _load_scores(score_dir):
    """读 score_dir 下全部年度分片 → 长表 (date, code, score)。"""
    files = sorted([p for p in Path(score_dir).glob('*.pq')] +
                   [p for p in Path(score_dir).glob('*.parquet')])
    if not files:
        raise FileNotFoundError(f'{score_dir} 下没有因子分片（*.pq/*.parquet）')
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=['date', 'code', 'score'])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ------------------------------------------------------------------ 逐日指标
def _rank_pct(a):
    """升序百分位（0~1，最大=1），等价 scipy.stats.rankdata(pct=True)，无重复值时精确。"""
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1)
    return r / len(a)


def _day_metrics(s, y_raw, y_rank):
    """单日截面的头部指标。s/y_raw/y_rank 均为 (n,) 已对齐、无 NaN。"""
    n = len(s)
    if n < 50:
        return None
    rk_s = _rank_pct(s)                       # 分数分位（高=好）
    rk_y = _rank_pct(y_raw)                   # 真实收益分位
    # RankIC = Spearman(s, y_raw)：秩的 Pearson
    ic = np.corrcoef(rk_s, rk_y)[0, 1]
    # 十分位（按分数）：bin 0=最低(G10) ... 9=最高(G1)
    dec = np.clip((rk_s * 10).astype(int), 0, 9)
    dec_ret = np.array([y_raw[dec == g].mean() if (dec == g).any() else np.nan
                        for g in range(10)])   # dec_ret[9]=G1, [0]=G10
    top_s = rk_s >= 0.9
    top_y = rk_y >= 0.9
    overlap = float((top_s & top_y).sum() / top_s.sum())
    # NDCG@10%（gain=y_rank）
    k = max(1, int(round(n * 0.1)))
    order_s = np.argsort(-s, kind='mergesort')
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    gains = y_rank[order_s][:k]
    dcg = float((gains * disc).sum())
    idcg = float((np.sort(y_rank)[::-1][:k] * disc).sum())
    ndcg = dcg / idcg if idcg > 0 else np.nan
    topq = float(y_rank[top_s].mean())         # G1 平均标签分位（=G1 平均真实分位）
    return dict(rankic=ic, ndcg10=ndcg, overlap10=overlap, topq=topq,
                g1=dec_ret[9], g2=dec_ret[8], g10=dec_ret[0],
                g1_g2=dec_ret[9] - dec_ret[8], g1_g10=dec_ret[9] - dec_ret[0],
                dec_ret=dec_ret)


def run_eval(score_dir, name):
    y_raw, y_rank, lbl_ok, dates, codes = _load_labels()
    df = _load_scores(score_dir)
    # 对齐：factor 行 → (t, j)
    date_idx = {d: i for i, d in enumerate(dates)}
    code_idx = {c: i for i, c in enumerate(codes)}
    df['_t'] = df['date'].map(date_idx)
    df['_j'] = df['code'].map(code_idx)
    df = df.dropna(subset=['_t', '_j'])
    df['_t'] = df['_t'].astype(int)
    df['_j'] = df['_j'].astype(int)

    rows = []
    dec_rows = []
    for t, g in df.groupby('_t', sort=True):
        j = g['_j'].to_numpy()
        ok = lbl_ok[t, j]
        yr = y_rank[t, j]
        yw = y_raw[t, j]
        m = ok & np.isfinite(yr) & np.isfinite(yw) & np.isfinite(g['score'].to_numpy())
        if m.sum() < 50:
            continue
        r = _day_metrics(g['score'].to_numpy()[m], yw[m], yr[m])
        if r is None:
            continue
        d = dates[t]
        rows.append(dict(date=d, rankic=r['rankic'], ndcg10=r['ndcg10'],
                         overlap10=r['overlap10'], topq=r['topq'],
                         g1=r['g1'], g2=r['g2'], g10=r['g10'],
                         g1_g2=r['g1_g2'], g1_g10=r['g1_g10']))
        dec_rows.append(dict(date=d, **{f'G{i + 1}': r['dec_ret'][9 - i] for i in range(10)}))

    daily = pd.DataFrame(rows)
    dec_daily = pd.DataFrame(dec_rows)
    daily['year'] = pd.to_datetime(daily['date']).dt.year
    daily['_fri'] = pd.to_datetime(daily['date']).dt.weekday == 4
    dec_daily['year'] = pd.to_datetime(dec_daily['date']).dt.year

    # ---- 汇总（全区间 + 分年；不用 groupby.apply，兼容各 pandas 版本）----
    def _agg_series(df):
        fri = df[df['_fri']]
        return pd.Series(dict(
            days=len(df),
            rankic_daily=df['rankic'].mean(),
            rankic_weekly=fri['rankic'].mean() if len(fri) else np.nan,
            icir_annual=df['rankic'].mean() / (df['rankic'].std() + 1e-12) * np.sqrt(243),
            g1=df['g1'].mean(), g2=df['g2'].mean(), g10=df['g10'].mean(),
            g1_g2=df['g1'].mean() - df['g2'].mean(),
            g1_g10=df['g1'].mean() - df['g10'].mean(),
            overlap10=df['overlap10'].mean(),
            ndcg10=df['ndcg10'].mean(),
            topq=df['topq'].mean(),
        ))

    yearly_rows = []
    for y, gdf in daily.groupby('year'):
        d = _agg_series(gdf).to_dict(); d['year'] = int(y)
        yearly_rows.append(d)
    full = _agg_series(daily)
    d = full.to_dict(); d['year'] = 'ALL'
    yearly_rows.append(d)
    yearly = pd.DataFrame(yearly_rows)
    yearly = yearly[['year'] + [c for c in yearly.columns if c != 'year']]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DIR / f'{name}_daily.csv', index=False)
    dec_daily.to_csv(OUT_DIR / f'{name}_decile_daily.csv', index=False)
    yearly.to_csv(OUT_DIR / f'{name}_yearly.csv', index=False)
    summ = full.to_frame().T
    summ.insert(0, 'name', name)
    summ.insert(1, 'score_dir', str(score_dir))
    summ.to_csv(OUT_DIR / f'{name}_summary.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'\n=== [{name}] 头部指标（全区间）===')
    print(summ.drop(columns=['score_dir']).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n分年 G1-G2 / overlap10 / NDCG@10% / topq：')
    cols = ['year', 'g1_g2', 'g1_g10', 'overlap10', 'ndcg10', 'topq', 'rankic_daily']
    print(yearly[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n产物: {OUT_DIR}/{name}_{{daily,yearly,summary}}.csv')
    return summ


def compare():
    rows = []
    for f in sorted(OUT_DIR.glob('*_summary.csv')):
        rows.append(pd.read_csv(f))
    if not rows:
        print(f'{OUT_DIR} 下没有 *_summary.csv')
        return
    tab = pd.concat(rows, ignore_index=True)
    tab = tab.drop_duplicates(subset=['name'], keep='last')
    tab.to_csv(OUT_DIR / 'summary.csv', index=False)
    pd.set_option('display.width', 240)
    show = ['name', 'rankic_daily', 'rankic_weekly', 'icir_annual', 'g1_g2', 'g1_g10',
            'overlap10', 'ndcg10', 'topq', 'g1', 'g2', 'g10']
    print('\n=== ABCD 横向对照（全区间）===')
    print(tab[show].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n汇总: {OUT_DIR / "summary.csv"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-dir', type=str, default=None)
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()
    if args.compare:
        compare()
        return
    if not args.score_dir or not args.name:
        ap.error('需要 --score-dir 与 --name（或用 --compare）')
    run_eval(args.score_dir, args.name)


if __name__ == '__main__':
    main()

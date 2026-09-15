# -*- coding: utf-8 -*-
"""数据准备编排 + 全A面板装配。

从原始 input_data 全流程自建共享面板。由 `python main.py prepare` 调起，prepare 阶段链：
  build_base(行情/VALID/Barra) → build_tech53(53因子) → build_idx63(63指数态) →
  build_labels(全A标签) → build_drm(DRM训练+推理+expo+risk120→183市场状态) →
  assemble_panel(全A面板,不瘦身) → build_backtest_aux(涨跌停 + 三池基准/成分)
其中 build_drm 是 GPU 重活（12 个年度模型 × 全市场），服务器上跑。
数据源 = data/raw_tables/（原始表随交付体自带；更换数据源只改各 build_* 的读取层）。
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CFG, DERIVED, RAW_TABLES, POOLS, ALLMKT, PANEL_PATH, EXPO_PATH, MSTATE_PATH
from input_data_update.build_base import build_base, listed_ok
from input_data_update.build_tech53 import build_tech53
from input_data_update.build_idx63 import build_idx63
from input_data_update.build_labels import build_labels
from input_data_update import build_drm

MM_INPUT = RAW_TABLES


def _pit_member(im_index, DATES, CODES):
    """单指数 PIT 成分掩码 (T,J) bool：[纳入日,剔除日]（S_CON_OUTDATE 为最后有效日）。"""
    T, J = len(DATES), len(CODES)
    member = np.zeros((T, J), bool)
    c2j = {c: j for j, c in enumerate(CODES)}
    for code, indate, outdate in zip(im_index['code'], im_index['S_CON_INDATE'], im_index['S_CON_OUTDATE']):
        j = c2j.get(code)
        if j is None:
            continue
        i0 = int(np.searchsorted(DATES, pd.to_datetime(str(indate), format='%Y%m%d')))
        i1 = int(np.searchsorted(DATES, pd.to_datetime(str(outdate), format='%Y%m%d'), side='right')) \
            if pd.notna(outdate) else T
        member[i0:i1, j] = True
    return member


def assemble_panel(force=False):
    """装配全A面板 panel.npz（不瘦身，J=全市场）。从 P1/P2 产物拼装，MASTER_POOL=VALID。"""
    out = PANEL_PATH
    if out.exists() and not force:
        print(f'{out.name} 已存在，跳过。如需重建加 --force。', flush=True)
        return out
    base = np.load(DERIVED / 'base.npz', allow_pickle=True)
    DATES, CODES = base['DATES'], base['CODES']
    PRICE, VALID, OBS = base['PRICE'], base['VALID'], base['OBS']
    T, J = VALID.shape
    ta = np.load(DERIVED / 'tech_arrays.npz')
    AMT, VWAP_H = ta['AMT'], ta['VWAP_H']
    from config import LABELS_PATH as _LP
    lab = np.load(_LP)
    Y_RANK, LBL_OK = lab['Y_RANK'], lab['LBL_OK']
    MSTATE = np.load(MSTATE_PATH, allow_pickle=True)['MSTATE']
    assert MSTATE.shape == (T, 183), f'市场状态 {MSTATE.shape}≠{(T, 183)}——先跑 build_drm'
    for f in (DERIVED / 'tech53.dat', EXPO_PATH, DERIVED / 'tech53_stats.npz'):
        assert f.exists(), f'缺 {f.name}——先跑 build_tech53 / build_drm'

    NEW_OK = listed_ok(OBS)                                      # 剔新股（研报 p6；不改 VALID→DRM 不受影响）
    MASTER_POOL = VALID & NEW_OK                                 # alpha 池基底（全市场网格上的掩码,不瘦身矩阵）
    POOL_LBL = MASTER_POOL & LBL_OK                              # 池∩可实现标签
    np.savez(out, DATES=DATES, CODES=CODES, PRICE=PRICE, AMT=AMT, VWAP_H=VWAP_H,
             MASTER_POOL=MASTER_POOL, Y_RANK=Y_RANK, POOL_LBL=POOL_LBL, LBL_OK=LBL_OK,
             MSTATE=MSTATE, tech_shape=np.array([T, J, 53]))
    print(f'{out.name} OK(全A,剔新股{CFG["min_listed_days"]}日): T={T} J={J} '
          f'VALID={int(VALID.sum(1).mean())}→池{int(MASTER_POOL.sum(1).mean())} '
          f'池内标签日均={int(POOL_LBL.sum(1)[POOL_LBL.sum(1) > 0].mean())} -> {out}', flush=True)
    return out


def build_backtest_aux(force=False):
    """回测辅助：涨跌停 BUYABLE/SELLABLE（全市场共享）+ 三池 BENCH_{pool}（指数点位）+ MEMBER_{pool}（PIT成分）。

    对齐 base.npz 的 DATES×CODES。三池评价：一个全A因子 → 筛 MEMBER_{pool} → 池内重排 → 对 BENCH_{pool}。
    """
    aux = DERIVED / 'backtest_aux.npz'
    if aux.exists() and not force:
        print('backtest_aux.npz 已存在，跳过。', flush=True)
        return aux
    base = np.load(DERIVED / 'base.npz', allow_pickle=True)
    DATES, CODES = pd.DatetimeIndex(base['DATES']), base['CODES']

    def _pv(df, col):
        return (df.pivot_table(index='date', columns='code', values=col, aggfunc='first')
                  .reindex(index=DATES, columns=CODES).values)

    sd = pd.concat([pd.read_parquet(p, columns=['date', 'code', 'open', 'limit_up', 'limit_down', 'volume'])
                    for p in sorted(glob.glob(str(MM_INPUT / 'stock_daily' / 'stock_daily_2*.pq')))],
                   ignore_index=True)
    opn, limu, limd = _pv(sd, 'open'), _pv(sd, 'limit_up'), _pv(sd, 'limit_down')
    trd = np.nan_to_num(_pv(sd, 'volume'), nan=0.0) > 0
    save = {'BUYABLE': trd & (opn < limu - 1e-6), 'SELLABLE': trd & (opn > limd + 1e-6)}

    idx = pd.read_parquet(MM_INPUT / 'index_daily' / 'index_daily_all.pq')
    im = pd.read_parquet(MM_INPUT / 'index_members' / 'index_members_all.pq')

    def _bench(code):
        b = idx[idx['code'].eq(code)].set_index('date').sort_index()
        return b['close'].reindex(DATES).ffill().values.astype('float32')

    for pool, code in POOLS.items():                            # 三池：基准点位 + PIT 成分掩码
        save[f'BENCH_{pool}'] = _bench(code)
        save[f'MEMBER_{pool}'] = _pit_member(im[im['index_code'].eq(code)], DATES, CODES)
    save[f'BENCH_{ALLMKT[0]}'] = _bench(ALLMKT[1])              # 全市场评价基准=中证800（池用 panel.MASTER_POOL，无需存掩码）

    np.savez(aux, **save)
    _cov = {p: f"BENCH有效{np.isfinite(save[f'BENCH_{p}']).mean():.0%}/成分日均{int(save[f'MEMBER_{p}'].sum(1)[save[f'MEMBER_{p}'].sum(1) > 0].mean())}"
            for p in POOLS}
    print(f'backtest_aux.npz OK: BUYABLE{save["BUYABLE"].mean():.0%} | ' + ' | '.join(f'{p}:{c}' for p, c in _cov.items())
          + f" | {ALLMKT[0]}:BENCH({ALLMKT[1]})有效{np.isfinite(save[f'BENCH_{ALLMKT[0]}']).mean():.0%}"
          + f' -> {aux}', flush=True)
    return aux


def main(force=False):
    DERIVED.mkdir(parents=True, exist_ok=True)
    build_base(force=force)                                     # P1.1 数据层
    build_tech53(force=force)                                   # P1.2 53因子
    build_idx63(force=force)                                    # P1.3 63指数态
    build_backtest_aux(force=force)                             # 回测辅助(含成分掩码;原生池 labels 依赖它,提前)
    build_labels(force=force)                                   # P1.4 标签(全A rank[0,1])
    build_drm.main()                                            # P2 DRM(GPU重,全市场共享)：训练+推理+expo+183市场状态
    assemble_panel(force=force)                                 # P3 面板装配(全A)


if __name__ == '__main__':
    main(force='--force' in sys.argv)

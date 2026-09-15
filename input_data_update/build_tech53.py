# -*- coding: utf-8 -*-
"""量价派生特征层：AMT/VWAP_H/TURN/SPEC + 53 技术因子 + 逐日全市场统计。

全市场（T,J,53）。因子公式详见 tech53因子计算说明.md。产物落 DERIVED：
  tech_arrays.npz : AMT(成交额,元) / VWAP_H(后复权vwap,ffill) / TURN(换手率%) / SPEC(特质收益,小数)
  tech53.dat      : (T,J,53) float32 memmap，53 个量价/行为/情绪因子（列序=sorted(TECH53)）
  tech53_stats.npz: MU/SD (T,53) 逐日全市场（有值截面）均值/标准差 —— day_batch 的 z 基准
数据源 = data/raw_tables/（随交付体自带）。
"""
import os
import sys
import glob
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CFG, DERIVED, RAW_TABLES

MM_INPUT = RAW_TABLES

TECH53 = sorted([
    'mom_1y', 'mom_3m', 'mom_6m', 'mom_2y', 'mom_1y_1m', 'reverse_1m',
    'specific_mom1', 'specific_mom6', 'specific_mom12', 'up_list', 'down_list', 'ideal_reversal',
    'clo_5d_60d', 'vwap_5d_60d', 'close_max_div_min_1m', 'close_max_div_min_3m', 'close_max_div_min_6m',
    'return_max_1m', 'return_std_1w', 'return_std_1m', 'return_std_3m', 'return_std_6m', 'return_std_12m',
    'amt_1m_3m', 'ln_volume_mean_1m', 'ln_volume_mean_3m', 'ln_volume_mean_6m', 'ln_volume_mean_12m',
    'ln_volume_std_1m', 'ln_volume_std_3m', 'ln_volume_std_6m', 'ln_volume_std_12m',
    'volume_1m_div_12m', 'volume_1m_minus_12m', 'volume_std_1m_div_12m',
    'swap_1m', 'swap_3m', 'swap_1y', 'turnover_mean_1m', 'turnover_mean_3m', 'turnover_mean_6m',
    'turnover_std_1m', 'turnover_std_3m', 'turnover_std_6m',
    'turnover_stdrate_1m', 'turnover_stdrate_3m', 'turnover_stdrate_6m',
    'illiq', 'corr_close_turnover', 'ivr', 'duvol', 'ncskew', 'spread_bias'])
assert len(TECH53) == 53


def build_tech_arrays(DATES, CODES):
    """派生矩阵：AMT/VWAP_H/TURN/SPEC，对齐 DATES×CODES。"""
    ds = CFG['data_start']
    _yge = lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]) >= ds
    sd = pd.concat([pd.read_parquet(p, columns=['date', 'code', 'amount', 'vwap', 'adj_factor'])
                    for p in sorted(glob.glob(str(MM_INPUT / 'stock_daily' / 'stock_daily_2*.pq')))
                    if _yge(p)], ignore_index=True)
    pv = lambda df, c: df.pivot_table(index='date', columns='code', values=c, aggfunc='first') \
                         .reindex(index=DATES, columns=CODES)
    AMT = pv(sd, 'amount').values.astype('float32')                       # 成交额(元)，停牌NaN
    VWAP_H = (pv(sd, 'vwap') * pv(sd, 'adj_factor')).ffill().values.astype('float32')
    dr = pd.concat([pd.read_parquet(p, columns=['date', 'code', 'S_DQ_TURN'])
                    for p in sorted(glob.glob(str(MM_INPUT / 'stock_derivative' / '*_2*.pq')))],
                   ignore_index=True)
    TURN = pv(dr, 'S_DQ_TURN').values.astype('float32')                   # 换手率(%)，仅交易日
    sp = pd.concat([pd.read_parquet(p) for p in
                    sorted(glob.glob(str(MM_INPUT / 'barra_special_returns' / '*_2*.pq')))], ignore_index=True)
    spc = [c for c in sp.columns if sp[c].dtype.kind == 'f'][0]
    SPEC = (pv(sp, spc) / 100.0).values.astype('float32')                 # 特质收益：百分点→小数
    return AMT, VWAP_H, TURN, SPEC


def build_tech53(force=False):
    DERIVED.mkdir(parents=True, exist_ok=True)
    tech_path = DERIVED / 'tech53.dat'
    stats_path = DERIVED / 'tech53_stats.npz'
    ta_path = DERIVED / 'tech_arrays.npz'
    if tech_path.exists() and stats_path.exists() and ta_path.exists() and not force:
        print('tech53 相关产物已存在，跳过。', flush=True)
        return tech_path

    z = np.load(DERIVED / 'base.npz', allow_pickle=True)
    PRICE, RET, OBS = z['PRICE'], z['RET'], z['OBS']
    DATES, CODES = pd.DatetimeIndex(z['DATES']), z['CODES']
    T, J = RET.shape
    CLOSE_H, VOL = PRICE[:, :, 3], PRICE[:, :, 4]

    AMT, VWAP_H, TURN, SPEC = build_tech_arrays(DATES, CODES)
    np.savez(ta_path, AMT=AMT, VWAP_H=VWAP_H, TURN=TURN, SPEC=SPEC)

    def _dfw(a):
        return pd.DataFrame(a, index=DATES, columns=CODES)

    def _rm(a, n):
        return _dfw(a).rolling(n, min_periods=max(2, int(n * 0.8))).mean().values.astype('float32')

    def _rs(a, n):
        return _dfw(a).rolling(n, min_periods=max(2, int(n * 0.8))).std().values.astype('float32')

    tmp = tech_path.with_name(f'{tech_path.name}.building.{os.getpid()}')   # tmp带pid:并发不互相截断
    TECH = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(T, J, 53))
    TECH[:] = np.nan
    put = lambda lab, arr: TECH.__setitem__((slice(None), slice(None), TECH53.index(lab)),
                                            np.asarray(arr, dtype='float32'))
    R = np.where(OBS, RET, np.nan).astype('float32')                      # 缺行NaN纪律：停牌收益0，未上市/退市空洞NaN
    R0 = RET                                                              # 0填充版仅供 spread_bias 相关矩阵

    # ---- momentum 族 ----
    def _mom(n, min_frac=0.8):
        ok = _dfw(OBS.astype('float32')).rolling(n).mean().values >= min_frac
        m = CLOSE_H / np.vstack([np.full((n, J), np.nan, 'float32'), CLOSE_H[:-n]]) - 1
        return np.where(ok, m, np.nan)
    _m252, _m21 = _mom(252), _mom(21)
    put('mom_1y', _m252); put('mom_3m', _mom(63)); put('mom_6m', _mom(126)); put('mom_2y', _mom(504))
    put('reverse_1m', _m21); put('mom_1y_1m', (1 + _m252) / (1 + _m21) - 1)
    for lab, n in [('specific_mom1', 21), ('specific_mom6', 126), ('specific_mom12', 252)]:
        put(lab, np.expm1(_dfw(np.log1p(SPEC)).rolling(n, min_periods=int(n * 0.8)).sum().values))
    _w20 = np.power(0.5, np.arange(19, -1, -1) / 10.0).astype('float32'); _w20 /= _w20.sum()

    def _decay20(ind01):
        acc = np.zeros((T - 19, J), 'float32')
        for k in range(20):
            acc += _w20[k] * ind01[k:T - 19 + k]
        out = np.full((T, J), np.nan, 'float32'); out[19:] = acc
        return out
    put('up_list', _decay20((_dfw(R).rank(axis=1, ascending=False, method='min') <= 80).values.astype('float32')))
    put('down_list', _decay20((_dfw(R).rank(axis=1, ascending=True, method='min') <= 80).values.astype('float32')))
    _ir = np.full((T, J), np.nan, 'float32')
    for _t0 in range(20, T, 250):
        _t1 = min(_t0 + 250, T)
        _aw = np.stack([AMT[s - 20:s + 1] for s in range(_t0, _t1)], 0)   # (b,21,J)
        _rw = np.stack([R[s - 20:s + 1] for s in range(_t0, _t1)], 0)
        _awv = np.where(_aw > 0, _aw, np.nan)
        _bad = np.isnan(_awv).any(1)
        _o = np.argsort(np.nan_to_num(_awv, nan=-1.0), axis=1)
        _sr = np.take_along_axis(_rw, _o, axis=1)
        _v = _sr[:, -10:, :].sum(1) - _sr[:, :10, :].sum(1)
        _v[_bad] = np.nan
        _ir[_t0:_t1] = _v
    put('ideal_reversal', _ir)

    # ---- behavior 族 ----
    put('clo_5d_60d', _rm(CLOSE_H, 5) / _rm(CLOSE_H, 60))
    put('vwap_5d_60d', _rm(VWAP_H, 5) / _rm(VWAP_H, 60))
    for lab, n in [('close_max_div_min_1m', 21), ('close_max_div_min_3m', 63), ('close_max_div_min_6m', 126)]:
        put(lab, (_dfw(CLOSE_H).rolling(n).max() / _dfw(CLOSE_H).rolling(n).min()).values)
    put('return_max_1m', _dfw(R).rolling(21).max().values)
    for lab, n in [('return_std_1w', 5), ('return_std_1m', 21), ('return_std_3m', 63),
                   ('return_std_6m', 126), ('return_std_12m', 252)]:
        put(lab, _rs(R, n))

    # ---- emotion 族 ----
    put('amt_1m_3m', _rm(AMT, 21) / _rm(AMT, 63))
    _Vn = np.where(VOL > 0, VOL, np.nan)
    for lab, n in [('ln_volume_mean_1m', 21), ('ln_volume_mean_3m', 63),
                   ('ln_volume_mean_6m', 126), ('ln_volume_mean_12m', 252)]:
        put(lab, np.log(_rm(_Vn, n)))
    for lab, n in [('ln_volume_std_1m', 21), ('ln_volume_std_3m', 63),
                   ('ln_volume_std_6m', 126), ('ln_volume_std_12m', 252)]:
        put(lab, np.log(_rs(_Vn, n)))
    put('volume_1m_div_12m', _rm(_Vn, 21) / _rm(_Vn, 252))
    put('volume_1m_minus_12m', _rm(_Vn, 21) - _rm(_Vn, 252))
    put('volume_std_1m_div_12m', _rs(_Vn, 21) / _rs(_Vn, 252))
    _TN = np.where(np.isfinite(TURN), TURN, np.nan)
    for lab, n in [('swap_1m', 21), ('swap_3m', 63), ('swap_1y', 252)]:
        put(lab, np.log(_rm(_TN, n) + CFG['eps']))
    for lab, n in [('turnover_mean_1m', 21), ('turnover_mean_3m', 63), ('turnover_mean_6m', 126)]:
        put(lab, _rm(_TN, n))
    for lab, n in [('turnover_std_1m', 21), ('turnover_std_3m', 63), ('turnover_std_6m', 126)]:
        put(lab, _rs(_TN, n))
    for lab, n in [('turnover_stdrate_1m', 21), ('turnover_stdrate_3m', 63), ('turnover_stdrate_6m', 126)]:
        put(lab, _rs(_TN, n) / (_rm(_TN, n) + CFG['eps']))
    put('illiq', _rm(np.abs(R) / (np.where(AMT > 0, AMT, np.nan) / 1e8), 21))
    _mx, _my = _rm(CLOSE_H, 20), _rm(_TN, 20)
    _mxy = _rm(CLOSE_H * np.nan_to_num(_TN, nan=np.nan), 20)
    _sx, _sy = _rs(CLOSE_H, 20), _rs(_TN, 20)
    put('corr_close_turnover', (_mxy - _mx * _my) / (_sx * _sy + CFG['eps']))
    put('ivr', _rs(SPEC, 60) / (_rs(R, 60) + CFG['eps']))
    _sd_up = _dfw(np.where(R > 0, R, np.nan)).rolling(60, min_periods=12).std().values.astype('float32')
    _sd_dn = _dfw(np.where(R < 0, R, np.nan)).rolling(60, min_periods=12).std().values.astype('float32')
    put('duvol', _sd_up / (_sd_dn + CFG['eps']))
    _m1, _m2, _m3 = _rm(SPEC, 60), _rm(SPEC ** 2, 60), _rm(SPEC ** 3, 60)
    _var = np.clip(_m2 - _m1 ** 2, 1e-12, None)
    put('ncskew', -(_m3 - 3 * _m1 * _m2 + 2 * _m1 ** 3) / _var ** 1.5)

    # ---- spread_bias（top10相关股组合，逐月末刷新252日相关矩阵，90%覆盖门槛）----
    _me = pd.Series(np.arange(T)).groupby(pd.Series(DATES).dt.to_period('M').values).max().values
    _rp = np.full((T, J), np.nan, 'float32')
    _elig_t = np.zeros((T, J), bool)
    for _k, _t_me in enumerate(_me):
        if _t_me < 252:
            continue
        Xw = R0[_t_me - 251:_t_me + 1].astype('float64')
        Xw = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-12)
        Cm = (Xw.T @ Xw) / 252
        np.fill_diagonal(Cm, -9)
        _elig = OBS[_t_me - 251:_t_me + 1].mean(axis=0) >= 0.90
        Cm[~_elig, :] = -9
        _top = np.argpartition(-Cm, 10, axis=0)[:10].T
        _lo, _hi = _t_me + 1, (_me[_k + 1] + 1 if _k + 1 < len(_me) else T)
        _rp[_lo:_hi] = R0[_lo:_hi][:, _top].mean(-1)
        _elig_t[_lo:_hi] = _elig
    _li = _dfw(np.log1p(R)).rolling(60).sum()
    _lp = _dfw(np.log1p(np.nan_to_num(_rp))).rolling(60).sum()
    _x = _lp - _li
    _sb = ((_x - _x.rolling(60).mean()) / (_x.rolling(60).std() + CFG['eps'])).values
    _sb[~_elig_t] = np.nan
    put('spread_bias', _sb)

    # ---- ±inf→NaN 清洗 + 生命期掩码 ----
    _alive = (np.cumsum(OBS, axis=0) > 0) & (np.cumsum(OBS[::-1], axis=0)[::-1] > 0)
    for _k in range(53):
        _sl = TECH[:, :, _k]
        _sl[~np.isfinite(_sl) | ~_alive] = np.nan
        TECH[:, :, _k] = _sl
    TECH.flush(); del TECH
    tmp.replace(tech_path)

    # ---- 逐日全市场统计（z 基准，全市场有值截面，非池内）----
    TECH = np.lib.format.open_memmap(tech_path, mode='r')
    MU = np.full((T, 53), np.nan, 'float32')
    SD = np.full((T, 53), np.nan, 'float32')
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for a in range(0, T, 200):
            b = min(a + 200, T)
            blk = np.asarray(TECH[a:b])
            MU[a:b] = np.nanmean(blk, axis=1)
            SD[a:b] = np.nanstd(blk, axis=1)
    np.savez(stats_path, MU=MU, SD=SD)
    print(f'tech53 OK: (T,J,53)=({T},{J},53) memmap + MU/SD{MU.shape} + tech_arrays -> {DERIVED}', flush=True)
    return tech_path


if __name__ == '__main__':
    build_tech53(force='--force' in sys.argv)

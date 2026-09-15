# -*- coding: utf-8 -*-
"""DRM 训练 + 推理 + expo 装配 + risk120/183 维市场状态。

全市场，年度滚动 PIT 训练。
依赖先跑 build_base（base.npz）+ build_idx63（idx63.npz）。产物落 DERIVED：
  deep_risk_exposure_daily/deep_risk_exposure_daily_{y}.pq  逐股逐日 d0..d9（链枢纽）
  expo.dat (T,J,10)                                          个股 70 维输入的后 10 维
  market_state.npz  MSTATE(T,183)=idx63(63)+risk120(120)     门控输入
模型/损失在 model_core/deep_risk_model.py。运行（服务器 GPU）：python -m input_data_update.build_drm
"""
try:
    import fcntl  # Linux 文件锁
except ImportError:  # Windows / macOS 无 fcntl：本地仅作 import 测试,DRM 实跑在 Linux
    fcntl = None
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (CFG, DERIVED, DRM_MODEL_DIR as MODEL_DIR, device, DA_OPT,   # DRM 全市场共享,不随训练池分叉
                    DTAG, DRM_CKPT, EXPO_PATH, MSTATE_PATH, EXPO_PQ_DIR)  # _drmdk 变体经 DTAG 整链分叉
from model_core.deep_risk_model import DRM, cs_zscore, drm_loss

EXPO_DIR = EXPO_PQ_DIR
FCOLS = [f'd{i}' for i in range(10)]


class DRMData:
    """DRM 的全市场批与滚动划分（base.npz）。"""

    def __init__(self):
        z = np.load(DERIVED / 'base.npz', allow_pickle=True)
        self.PRICE, self.BARRA = z['PRICE'], z['BARRA']
        self.RET, self.VALID, self.LABEL_OK = z['RET'], z['VALID'], z['LABEL_OK']
        self.DATES, self.CODES = pd.DatetimeIndex(z['DATES']), z['CODES']
        self.T, self.J = self.VALID.shape
        self._d2i = {d: i for i, d in enumerate(self.DATES)}
        self.DEV = device()

    def day_batch(self, t, with_label=True):
        """交易日 t 截面批：X (n,40,15)=5量价比值(÷窗末+MAD)⊕10 Barra；Y (n,20)=未来20日逐日收益。"""
        W = CFG['drm_window']                                   # DRM 专属窗(主线40/deck变体20),与 MASTER 的40解耦
        jj = np.where(self.VALID[t] & self.LABEL_OK[t] if with_label else self.VALID[t])[0]
        win = self.PRICE[t - W + 1: t + 1, jj, :]                # (40,n,5)
        ratio = win / (win[-1:, :, :] + CFG['eps'])              # ÷窗末日值（研报口径）
        med = np.nanmedian(ratio, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(ratio - med), axis=1, keepdims=True)
        x_p = np.clip(ratio, med - CFG['mad_k'] * mad, med + CFG['mad_k'] * mad)
        x_b = self.BARRA[t - W + 1: t + 1, jj, :]                # Barra 已标准化，不再处理
        x = torch.from_numpy(np.ascontiguousarray(
            np.concatenate([x_p, x_b], axis=-1).transpose(1, 0, 2)))   # (n,40,15)
        if not with_label:
            return x, jj
        y = torch.from_numpy(np.ascontiguousarray(self.RET[t + 1: t + 1 + CFG['horizon'], jj].T))
        return x, y, jj

    def split_days(self, expo_year):
        """年度 Y 训练/验证日（PIT + 防泄露）：训练窗=[Y-4,Y-1]，训练标签实现日<验证首日。"""
        D, d2i = self.DATES, self._d2i
        w0 = d2i[D[D >= pd.Timestamp(f'{expo_year - CFG["drm_train_years"]}-01-01')][0]]
        w1 = d2i[D[D <= pd.Timestamp(f'{expo_year - 1}-12-31')][-1]]
        days = [t for t in range(max(w0, CFG['drm_window'] - 1), w1 - CFG['horizon'] + 1)
                if self.VALID[t].sum() >= 100]
        n_val = max(int(len(days) * CFG['drm_val_frac']), 20)
        val = days[-n_val:]
        train = [t for t in days[:-n_val] if t + CFG['horizon'] < val[0]]
        return train, val


def expo_years(data):
    """暴露产出年份 = 首年..面板末年（含端点），保证最新 OOS 年拿到暴露与门控输入。"""
    return list(range(CFG['drm_first_expo_year'], int(data.DATES[-1].year) + 1))


def train_year(data, expo_year, quiet=False):
    """单年度 DRM（早停 + 上一年暖启动 + PIT 元信息）。断点续跑：已存在则跳过。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f'{DRM_CKPT}_{expo_year}.pt'
    if ckpt.exists():
        return ckpt
    train, val = data.split_days(expo_year)
    torch.manual_seed(CFG['seed'] + expo_year); np.random.seed(CFG['seed'] + expo_year)
    dev = data.DEV
    model = DRM().to(dev)
    prev = MODEL_DIR / f'{DRM_CKPT}_{expo_year - 1}.pt'
    warm = bool(CFG['drm_warm_start'] and prev.exists())
    if warm:
        model.load_state_dict(torch.load(prev, map_location='cpu')['state'])   # 因子身份跨年连续
    lr0 = CFG['drm_lr']                            # 暖启动年沿用 drm_lr，不降学习率
    opt = torch.optim.Adam(model.parameters(), lr=lr0)

    def _val():
        model.eval()
        with torch.no_grad():
            return float(np.mean([drm_loss(model(data.day_batch(t)[0].to(dev)),
                                           data.day_batch(t)[1].to(dev)).item() for t in val]))

    best, best_state, bad, epoch = float('inf'), None, 0, 0
    best_ep, stall, nhalf = -1, 0, 0
    if DA_OPT == 'ft':                     # epoch0 基线:沿用上年权重(或初始权重)直接过验证——
        best = _val()                      # "不更新"成为合法最优解;主线(base)不评,行为逐位不变
        best_state, best_ep = {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        if not quiet:
            print(f'  DRM {expo_year} ep0(沿用{"上年权重" if warm else "初始权重"}) val={best:.5f} lr0={lr0:g}', flush=True)
    for epoch in range(CFG['drm_max_epoch']):
        model.train()
        for t in tqdm(np.random.permutation(train), desc=f'DRM{expo_year} ep{epoch + 1}',
                      leave=False, disable=quiet):
            x, y, _ = data.day_batch(int(t))                     # daily_batch：每日截面即一个 batch
            loss = drm_loss(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([drm_loss(model(data.day_batch(t)[0].to(dev)),
                                         data.day_batch(t)[1].to(dev)).item() for t in val]))
        if vl < best - 1e-6:
            best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            best_ep, stall = epoch + 1, 0
        else:
            bad += 1
            if DA_OPT == 'ft':
                stall += 1
                if stall >= CFG['plateau_stall']:                    # plateau 退火:停滞3轮→lr减半
                    for g_ in opt.param_groups:
                        g_['lr'] *= CFG['plateau_factor']
                    stall = 0; nhalf += 1
                    if not quiet:
                        print(f'  ⤷ DRM {expo_year} plateau: lr → {opt.param_groups[0]["lr"]:.2e}', flush=True)
        if not quiet:
            print(f'  DRM {expo_year} ep{epoch + 1} val={vl:.5f} best={best:.5f} bad={bad}', flush=True)
        if bad >= CFG['drm_patience']:
            break
    _tmp = ckpt.with_name(f'{ckpt.name}.{os.getpid()}.tmp')
    torch.save({'state': best_state, 'meta': dict(
        model_version=f'{DRM_CKPT}_{expo_year}', expo_year=expo_year, best_val=best, epochs=epoch + 1,
        best_epoch=best_ep, lr0=lr0, lr_halvings=nhalf,              # best_epoch=0 ⇒ 该年沿用上年权重
        train_start=str(data.DATES[train[0]].date()), train_end=str(data.DATES[train[-1]].date()),
        val_start=str(data.DATES[val[0]].date()), val_end=str(data.DATES[val[-1]].date()),
        warm_started=bool(CFG['drm_warm_start'] and prev.exists()))}, _tmp)
    os.replace(_tmp, ckpt)                                       # 原子落盘:并发/中断都不会留半成品被 exists 误认
    return ckpt


def infer_expo(data, quiet=False):
    """年度 Y 用"训练截止 Y-1"的模型逐日推理 → deep_risk_exposure_daily_{y}.pq（cs_zscore 后）。cell8。"""
    EXPO_DIR.mkdir(parents=True, exist_ok=True)
    dev = data.DEV
    for y in expo_years(data):
        path = EXPO_DIR / f'deep_risk_exposure_daily_{y}.pq'
        if path.exists():
            continue
        st = torch.load(MODEL_DIR / f'{DRM_CKPT}_{y}.pt', map_location='cpu')
        model = DRM().to(dev); model.load_state_dict(st['state']); model.eval()
        days = [t for t in range(data.T) if data.DATES[t].year == y and data.VALID[t].sum() >= 100]
        out = []
        with torch.no_grad():
            for t in tqdm(days, desc=f'expo {y}', disable=quiet):
                x, jj = data.day_batch(t, with_label=False)
                f = cs_zscore(model(x.to(dev))).cpu().numpy()
                df = pd.DataFrame(f, columns=FCOLS)
                df.insert(0, 'code', data.CODES[jj]); df.insert(0, 'date', data.DATES[t])
                out.append(df)
        df = pd.concat(out, ignore_index=True)
        df['model_version'] = st['meta']['model_version']
        _tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        df.to_parquet(_tmp, index=False)
        os.replace(_tmp, path)                                   # 原子落盘
        print(f'expo {y}: {df.shape} -> {path.name}', flush=True)


def _load_expo_all():
    files = sorted(EXPO_DIR.glob('deep_risk_exposure_daily_2*.pq'))
    if not files:
        raise FileNotFoundError('无 deep_risk_exposure_daily —— 先训练+推理 DRM')
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def build_expo(data, force=False):
    """装配 (T,J,10) 暴露 memmap expo.dat（缺失NaN + 时间轴≤10日ffill）。cell18。"""
    out = EXPO_PATH
    if out.exists() and not force:
        print(f'{out.name} 已存在，跳过。', flush=True)
        return out
    expo_all = _load_expo_all()
    T, J = data.T, data.J
    tmp = out.with_name(f'{out.name}.building.{os.getpid()}')   # tmp带pid:并发不互相截断
    EXPO = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(T, J, 10))
    EXPO[:] = np.nan
    ci = pd.Index(data.CODES)
    for d, g in expo_all.groupby('date'):
        t = data._d2i.get(d)
        if t is None:
            continue
        jj = ci.get_indexer(g['code']); ok = jj >= 0
        EXPO[t, jj[ok]] = g[FCOLS].values[ok].astype('float32')
    for j0 in range(0, J, 500):                                  # 时间轴限10日前向填充（短停牌沿用近期暴露）
        j1 = min(j0 + 500, J)
        blk = pd.DataFrame(EXPO[:, j0:j1, :].reshape(T, -1)).ffill(limit=10).values
        EXPO[:, j0:j1, :] = blk.reshape(T, j1 - j0, 10)
    EXPO.flush(); del EXPO
    tmp.replace(out)
    print(f'{out.name} OK: (T,J,10)=({T},{J},10) -> {out}', flush=True)
    return out


def build_mstate(data, force=False):
    """risk120（全市场当日涨幅Top10%赢家因子均值×{5,10,20,40}窗×{mu,sd,amp}）+ 拼 idx63 → MSTATE(183)。cell13。"""
    out = MSTATE_PATH
    if out.exists() and not force:
        print(f'{out.name} 已存在，跳过。', flush=True)
        return out
    expo_all = _load_expo_all()
    DATES, CODES, RET, VALID, J = data.DATES, data.CODES, data.RET, data.VALID, data.J
    ci = pd.Index(CODES)
    rows = []
    for d, g in expo_all.groupby('date'):                       # q_{k,t}: 全市场涨幅前10%赢家因子均值
        t = data._d2i.get(d)
        if t is None:
            continue
        jj = ci.get_indexer(g['code'])
        ok = (jj >= 0) & VALID[t, np.clip(jj, 0, J - 1)]        # 赢家池=全市场（非策略池）
        if ok.sum() < 300:
            continue
        r = RET[t, jj[ok]]
        top = r >= np.quantile(r, 0.9)                          # 当日涨幅前10%
        rows.append([d] + list(g[FCOLS].values[ok][top].mean(axis=0)))
    Q = pd.DataFrame(rows, columns=['date'] + FCOLS).set_index('date').sort_index()
    o = {}
    for k in FCOLS:
        q = Q[k]
        for n in [5, 10, 20, 40]:
            mu = q.rolling(n).mean()
            o[f'risk120_{k}_mu{n}'] = mu
            o[f'risk120_{k}_sd{n}'] = q.rolling(n).std()
            o[f'risk120_{k}_amp{n}'] = mu / (q + CFG['eps'])    # 放大性（研报图22；重尾，门控前 robust 标准化在模块四）
    STATE120 = pd.DataFrame(o).reindex(DATES)
    assert STATE120.shape[1] == 120, f'risk120 维度 {STATE120.shape[1]}≠120'
    idx = np.load(DERIVED / 'idx63.npz', allow_pickle=True)
    STATE63 = pd.DataFrame(idx['STATE63'], index=DATES, columns=list(idx['cols']))
    MSTATE = pd.concat([STATE63, STATE120], axis=1)
    assert MSTATE.shape[1] == 183, f'市场状态维度 {MSTATE.shape[1]}≠183'
    np.savez(out, MSTATE=MSTATE.values.astype('float32'), cols=np.asarray(MSTATE.columns))
    mm = np.isfinite(MSTATE.values).all(axis=1)
    _fin = np.where(mm)[0]
    print(f'{out.name} OK: 183维 满窗{int(mm.sum())}日 '
          f'({DATES[_fin[0]].date()}~{DATES[_fin[-1]].date()}) -> {out}', flush=True)
    return out


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if fcntl is None:                              # 非 Linux：直接跑(不并发锁)
        _drm_main()
        return
    with open(MODEL_DIR / f'.drm{DTAG}.lock', 'w') as _lk:              # 多池 prepare 并发时整段 DRM 串行:
        fcntl.flock(_lk, fcntl.LOCK_EX)                          # 先到者建,后到者拿到锁时产物已在,exists秒过
        _drm_main()


def _drm_main():
    data = DRMData()
    ys = expo_years(data)
    print(f'DRM 设备={data.DEV} | 全市场 T={data.T} J={data.J} | 暴露年={ys[0]}..{ys[-1]}', flush=True)
    for y in ys:
        train_year(data, y)                                     # 逐年滚动训练（断点续跑）
    infer_expo(data)                                            # 逐日推理落 deep_risk_exposure_daily
    build_expo(data)                                           # 装配 expo.dat
    build_mstate(data)                                         # risk120 + 183维市场状态
    print('DRM 全流程完成: deep_risk_exposure_daily + expo.dat + market_state.npz', flush=True)


if __name__ == '__main__':
    main()

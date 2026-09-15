# -*- coding: utf-8 -*-
"""MASTER 骨干的逐年监督训练原语。

供两处调用：①静态 MASTER 因子线（update/master_score.py）逐年产模型；
②DoubleAdapt 暖启动变体（da_warm_start=True 时载入；定稿主线为冷启动，不依赖本产物）。

划分口径：训练=[Y−6,Y−2]五年；验证位于 Y−1 且标签 t+11 仍在 Y−1 年内；
训练标签实现日 < 验证年首日，二者均不触碰 OOS 年。
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import CFG, MODEL_DIR, VTAG, ATAG, DA_VAL_METRIC
from model_core.master_backbone import MasterBackbone, wmse, train_loss
from model_core.market_gate import MarketGate, GateNormalizer


def split_master(panel, oos_year):
    """返回 (train_days, val_days)：训练=[Y−6,Y−2]，验证=Y−1 年内标签可完整实现部分。"""
    D = panel.DATES
    w0 = np.searchsorted(D, pd.Timestamp(f'{oos_year - 6}-01-01'))
    v0 = np.searchsorted(D, pd.Timestamp(f'{oos_year - 1}-01-01'))       # 验证年(Y−1)首日
    w1 = np.searchsorted(D, pd.Timestamp(f'{oos_year}-01-01')) - 1       # 窗末=Y−1 年末
    days = [int(t) for t in panel.MM_DAYS
            if w0 <= t and t + CFG['label_h'] <= w1
            and panel.POOL_LBL[t].sum() > 0]
    train = [t for t in days if t + CFG['label_h'] < v0]                 # 标签不越入验证年
    val = [t for t in days if t >= v0]
    return train, val


def train_master_year(panel, oos_year, quiet=False):
    """训练 OOS 年 Y 的静态 MASTER，存 master_{Y}.pt（含 model/gate/gn/meta）。断点续跑：已存在则跳过。"""
    ckpt = MODEL_DIR / f'master{VTAG}{ATAG}_{oos_year}.pt'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        return ckpt
    train, val = split_master(panel, oos_year)
    torch.manual_seed(CFG['seed'] + oos_year)
    np.random.seed(CFG['seed'] + oos_year)
    dev = panel.DEV

    gn = GateNormalizer().fit(panel.MSTATE[train])                       # 门控标准化只在训练窗拟合（PIT）
    model = MasterBackbone().to(dev)
    gate = MarketGate(mode=CFG['gate_mode']).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=CFG['m_lr'])

    def _day_loss(t):
        x, m, jj = panel.day_batch(t)
        y = torch.from_numpy(panel.Y_RANK[t, jj].astype('float32')).to(dev)
        ok = torch.isfinite(y)                                          # 标签可实现子集只进损失（全截面前向）
        pred = model(x.to(dev), gate(gn.transform(m.numpy()).to(dev)))
        return train_loss(pred[ok], y[ok])

    best, best_state, bad = float('inf'), None, 0

    # ---- 断点续跑（全A 单轮 epoch 数分钟、上限 100 轮，断线后可从存档继续）----
    rs = MODEL_DIR / f'master{VTAG}{ATAG}_{oos_year}_resume.pt'
    ep0 = 0
    if rs.exists():
        st = torch.load(rs, map_location='cpu')
        model.load_state_dict(st['model']); gate.load_state_dict(st['gate']); opt.load_state_dict(st['opt'])
        np.random.set_state(st['rng_np']); torch.set_rng_state(st['rng_torch'])   # 日序与不中断逐位一致
        best, bad, best_state, ep0 = st['best'], st['bad'], st['best_state'], st['epoch'] + 1
        print(f'  MASTER {oos_year} 续跑：从 epoch{ep0 + 1} 起（best={best:.5f} bad={bad}）', flush=True)

    epoch = ep0 - 1
    for epoch in range(ep0, CFG['m_max_epoch']):
        model.train(); gate.train()
        for t in tqdm(np.random.permutation(train), desc=f'MASTER预训练 ep{epoch + 1}/{CFG["m_max_epoch"]}',
                      leave=False, disable=quiet):
            loss = _day_loss(int(t))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); gate.eval()
        with torch.no_grad():
            ws, ics = [], []
            for t in val:
                x, m, jj = panel.day_batch(int(t))
                y = torch.from_numpy(panel.Y_RANK[int(t), jj].astype('float32')).to(dev)
                ok = torch.isfinite(y)
                pred = model(x.to(dev), gate(gn.transform(m.numpy()).to(dev)))
                ws.append(float(wmse(pred[ok], y[ok])))
                if int(ok.sum()) >= 2:
                    pv, yv = pred[ok].cpu().numpy(), y[ok].cpu().numpy()
                    ics.append(float(np.corrcoef(pv, yv)[0, 1]))
        vl, vic = float(np.mean(ws)), (float(np.mean(ics)) if ics else float('nan'))
        crit = vl if DA_VAL_METRIC == 'wmse' else -vic                  # 共用一键,统一最小化
        if crit < best - 1e-7:
            best, bad = crit, 0
            best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                          'gate': {k: v.cpu().clone() for k, v in gate.state_dict().items()},
                          'gn': gn.state_dict()}
        else:
            bad += 1
        if not quiet:
            print(f'  MASTER {oos_year} epoch{epoch + 1} wMSE={vl:.5f} IC={vic:+.4f} '
                  f'[判据={DA_VAL_METRIC}] best={best:.5f} bad={bad}', flush=True)
        torch.save({'model': model.state_dict(), 'gate': gate.state_dict(), 'opt': opt.state_dict(),
                    'rng_np': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                    'epoch': epoch, 'best': best, 'bad': bad, 'best_state': best_state}, rs)   # 每轮存档
        if bad >= CFG['m_patience']:
            break
    torch.save({**best_state, 'meta': dict(
        oos_year=oos_year, train_start=str(panel.DATES[train[0]].date()),
        train_end=str(panel.DATES[train[-1]].date()),
        val_start=str(panel.DATES[val[0]].date()), val_end=str(panel.DATES[val[-1]].date()),
        best_val=best, epochs=epoch + 1, val_metric=DA_VAL_METRIC)}, ckpt)
    rs.unlink(missing_ok=True)                                          # 完成后清掉续跑存档
    return ckpt

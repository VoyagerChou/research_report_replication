# -*- coding: utf-8 -*-
"""静态 MASTER 对照线：`python main.py master [起始年]`。

逐年训练静态 MASTER（model_core/master_train.train_master_year，训练=[Y−6,Y−2]、验证=Y−1）
并对当年逐日打分落库。README：对照线覆盖 2019–2024（可传年份列表扩展）；
主线(DA 元增量)是交付线，本线用于"基线→MASTER→最终模型"的递进对照。

产物：data/factor_values/master_score{VTAG}{ATAG}/master_score_{year}.pq  列 date/code/score/model_version
（score = sigmoid 输出 ∈(0,1)，截面可比；与 DA 线 H⁻¹ 还原后的语义同为"多头打分"）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from config import CFG, MODEL_DIR, MASTER_SCORE_DIR, VTAG, ATAG
from model_core.prepare_data import PanelData
from model_core.master_train import train_master_year
from model_core.master_backbone import MasterBackbone
from model_core.market_gate import MarketGate, GateNormalizer

MODEL_VERSION = f'master_score{VTAG}{ATAG}'


def score_year(panel, year, quiet=False):
    """载入 master_{year}.pt, 对当年逐日打分并落库."""
    out = MASTER_SCORE_DIR / f'{MODEL_VERSION}_{year}.pq'
    if out.exists():
        print(f'  {year} 已落库，跳过', flush=True)
        return out
    ckpt = MODEL_DIR / f'master{VTAG}{ATAG}_{year}.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f'{ckpt} 不存在——先跑 train_master_year({year})。')
    st = torch.load(ckpt, map_location='cpu')
    dev = panel.DEV
    model = MasterBackbone().to(dev); model.load_state_dict(st['model']); model.eval()
    gate = MarketGate(mode=CFG['gate_mode']).to(dev); gate.load_state_dict(st['gate']); gate.eval()
    gn = GateNormalizer().load_state_dict(st['gn'])

    days = [int(t) for t in panel.MM_DAYS if panel.DATES[t].year == year]
    rows = []
    with torch.no_grad():
        for t in days:
            x, m, jj = panel.day_batch(int(t))
            if len(jj) == 0:
                continue
            sc = model(x.to(dev), gate(gn.transform(m.numpy()).to(dev))).cpu().numpy()
            rows.append(pd.DataFrame({'date': panel.DATES[t], 'code': panel.CODES[jj],
                                      'score': sc.astype('float32')}))
    if not rows:
        print(f'  {year}: 无可打分日', flush=True)
        return None
    df = pd.concat(rows, ignore_index=True)
    df['model_version'] = MODEL_VERSION
    MASTER_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if not quiet:
        print(f'  落库 {out.name}: {len(df):,} 行 '
              f'{pd.to_datetime(df["date"]).min().date()}~{pd.to_datetime(df["date"]).max().date()}', flush=True)
    return out


def main(yrs=None):
    print('== 静态 MASTER 对照线 ==', flush=True)
    panel = PanelData()
    years = yrs or list(range(2019, 2025))        # README: 2019–2024（主线覆盖至 2026 由 DA update 承担）
    for y in years:
        ckpt = MODEL_DIR / f'master{VTAG}{ATAG}_{y}.pt'
        if not ckpt.exists():
            train_master_year(panel, y)            # 断点续跑内建
        score_year(panel, y)
    print('静态 MASTER 因子线完成。', flush=True)


if __name__ == '__main__':
    yrs = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(yrs or None)

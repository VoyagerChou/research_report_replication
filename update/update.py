# -*- coding: utf-8 -*-
"""在线元增量：`python main.py update`（主线因子生产）。

流程（README §4 / model_core.meta_train 注释）：
  离线只训一次（da_offline{VTAG}{ATAG}.pt，含 φ=骨干+门控、ψ=DataAdapter、opt、gn）。
  此后不再滚动重训，每 20 个交易日一个更新节点：元更新(outer_step) →
  以最近已实现标签块做内层适应(adapt_delta) → 节点窗内逐日打分(score_days) →
  按年分片落库到 data/factor_values/meta_master_score{VTAG}{ATAG}/。

实验开关（默认关闭；产物经 ATAG 后缀与主线完全隔离，须用相同环境变量先跑 train）：
  · DA_EVENT=1 事件驱动更新（Step2 改调度）：读 data/derived/switch_alarm_daily.csv，
    把风格切换报警段起点并入更新节点（报警当日立即更新，窗口=到下一节点为止），
    事件节点内层适应学习率放大 CFG['event_trust'] 倍（信任域调制）。
    见 model_core/event_schedule.py。
  · DA_FAST=1 快慢双层（Step3）：在线打分层叠加日频线性修正器（model_core/fast_layer.py），
    用最近已实现标签滚动拟合 score→标签截面映射（含 10 维 DRM 暴露），强收缩先验；
    原始 score 另存 sidecar（FAST_RAW_DIR/raw_{年}.pq）供断点续跑重建。
  · DA_MCLEAN=1 输入层清理（Step3）：市场状态在 PanelData 侧做 EWMA+PC 去噪，离线/在线一致。

断点续跑：整年定稿落库 + 年界状态存档 da_online_state{VTAG}{ATAG}_{year}.pt（φ+ψ+opt+gn）；
DA_FAST 时同时落 raw sidecar，恢复后自动重建修正器滚动统计。
（排序类损失的 pair 采样走 CPU RNG，续跑采样序列与连续跑不同，统计等价而非逐位一致。）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from config import CFG, MODEL_DIR, SCORE_DIR, VTAG, ATAG, FAST_RAW_DIR
from model_core.prepare_data import PanelData
from model_core.meta_train import DoubleAdaptTrainer
from model_core.market_gate import GateNormalizer
from model_core.adapter import DataAdapter, MetaMasterNet

MODEL_VERSION = f'meta_master_score{VTAG}{ATAG}'


def main(quiet=False):
    print(f'== 在线元增量 {CFG["da_online_y0"]}–{CFG["da_online_y1"]} ==', flush=True)
    panel = PanelData()
    trainer = DoubleAdaptTrainer(panel)
    dev = panel.DEV

    ckpt = MODEL_DIR / f'da_offline{VTAG}{ATAG}.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f'{ckpt} 不存在——先跑 `python main.py train`。')
    st = torch.load(ckpt, map_location='cpu')
    phi = MetaMasterNet().to(dev); phi.load_state_dict(st['phi'])
    adapter = DataAdapter().to(dev); adapter.load_state_dict(st['adapter'])
    gn = GateNormalizer().load_state_dict(st['gn'])
    opt = torch.optim.Adam([{'params': list(phi.parameters()), 'lr': CFG['da_online_lr']},
                            {'params': list(adapter.parameters()), 'lr': CFG['da_psi_lr']}])
    opt.load_state_dict(st['opt'])
    print(f'已载入 {ckpt.name}（φ+ψ+opt+gn）', flush=True)

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    y0, y1 = CFG['da_online_y0'], CFG['da_online_y1']

    def _out(year):
        return SCORE_DIR / f'{MODEL_VERSION}_{year}.pq'

    def _state(year):
        return MODEL_DIR / f'da_online_state{VTAG}{ATAG}_{year}.pt'

    def _raw(year):
        return FAST_RAW_DIR / f'raw_{year}.pq'

    def _flush(year, parts, raw_parts):
        """整年定稿落库（覆盖写）+ 年界状态存档（断点续跑恢复点）+（DA_FAST）raw sidecar。

        状态含 φ+ψ+opt+gn：在线元增量是时序递推，恢复年份必须从上一年的年末状态继续，
        否则该年缺失此前全部元更新轨迹（与连续跑不一致）。"""
        if not parts:
            return
        acc = pd.concat(parts, ignore_index=True)
        acc = acc.sort_values(['date', 'code']).drop_duplicates(['date', 'code'], keep='last')
        acc.to_parquet(_out(year), index=False)
        raw_n = 0
        if CFG['fast'] and raw_parts:
            FAST_RAW_DIR.mkdir(parents=True, exist_ok=True)
            raw = pd.concat(raw_parts, ignore_index=True)
            raw = raw.sort_values(['date', 'code']).drop_duplicates(['date', 'code'], keep='last')
            raw.to_parquet(_raw(year), index=False)
            raw_n = len(raw)
        torch.save({'fmt': 'v1', 'phi': phi.state_dict(), 'adapter': adapter.state_dict(),
                    'opt': opt.state_dict(), 'gn': gn.state_dict()}, _state(year))
        if not quiet:
            _extra = f'（raw sidecar {raw_n:,} 行）' if raw_n else ''
            print(f'  落库 {_out(year).name}: {len(acc):,} 行 '
                  f'{pd.to_datetime(acc["date"]).min().date()}~{pd.to_datetime(acc["date"]).max().date()}'
                  f'{_extra}（状态 {_state(year).name}）', flush=True)

    # ---- 更新网格（DA_EVENT=1 时并入报警段起点）----
    if CFG['event']:
        from model_core.event_schedule import load_alarm_series, build_event_schedule
        alarm = load_alarm_series()
        if alarm is None:
            print('警告: DA_EVENT=1 但未找到 switch_alarm_daily.csv，退回常规排期。', flush=True)
            grid = [(s, w, False) for s, w in panel.update_grid(y0, y1)]
        else:
            grid = build_event_schedule(panel, y0, y1, alarm)
            n_evt = sum(1 for _, _, e in grid if e)
            print(f'事件驱动排期: {len(grid)} 个节点（其中事件节点 {n_evt} 个）', flush=True)
    else:
        grid = [(s, w, False) for s, w in panel.update_grid(y0, y1)]

    # ---- 预扫描待处理年份 ----
    years_seq = []
    for s, _, _ in grid:
        yr = int(panel.DATES[s].year)
        if not years_seq or years_seq[-1] != yr:
            years_seq.append(yr)
    todo = [yr for yr in years_seq if not _out(yr).exists()]
    if not todo:
        print('全部年份已落库，无待处理年份。', flush=True)
        return
    first_todo = min(todo)

    # ---- 断点续跑：first_todo 之前最近的「落库+状态」配对恢复点 ----
    resume_year = None
    if first_todo > y0:
        for yr in range(first_todo - 1, y0 - 1, -1):
            if not (_out(yr).exists() and _state(yr).exists()):
                continue
            try:
                st2 = torch.load(_state(yr), map_location='cpu')
            except Exception:
                continue
            if st2.get('fmt') != 'v1':
                continue
            phi.load_state_dict(st2['phi']); adapter.load_state_dict(st2['adapter'])
            opt.load_state_dict(st2['opt'])
            resume_year = yr
            print(f'断点续跑：从 {yr} 年末状态恢复（{_state(yr).name}）', flush=True)
            break
        if resume_year is None:
            raise RuntimeError(
                f'年份 {first_todo} 待处理，但其前的年份缺少「落库+状态」配对，无法安全续跑'
                f'（直接跳过会使后续年份缺失元更新轨迹）。请删除 {_out(first_todo).name} 之前的'
                f'全部 {MODEL_VERSION}_*.pq 后整段重跑，或放回 da_online_state{VTAG}{ATAG}_*.pt。')

    # ---- 快慢双层初始化（DA_FAST=1；恢复时从 sidecar 重建滚动统计）----
    fast = None
    if CFG['fast']:
        from model_core.fast_layer import FastCorrector
        fast = FastCorrector(panel)
        if resume_year is not None:
            fast.rebuild_from_sidecar(resume_year)

    # ---- 在线滚动：排期节点（+事件节点）；恢复点之后的年份全部（重）处理 ----
    skip_upto = resume_year if resume_year is not None else y0 - 1
    cur_year, parts, raw_parts = None, [], []
    for s, win, is_event in grid:
        year = int(panel.DATES[s].year)
        if year <= skip_upto:
            continue                                       # 已含在恢复点状态内
        if cur_year is not None and year != cur_year:      # 年界：落库上一年 + 存状态
            _flush(cur_year, parts, raw_parts)
            parts, raw_parts = [], []
        cur_year = year
        sup, qry = panel.hist_task(s)
        if len(sup) >= 10 and len(qry) >= 10:
            phi.train()
            trainer.outer_step(phi, adapter, opt, sup, qry, gn)          # 元更新
        lr_scale = CFG['event_trust'] if is_event else 1.0                # 信任域调制（事件节点放大）
        delta = trainer.adapt_delta(phi, adapter, panel.ft_block(s), gn, opt, lr_scale=lr_scale)
        rows = trainer.score_days(phi, adapter, delta, [int(t) for t in win], gn)
        if not rows:
            continue
        if fast is not None:                                              # 快慢双层：日频修正
            raw_rows, corr_rows = [], []
            for df in rows:
                if not len(df):
                    continue
                t = panel._d2i[df['date'].iat[0]]
                raw_rows.append(df)
                corr_rows.append(fast.apply_day(df, t))
            if not corr_rows:
                continue
            raw_parts.append(pd.concat(raw_rows, ignore_index=True))
            sub = pd.concat(corr_rows, ignore_index=True)
        else:
            sub = pd.concat(rows, ignore_index=True)   # score_days 返回逐日 DataFrame 列表
        sub['model_version'] = MODEL_VERSION
        parts.append(sub)
        if not quiet:
            tag = ' [事件]' if is_event else ''
            print(f'  {MODEL_VERSION} {year}: 节点 {panel.DATES[s].date()}{tag} 打分 {len(sub):,} 行', flush=True)
    _flush(cur_year, parts, raw_parts)
    print('在线元增量完成。', flush=True)


if __name__ == '__main__':
    main()

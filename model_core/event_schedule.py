# -*- coding: utf-8 -*-
"""事件驱动更新调度：把风格切换检测器的报警日并入在线更新网格（DA_EVENT=1 启用）。

背景（Step2「改调度」）：
  主线在线阶段每 20 个交易日一个更新节点（元更新 → 内层适应 → 窗口打分），是匀速机制；
  风格切换（断点）在几天内完成，匀速节拍在断点后平均滞后约 7~8 个交易日才更新（diag/step5
  回放实证：2024H1 滞后 8 日、2024Q4 滞后 7 日）。本模块把「报警段起点」直接变成更新节点：
  报警当日立即做一次元更新+内层适应，并把该节点的内层适应学习率放大（信任域调制，见
  update.py 传入 lr_scale），使模型在切换窗口内更快调整。

报警序列来源：
  data/derived/switch_alarm_daily.csv —— diag/step5 风格切换检测器（零训练、纯测量）的定稿输出：
    date / bpi / state / alarm_flag / style_argmax / style_sign
  口径：触发记忆式状态机（近 3 日 2 日 BPI>3.5 且第二风格确认 / 单日 BPI>4.5 / 涨跌停广度极端），
  报警段起点 = 进入 ALARM 的首日（alarm_flag==1）。该序列在 diag 工程中标定（2004-2018 标定、
  2019-2026 冻结验证）后随交付体提供，服务器侧只读、不重算。

调度语义（build_event_schedule）：
  - 排期节点：y0 首个可训日起每 CFG['da_incr'] 个可训日（与原 update_grid 一致）；
  - 事件节点：报警段起点后第一个可训日；
  - 合并：与上一保留节点相距 < CFG['event_gap'] 的节点丢弃；若被丢弃的是事件节点，
    则把信任域标记升级到前一保留节点（事件信息不丢）；
  - 打分窗口：每个节点的窗口 = 到下一个节点为止的可训日（无事件时退化为原 20 日窗），
    年界按 y1 截断——保证每一天只被"最近一次更新后的模型"打分一次。
"""
import numpy as np
import pandas as pd

from config import CFG, SWITCH_ALARM_CSV


def load_alarm_series(path=None):
    """读取报警序列 CSV；文件缺失返回 None（调用方退回常规排期）。"""
    p = path if path is not None else SWITCH_ALARM_CSV
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df.sort_values('date').reset_index(drop=True)


def build_event_schedule(panel, y0, y1, alarm_df, min_gap=None):
    """构建事件驱动更新网格。返回 [(s, win, is_event)]。

    s      : 节点日（panel 行索引）
    win    : 该节点的打分窗口（panel 行索引列表，到下一节点为止，年界截断）
    is_event: 是否为事件节点（信任域调制标记）
    """
    min_gap = CFG['event_gap'] if min_gap is None else min_gap
    mmd = np.asarray(panel._MMD)                       # 183 维完整的可训日（与原 update_grid 同源）
    mmd_dates = pd.DatetimeIndex(panel.DATES[mmd])

    ii = np.where(mmd_dates.year == y0)[0]
    if not len(ii):
        return []
    cand = [(int(i), False) for i in range(int(ii[0]), len(mmd), CFG['da_incr'])]   # 排期节点

    if alarm_df is not None:                           # 事件节点：报警段起点 → 第一个 >= 的可训日
        for d in alarm_df.loc[alarm_df['alarm_flag'] == 1, 'date']:
            if d.year < y0 or d.year > y1:
                continue
            j = int(np.searchsorted(mmd_dates.values, np.datetime64(d), side='left'))
            if j < len(mmd):
                cand.append((j, True))
    cand.sort()

    kept = []                                          # 合并：与上一保留节点过近则丢弃
    for j, is_evt in cand:
        if kept and j - kept[-1][0] < min_gap:
            if is_evt:                                 # 事件被并掉 → 信任域标记升级到前一节点
                kept[-1] = (kept[-1][0], True)
            continue
        kept.append((j, is_evt))

    grid = []
    for k, (j, is_evt) in enumerate(kept):
        s = int(mmd[j])
        if panel.DATES[s].year > y1:
            break
        nxt = kept[k + 1][0] if k + 1 < len(kept) else j + CFG['da_incr']
        win = [int(t) for t in mmd[j:nxt] if panel.DATES[t].year <= y1]
        grid.append((s, win, is_evt))
    return grid

# -*- coding: utf-8 -*-
"""数据层：加载对齐面板 + 提供 day_batch + 元任务采样。

DoubleAdapt 的输入单元不是"一天一批"而是"元任务(support/query)"，故在 day_batch 之外
另提供 hist_task / offline_tasks / update_grid 三个元任务采样接口。

数据由 input_data_update/prepare_local_data.py 构造并落盘到 data/derived/：
panel.npz(对齐矩阵) + tech53.dat(53因子memmap) + expo.dat(10暴露memmap)。
被 train/update/evaluate 各阶段共同 import，不单独运行。
"""
import numpy as np
import pandas as pd
import torch

from config import CFG, DERIVED, PANEL_PATH, EXPO_PATH, device


class PanelData:
    """持有对齐后的行情/因子/暴露/市场状态/标签矩阵，提供元学习所需的批与任务。"""

    def __init__(self):
        z = np.load(PANEL_PATH, allow_pickle=True)
        self.DATES = pd.DatetimeIndex(z['DATES'])
        self.CODES = z['CODES']
        self.PRICE = z['PRICE']          # (T,J,5) 后复权 OHLC + 原始 volume
        self.AMT = z['AMT']              # (T,J)   成交额
        self.VWAP_H = z['VWAP_H']        # (T,J)   后复权 vwap（已 ffill）
        self.MASTER_POOL = z['MASTER_POOL']   # (T,J) VALID ∧ PIT沪深300成分
        self.Y_RANK = z['Y_RANK']        # (T,J)   [0,1] 排序标签
        self.POOL_LBL = z['POOL_LBL']    # (T,J)   MASTER_POOL ∧ 标签可实现
        self.LBL_OK = z['LBL_OK']        # (T,J)   标签可实现（建仓/平仓日均成交）
        self.MSTATE = z['MSTATE'].astype('float32')   # (T,183) 市场状态
        if CFG['mclean']:                             # 输入层清理（DA_MCLEAN=1）：状态依赖EWMA + 120维PC去噪
            from model_core.mstate_clean import clean_mstate
            self.MSTATE = clean_mstate(self.MSTATE, self.DATES)
        self.T, self.J = self.MASTER_POOL.shape

        # tech53.dat / expo.dat 由 build_tech53 / build_drm 用 np.lib.format.open_memmap 写成 .npy
        # （带 128 字节头）。必须同样用 open_memmap 读——裸 np.memmap 不跳头会整体错位
        # 32 个 float32，令 63/70 维输入取到相邻股票的数据。
        _ts = tuple(int(v) for v in z['tech_shape'])                  # (T,J,53)
        self.TECH = np.lib.format.open_memmap(DERIVED / 'tech53.dat', mode='r')
        assert self.TECH.shape == _ts, f'tech53.dat 形状 {self.TECH.shape}≠{_ts}'
        _st = np.load(DERIVED / 'tech53_stats.npz')
        self.TECH_MU, self.TECH_SD = _st['MU'], _st['SD']            # (T,53) 全市场逐日均值/标准差
        self.EXPO = np.lib.format.open_memmap(EXPO_PATH, mode='r')
        assert self.EXPO.shape == (self.T, self.J, 10), \
            f'{EXPO_PATH.name} 形状 {self.EXPO.shape}≠{(self.T, self.J, 10)}'

        self._d2i = {d: i for i, d in enumerate(self.DATES)}
        self.MM_DAYS = np.where(np.isfinite(self.MSTATE).all(axis=1))[0]   # 183 维完整的可训日
        self._MMD = np.asarray(self.MM_DAYS)
        self._LBL_DAYS = self._MMD[self.POOL_LBL[self._MMD].sum(axis=1) > 0]   # 有标签即可训(旧300池哑门min_cs已删)
        self.DEV = device()
        self._selfcheck()

    def _selfcheck(self):
        """启动自检:抽一日验 53 因子 z-score 尺度,静默拦截 memmap 头偏移/通道错排
        (正确路径 |z|max≈10、std≈0.8;错位路径 std 会炸到 1e8 量级)。"""
        if not len(self._MMD):
            return
        t = int(self._MMD[len(self._MMD) // 2])
        sl = slice(t - CFG['window'] + 1, t + 1)
        jj = np.where(self.MASTER_POOL[t])[0][:500]
        if not len(jj):
            return
        f53 = np.asarray(self.TECH[sl])[:, jj, :]
        z = (f53 - self.TECH_MU[sl][:, None, :]) / (self.TECH_SD[sl][:, None, :] + CFG['eps'])
        z = z[np.isfinite(z)]
        if not z.size:
            return
        bad = float(np.mean(np.abs(z) > 10))
        assert bad < 0.05, (f'53因子 z-score 异常: |z|>10 占 {bad:.1%}(正常<1%)，std={float(z.std()):.3g}。'
                            f'典型病因=tech53.dat/expo.dat 读写格式不一致导致通道错位，'
                            f'检查 build_tech53/build_drm(open_memmap 写) 与本文件(open_memmap 读)是否一致。')

    # ------------------------------------------------------------------ 批
    def day_batch(self, t):
        """交易日 t 的截面批：X (n,40,70) + 市场状态 m (183,) + 池内列索引 jj。
        70 维 = 7 量价(÷窗末→截面MAD) | 53 因子(逐窗日全市场 z) | 10 暴露(原样)。"""
        sl = slice(t - CFG['window'] + 1, t + 1)
        _e_ok = np.isfinite(self.EXPO[sl]).all(axis=(0, 2))          # 40 日暴露窗全有效才准入
        jj = np.where(self.MASTER_POOL[t] & _e_ok)[0]
        p7 = np.concatenate([self.PRICE[sl][:, jj, :],
                             self.AMT[sl][:, jj, None], self.VWAP_H[sl][:, jj, None]], axis=-1)  # (40,n,7)
        p7 = np.nan_to_num(p7, nan=0.0)
        ratio = p7 / (np.abs(p7[-1:, :, :]) + CFG['eps'])
        med = np.nanmedian(ratio, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(ratio - med), axis=1, keepdims=True)
        x_p = np.clip(ratio, med - CFG['mad_k'] * mad, med + CFG['mad_k'] * mad)
        f53 = self.TECH[sl][:, jj, :]
        x_f = np.nan_to_num((f53 - self.TECH_MU[sl][:, None, :]) / (self.TECH_SD[sl][:, None, :] + CFG['eps']),
                            nan=0.0)
        x_e = self.EXPO[sl][:, jj, :]
        x = np.concatenate([x_p, x_f, x_e], axis=-1).transpose(1, 0, 2)   # (n,40,70)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.MSTATE[t].copy()), jj)

    def da_day(self, t, gn):
        """DoubleAdapt 批：(x, robust-z 市场状态 mz, rank 标签 y, 有效掩码 ok, jj)。全截面前向纪律。"""
        x, m, jj = self.day_batch(t)
        x, mz = x.to(self.DEV), gn.transform(m.numpy()).to(self.DEV)
        y = torch.from_numpy(self.Y_RANK[t, jj].astype('float32')).to(self.DEV)
        return x, mz, y, torch.isfinite(y), jj

    # ------------------------------------------------------------------ 元任务采样
    def lbl_in(self, lo, hi):
        """[lo,hi) 内的标签日（可训日 ∩ 池内标签充足）。"""
        return [int(t) for t in self._LBL_DAYS[(self._LBL_DAYS >= lo) & (self._LBL_DAYS < hi)]]

    def hist_task(self, s):
        """更新日 s 的最近已实现任务：query 末日=s−12，其标签 t+11 在 s−1 实现（严格 <s）。"""
        q_lo, q_hi = s - CFG['da_incr'] - CFG['label_h'], s - CFG['label_h']
        sup, qry = self.lbl_in(q_lo - CFG['da_support'], q_lo), self.lbl_in(q_lo, q_hi)
        if qry:
            assert max(qry) + CFG['label_h'] < s, 'hist task 标签未实现'
        return sup, qry

    def ft_block(self, s):
        """内层适应用的最近 20 日已实现标签块。"""
        return self.lbl_in(s - CFG['label_h'] - CFG['da_incr'], s - CFG['label_h'])

    def offline_tasks(self):
        """离线元训练任务（2013—da_off_end）：步长 20 的相邻 support+query；query 标签不溢入验证年。"""
        val_t0 = int(self._MMD[self.DATES[self._MMD].year >= CFG['da_val_year']][0])
        off = self._MMD[self.DATES[self._MMD].year <= CFG['da_off_end']]
        tasks = []
        for i in range(0, len(off) - CFG['da_support'] - CFG['da_query'], CFG['da_incr']):
            sup = self.lbl_in(off[i], off[i + CFG['da_support']])
            j = i + CFG['da_support'] + CFG['da_query']
            q_end = int(off[j]) if j < len(off) else int(off[-1]) + 1
            qry = self.lbl_in(off[i + CFG['da_support']], q_end)
            if len(sup) >= 10 and len(qry) >= 10 and max(qry) + CFG['label_h'] < val_t0:
                tasks.append((sup, qry))
        return tasks

    def update_grid(self, y0, y1):
        """在线/轨迹验证更新网格：从 y0 首个可训日起每 20 个可训日一个 s（跨年连续，不年初重置）。"""
        ii = np.where(self.DATES[self._MMD].year == y0)[0]
        grid = []
        for i in range(int(ii[0]), len(self._MMD), CFG['da_incr']):
            s = int(self._MMD[i])
            if self.DATES[s].year > y1:
                break
            grid.append((s, [int(t) for t in self._MMD[i:i + CFG['da_incr']] if self.DATES[t].year <= y1]))
        return grid

    def fridays(self, ts):
        """一组交易日中每个 W-FRI 周的最后一日（周频打分/验收）。"""
        ser = pd.Series(list(ts), index=pd.DatetimeIndex(self.DATES[list(ts)]))
        return [int(v) for v in ser.groupby(ser.index.to_period('W-FRI')).last()]

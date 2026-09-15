# -*- coding: utf-8 -*-
"""快慢双层 · 日频线性修正器（DA_FAST=1 时在线启用）。

动机（Step3「快慢双层」）：
  在线元更新是 20 个交易日一拍的「慢层」；风格切换期 score 与标签的截面关系（尤其沿风格
  暴露方向）可能在两次更新之间发生系统性偏移。本修正器在每个打分日执行一个「快层」：

  1. 用最近 fast_window 个「标签已实现」交易日（u + label_h < t）的样本
     (score, y_rank, 10 维 DRM 暴露 robust-z)，OLS 拟合截面映射
         ŷ = α + β_s·(score − 0.5) + Σ_j β_zj·z_j；
  2. 对系数施加强收缩先验（默认：截距向 0.5 收缩到 30%、score 斜率向 1 收缩到 50%、
     风格系数向 0 收缩到 30%；样本天数不足按比例进一步缩），得到修正
         s' = s + clip(ŷ − s, ±fast_cap)；
  3. 平时证据弱 → 修正 ≈ 0（不闹）；断点期新标签快速累积 → 修正先于慢层跟上。

时点纪律：打分日 t 只吸收标签已实现日（u + label_h < t）；当日原始 score 在应用修正后才
入缓冲，供未来拟合。断点续跑：从 sidecar（FAST_RAW_DIR/raw_{年}.pq，update.py 落盘）重建
最近若干打分日的缓冲与滚动统计，再继续增量累积。
"""
from collections import deque

import numpy as np
import pandas as pd

from config import CFG, FAST_RAW_DIR

_N_FEAT = 12   # [1, score-0.5, z1..z10]


class FastCorrector:
    """滚动拟合 score→标签截面映射并施加收缩修正；接口：apply_day(df, t) / rebuild_from_sidecar。"""

    def __init__(self, panel):
        self.panel = panel
        self._cidx = pd.Index(panel.CODES)
        self.buf = {}                 # t -> (jj, raw_scores)，已打分、尚未吸收的日
        self._pending = []            # buf 中待吸收的 t 列表
        self.stats = deque()          # (t, XtX, Xty, n)，最近 fast_window 个已实现日
        self.XtX = np.zeros((_N_FEAT, _N_FEAT))
        self.Xty = np.zeros(_N_FEAT)
        self.n_rows = 0
        self.beta = None              # 当前系数（含先验收缩）；None=证据不足不修正
        self.activated = False        # 首次生效标记（供日志）

    # ------------------------------------------------------------------ 特征/样本
    def _day_features(self, t, jj):
        """当日截面 10 维 DRM 暴露的 robust-z（截 ±3）。"""
        e = np.asarray(self.panel.EXPO[t, jj, :], dtype='float64')
        med = np.nanmedian(e, axis=0)
        mad = np.nanmedian(np.abs(e - med), axis=0) * 1.4826 + 1e-9
        return np.nan_to_num(np.clip((e - med) / mad, -3.0, 3.0), nan=0.0)

    def _day_xy(self, t):
        """已打分日 t 的 (X, y)；样本不足返回 None。"""
        item = self.buf.get(t)
        if item is None:
            return None
        jj, s = item
        y = self.panel.Y_RANK[t, jj].astype('float64')
        ok = np.isfinite(y)
        if int(ok.sum()) < 100:
            return None
        z = self._day_features(t, jj)[ok]
        X = np.column_stack([np.ones(int(ok.sum())), s[ok] - 0.5, z])
        return X, y[ok]

    def _absorb(self, t):
        """把日 t 并入滚动统计（并弹出过期日）。"""
        r = self._day_xy(t)
        self.buf.pop(t, None)
        if r is None:
            return
        X, y = r
        A = X.T @ X
        b = X.T @ y
        self.XtX += A
        self.Xty += b
        self.n_rows += int(len(y))
        self.stats.append((t, A, b, int(len(y))))
        while len(self.stats) > CFG['fast_window']:
            _, A0, b0, n0 = self.stats.popleft()
            self.XtX -= A0
            self.Xty -= b0
            self.n_rows -= n0

    def _solve(self):
        """OLS + 强收缩先验。证据不足返回 None。"""
        if len(self.stats) < CFG['fast_min_days'] or self.n_rows < CFG['fast_min_rows']:
            return None
        try:
            beta_ols = np.linalg.solve(self.XtX + 1e-8 * np.eye(_N_FEAT), self.Xty)
        except np.linalg.LinAlgError:
            return None
        beta0 = np.zeros(_N_FEAT)
        beta0[0] = 0.5
        beta0[1] = 1.0
        sh = np.array([CFG['fast_sh_a'], CFG['fast_sh_s']] + [CFG['fast_sh_z']] * 10)
        w = min(1.0, len(self.stats) / float(CFG['fast_window']))     # 天数不足按比例缩
        return beta0 + sh * w * (beta_ols - beta0)

    def step_to(self, t):
        """推进到打分日 t：吸收所有标签已实现日（u + label_h < t），重解系数。"""
        ready = [u for u in self._pending if u + CFG['label_h'] < t]
        for u in sorted(ready):
            self._pending.remove(u)
            self._absorb(u)
        self.beta = self._solve()

    # ------------------------------------------------------------------ 应用
    def apply_day(self, df, t):
        """对打分日 t 的 DataFrame(date/code/score) 应用修正；原始 score 入缓冲供未来拟合。"""
        jj = self._cidx.get_indexer(df['code'].values)
        ok = jj >= 0
        s = df['score'].values.astype('float64')
        self.step_to(t)
        if self.beta is not None and bool(ok.any()):
            z = self._day_features(t, jj[ok])
            s_ok = s[ok]
            pred = self.beta[0] + self.beta[1] * (s_ok - 0.5) + z @ self.beta[2:]
            corr = np.clip(pred - s_ok, -CFG['fast_cap'], CFG['fast_cap'])
            s2 = s.copy()
            s2[ok] = s_ok + corr
            df = df.copy()
            df['score'] = s2.astype('float32')
            if not self.activated:
                self.activated = True
                print(f'  [fast] 修正器生效：{len(self.stats)} 个标签日 / {self.n_rows:,} 行', flush=True)
        self.buf[t] = (jj[ok], s[ok])
        self._pending.append(t)
        return df

    # ------------------------------------------------------------------ 断点续跑
    def rebuild_from_sidecar(self, resume_year):
        """从 sidecar（fast_raw{...}/raw_{resume_year[-1]}.pq）重建最近打分日缓冲。"""
        files = [FAST_RAW_DIR / f'raw_{resume_year}.pq', FAST_RAW_DIR / f'raw_{resume_year - 1}.pq']
        dfs = [pd.read_parquet(f) for f in files if f.exists()]
        if not dfs:
            print('  [fast] 未找到 raw sidecar，修正器从零重建', flush=True)
            return
        raw = pd.concat(dfs, ignore_index=True)
        raw['date'] = pd.to_datetime(raw['date'])
        raw = raw.sort_values(['date', 'code']).drop_duplicates(['date', 'code'], keep='last')
        keep_n = CFG['fast_window'] + CFG['label_h'] + 10
        keep_dates = np.sort(raw['date'].unique())[-keep_n:]
        raw = raw[raw['date'].isin(keep_dates)]
        for d, g in raw.groupby('date'):
            t = self.panel._d2i.get(pd.Timestamp(d))
            if t is None:
                continue
            jj = self._cidx.get_indexer(g['code'].values)
            ok = jj >= 0
            self.buf[t] = (jj[ok], g['score'].values.astype('float64')[ok])
            self._pending.append(t)
        print(f'  [fast] sidecar 重建：{raw["date"].nunique()} 个打分日入缓冲', flush=True)

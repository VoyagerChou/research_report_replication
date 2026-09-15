# -*- coding: utf-8 -*-
"""DoubleAdapt 元训练器（离线双层优化 + 轨迹验证 + FOMAML 基元）。

训练循环为 DoubleAdapt 的双层优化：
  内层(support)：对 φ 走一步继承状态 Adam 得更新量 delta；
  外层(query)：经适配器 H_inv 算 query 损失，反传同时更新 φ 与适配器 ψ；
  REINFORCE 标签正则：按适应是否有效决定把 H(y) 拉近还是推离 y。
在线滚动（每 20 日 meta-update→适应→打分落库）见 update/update.py。
"""
import copy
import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import CFG, MODEL_DIR, VTAG, ATAG, DA_VAL_METRIC, DA_OPT
from model_core.master_backbone import wmse, train_loss
from model_core.market_gate import GateNormalizer
from model_core.adapter import DataAdapter, MetaMasterNet, ParamSwap
from model_core.master_train import train_master_year


class DoubleAdaptTrainer:
    """持有 PanelData，实现 DoubleAdapt 的内/外层优化、离线元训练、轨迹验证与打分。"""

    def __init__(self, panel):
        self.panel = panel
        self.DEV = panel.DEV

    @staticmethod
    def _named(phi):
        return [(k, p) for k, p in phi.named_parameters()]

    # ------------------------------------------------------------------ FOMAML 基元
    def inner_delta(self, phi, adapter, sup_ts, gn, opt, lr_scale=1.0):
        """内层 1 步继承状态 Adam 的更新量 delta（全 detach）。逐日梯度累加=和的梯度，FOMAML
        故 ψ 的 support 路径不回传（与参照线 g.detach 同义）。
        lr_scale：信任域调制倍数（DA_EVENT 事件节点放大内层适应预算；默认 1.0=原行为）。"""
        named = self._named(phi)
        params = [p for _, p in named]
        gsum = [torch.zeros_like(p) for p in params]
        for t in sup_ts:
            x, mz, y, ok, _ = self.panel.da_day(t, gn)
            s_ok = adapter.strengths(x[ok])
            pred = phi(adapter.G(x), mz)                                # 全截面前向，标签掩码只进损失
            loss = train_loss(pred[ok], adapter.H(y[ok], s_ok)) / len(sup_ts)
            for g_, gnew in zip(gsum, torch.autograd.grad(loss, params, allow_unused=True)):
                if gnew is not None:
                    g_ += gnew.detach()
        if DA_OPT == 'ft':                                          # 可学习内层lr(限幅防跑飞)
            lr = float(adapter.log_ilr.detach().clamp(math.log(1e-5), math.log(1e-1)).exp()) * lr_scale
        else:
            lr = CFG['da_inner_lr'] * lr_scale
        b1, b2, eps = 0.9, 0.999, 1e-8
        delta = {}
        for (k, p), g in zip(named, gsum):                             # 继承外层 Adam 的动量状态
            st = opt.state.get(p, {}) if opt is not None else {}
            m_ = st['exp_avg'].detach().clone() if 'exp_avg' in st else torch.zeros_like(p)
            v_ = st['exp_avg_sq'].detach().clone() if 'exp_avg_sq' in st else torch.zeros_like(p)
            t0 = st.get('step', 0)
            t_ = float(t0.item() if torch.is_tensor(t0) else t0) + 1.0
            m_ = b1 * m_ + (1 - b1) * g
            v_ = b2 * v_ + (1 - b2) * g * g
            delta[k] = (lr * (m_ / (1 - b1 ** t_)) / ((v_ / (1 - b2 ** t_)).sqrt() + eps)).detach()
        return delta

    def outer_step(self, phi, adapter, opt, sup_ts, qry_ts, gn):
        """一个元任务的外层步：内层 delta → query 损失(逐日 backward 累加) → REINFORCE 正则 → clip+step。"""
        opt.zero_grad()
        delta = self.inner_delta(phi, adapter, sup_ts, gn, opt)
        with torch.no_grad():                                          # REINFORCE 基线：未适应 φ 的 query 损失
            L_phi = 0.0
            for t in qry_ts:
                x, mz, y, ok, _ = self.panel.da_day(t, gn)
                pred0 = adapter.H_inv(phi(adapter.G(x), mz), adapter.strengths(x))
                L_phi += float(train_loss(pred0[ok], y[ok])) / len(qry_ts)
        L_q = 0.0
        with ParamSwap(phi, delta):                                    # 适应后 φ 上的 query 损失，逐日反传
            for t in qry_ts:
                x, mz, y, ok, _ = self.panel.da_day(t, gn)
                pred = adapter.H_inv(phi(adapter.G(x), mz), adapter.strengths(x))
                loss = train_loss(pred[ok], y[ok]) / len(qry_ts)
                loss.backward()
                L_q += float(loss)
        if DA_OPT == 'ft':
            # 内层lr的一阶元梯度:fast=θ−delta,delta=lr·D(D与lr无关) ⇒ dL_q/d(log lr)=−Σ grad_fast·delta。
            # ParamSwap 块内 backward 后 φ.grad 恰是快权重处的 query 梯度(REINFORCE 正则不碰 φ),直接内积。
            _lg = sum((p.grad.detach() * delta[k]).sum() for k, p in self._named(phi) if p.grad is not None)
            adapter.log_ilr.grad = (-_lg).reshape(()).to(adapter.log_ilr.device)
        coef = (L_phi - L_q) / CFG['da_sigma'] + CFG['da_reg']         # 适应有效→把 H(y) 拉近 y，反之推离
        for t in sup_ts:                                              # L_reg 只训 ψ（H 不含 φ）
            x, _, y, ok, _ = self.panel.da_day(t, gn)
            (coef * ((adapter.H(y[ok], adapter.strengths(x[ok])) - y[ok]) ** 2).mean() / len(sup_ts)).backward()
        torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(adapter.parameters()), CFG['da_grad_clip'])
        opt.step()
        if DA_OPT == 'ft':
            adapter.log_ilr.data.clamp_(math.log(1e-5), math.log(1e-1))
        return L_q

    @torch.no_grad()
    def score_days(self, phi, adapter, delta, ts, gn):
        """fast=φ−delta 换值后逐日打分，H⁻¹ 还原。返回长表行列表。"""
        phi.eval()
        rows = []
        with ParamSwap(phi, delta):
            for t in ts:
                x, mz, _, _, jj = self.panel.da_day(t, gn)
                sc = adapter.H_inv(phi(adapter.G(x), mz), adapter.strengths(x)).cpu().numpy()
                rows.append(pd.DataFrame({'date': self.panel.DATES[t],
                                          'code': self.panel.CODES[jj], 'score': sc}))
        return rows

    def adapt_delta(self, phi, adapter, ft_ts, gn, opt, lr_scale=1.0):
        """推理期内层适应：与训练同一 delta 计算（ψ/φ 全 detach），无 ft 数据则零 delta。
        lr_scale：信任域调制——事件节点（DA_EVENT）放大内层适应预算；默认 1.0=原行为。"""
        if not len(ft_ts):
            return {k: torch.zeros_like(p) for k, p in self._named(phi)}
        phi.eval()
        with torch.enable_grad():
            return self.inner_delta(phi, adapter, ft_ts, gn, opt, lr_scale=lr_scale)

    # ------------------------------------------------------------------ 轨迹验证 / 预热
    def walk(self, phi, adapter, opt, gn, y0, y1, collect=True):
        """与在线同构：元更新→适应→周五打分。collect=True 返回 (加权MSE, 平均IC, RankIC, Top10%标签分位)。
        早停判据由 config.DA_VAL_METRIC 选（wmse=定稿默认）：wmse=轨迹验证加权 MSE |
        ic=平均 Pearson IC | rankic=全截面 Spearman |
        topq=多头端判据（打分前 10% 股票的真实标签平均分位；标签即截面排序归一 [0,1]，
             故 0.5=无技能、0.95=完美，只度量头部排序质量）。
        四指标每轮全记，便于事后对照。"""
        opt.param_groups[0]['lr'] = CFG['da_online_lr']                # φ 在线 lr；ψ 组保持 0.01
        p = self.panel
        wmses, ics, rics, tops = [], [], [], []
        for s, win in p.update_grid(y0, y1):
            sup, qry = p.hist_task(s)
            if len(sup) >= 10 and len(qry) >= 10:
                phi.train()
                self.outer_step(phi, adapter, opt, sup, qry, gn)
            if not collect:
                continue
            delta = self.adapt_delta(phi, adapter, p.ft_block(s), gn, opt)
            for df in self.score_days(phi, adapter, delta, p.fridays(win), gn):
                t = p._d2i[df['date'].iat[0]]
                jj = pd.Index(p.CODES).get_indexer(df['code'])
                y = p.Y_RANK[t, jj]
                ok = np.isfinite(y) & p.LBL_OK[t, jj]
                if ok.sum() >= 2:                                      # 相关系数最低样本
                    pred, yy = df['score'].values[ok], y[ok]
                    w = 1.0 / (1.0 + np.exp(-CFG['m_gamma'] * (pred - 0.5)))   # Sigmoid(γ(ŷ−0.5))：多头端加权
                    wmses.append(float(np.mean(w * (pred - yy) ** 2)))         # 与训练 wmse 同形式
                    ics.append(pd.Series(pred).corr(pd.Series(yy), method='pearson'))    # 平均IC(前作字面)
                    rics.append(pd.Series(pred).corr(pd.Series(yy), method='spearman'))  # RankIC(监控)
                    _k = max(1, int(round(len(pred) * CFG['val_top_frac'])))              # 多头端判据
                    tops.append(float(yy[np.argsort(-pred)[:_k]].mean()))                 # 前10%的标签平均分位
        return (float(np.nanmean(wmses)) if wmses else np.nan,
                float(np.nanmean(ics)) if ics else np.nan,
                float(np.nanmean(rics)) if rics else np.nan,
                float(np.nanmean(tops)) if tops else np.nan)

    # ------------------------------------------------------------------ 离线元训练
    def train_offline(self, quiet=False):
        """离线元训练：φ(冷启动=随机 / 暖启动=载入预训练 MASTER) + 新 ψ，逐 epoch 轨迹验证早停。
        存 da_offline{VTAG}{ATAG}.pt（判据/架构变体产物后缀隔离，与 wmse 版可并行互不覆盖）。"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)           # 冷启动不经 train_master_year,原生池目录可能不存在
        ckpt = MODEL_DIR / f'da_offline{VTAG}{ATAG}.pt'
        if ckpt.exists():
            return ckpt
        torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])
        phi = MetaMasterNet().to(self.DEV)
        adapter = DataAdapter().to(self.DEV)
        if CFG['da_warm_start']:                                        # 暖启动：φ 载入预训练 MASTER + 同款 gn
            pre_ckpt = train_master_year(self.panel, CFG['da_pretrain_year'], quiet=quiet)
            pre = torch.load(pre_ckpt, map_location='cpu')
            phi.net.load_state_dict(pre['model']); phi.gate.load_state_dict(pre['gate'])
            gn = GateNormalizer().load_state_dict(pre['gn'])
            src = str(pre_ckpt.name)
        else:                                                          # 冷启动：φ 随机 + gn 在离线窗(≤da_off_end)拟合（同 notebook M7 冷启动）
            off_days = [int(t) for t in self.panel._MMD
                        if self.panel.DATES[t].year <= CFG['da_off_end']]
            gn = GateNormalizer().fit(self.panel.MSTATE[off_days])
            src = 'cold_start(random_phi)'
        opt = torch.optim.Adam([{'params': list(phi.parameters()), 'lr': CFG['da_outer_lr']},
                                {'params': list(adapter.parameters()), 'lr': CFG['da_psi_lr']}])
        tasks = self.panel.offline_tasks()
        rng = np.random.default_rng(CFG['seed'])
        best, best_state, best_ep, bad = np.inf, None, -1, 0          # 统一最小化(IC 判据取负号)
        best_ic, stall = np.nan, 0                                     # stall:ft 模式 plateau 退火计数

        # ---- 断点续跑（单轮 epoch 约 10 分钟、上限 100 轮，断线后可从存档继续）----
        rs_path = MODEL_DIR / f'da_offline_resume{VTAG}{ATAG}.pt'
        ep0 = 0
        if rs_path.exists():
            rs = torch.load(rs_path, map_location='cpu')
            phi.load_state_dict(rs['phi']); adapter.load_state_dict(rs['adapter']); opt.load_state_dict(rs['opt'])
            gn = GateNormalizer().load_state_dict(rs['gn'])
            rng.bit_generator.state = rs['rng_np']                     # 任务顺序与不中断时逐位一致
            torch.set_rng_state(rs['rng_torch'])
            best, best_ic, best_ep, bad = rs['best'], rs['best_ic'], rs['best_ep'], rs['bad']
            stall = rs.get('stall', 0)
            best_state = rs['best_state']
            ep0 = rs['ep'] + 1
            print(f'续跑：从 epoch{ep0 + 1} 起（已完成 {ep0} 轮，best={best:.5f}@{best_ep + 1} bad={bad}）', flush=True)

        for ep in range(ep0, CFG['da_max_epoch']):
            phi.train()
            for i in tqdm(rng.permutation(len(tasks)), desc=f'DA元训练 ep{ep + 1}/{CFG["da_max_epoch"]}',
                          leave=False, disable=quiet):
                self.outer_step(phi, adapter, opt, *tasks[i], gn)
            bak = (copy.deepcopy(phi.state_dict()), copy.deepcopy(adapter.state_dict()),
                   copy.deepcopy(opt.state_dict()))                    # 轨迹验证在副本轨道，测完恢复
            vwmse, vip, vir, vtq = self.walk(phi, adapter, opt, gn, CFG['da_val_year'], CFG['da_val_year'], collect=True)
            phi.load_state_dict(bak[0]); adapter.load_state_dict(bak[1]); opt.load_state_dict(bak[2])
            crit = {'wmse': vwmse, 'ic': -vip, 'rankic': -vir, 'topq': -vtq}[DA_VAL_METRIC]   # 统一最小化
            if not quiet:
                _m = {'wmse': 'wMSE', 'ic': 'IC', 'rankic': 'RankIC', 'topq': 'TopQ'}[DA_VAL_METRIC]
                print(f'DA offline epoch{ep + 1} wMSE={vwmse:.5f} IC={vip:+.4f} RankIC={vir:+.4f} '
                      f'TopQ={vtq:.4f} [判据={_m}] best={best:.5f}@{best_ep + 1} bad={bad}', flush=True)
            if crit < best - 1e-7:
                best, best_ic, best_ep, bad, stall = crit, vir, ep, 0, 0
                best_state = (copy.deepcopy(phi.state_dict()), copy.deepcopy(adapter.state_dict()),
                              copy.deepcopy(opt.state_dict()))
            else:
                bad += 1
                if DA_OPT == 'ft':
                    stall += 1
                    if stall >= CFG['plateau_stall']:                  # plateau 退火:停滞3轮→全组lr减半
                        for g_ in opt.param_groups:
                            g_['lr'] *= CFG['plateau_factor']
                        stall = 0
                        print(f'  ⤷ plateau: lr 减半 → φ={opt.param_groups[0]["lr"]:.2e} '
                              f'ψ={opt.param_groups[1]["lr"]:.2e}', flush=True)
            torch.save({'phi': phi.state_dict(), 'adapter': adapter.state_dict(), 'opt': opt.state_dict(),
                        'gn': gn.state_dict(), 'rng_np': rng.bit_generator.state,
                        'rng_torch': torch.get_rng_state(), 'ep': ep, 'best': best, 'best_ic': best_ic,
                        'best_ep': best_ep, 'bad': bad, 'stall': stall, 'best_state': best_state}, rs_path)   # 每轮存档
            if bad >= CFG['da_patience']:
                break
        phi.load_state_dict(best_state[0]); adapter.load_state_dict(best_state[1]); opt.load_state_dict(best_state[2])
        self.walk(phi, adapter, opt, gn, CFG['da_val_year'], CFG['da_val_year'], collect=False)   # 预热到验证年末
        torch.save({'phi': phi.state_dict(), 'adapter': adapter.state_dict(), 'opt': opt.state_dict(),
                    'gn': gn.state_dict(), 'meta': dict(best_epoch=best_ep + 1, val_metric=DA_VAL_METRIC,
                    val_best=round(best, 5), val_rankic_at_best=round(best_ic, 4),
                    n_tasks=len(tasks), init=src)}, ckpt)
        rs_path.unlink(missing_ok=True)                                # 训练完成，清掉续跑存档
        return ckpt

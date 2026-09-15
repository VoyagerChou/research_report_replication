# -*- coding: utf-8 -*-
"""集中配置：路径约定、实验变体开关、全部超参（CFG 字典）。

被所有模块 import，本身不产生副作用。每个参数的取值出处（研报字面/论文官方/复现自定）
逐条记录在 `参数溯源审计表.md`，此处注释只说明参数的含义。

实验变体开关（DA_* 环境变量）：默认值即定稿主线；非默认值为已测试并否决的实验变体，
保留供复核。变体产物经 VTAG/ATAG/DTAG 后缀与主线完全隔离，可同机并行互不覆盖。
"""
import os
from pathlib import Path

# ---- 路径约定 ----
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
IN_DIR = DATA_DIR / 'factor_values'      # 因子值落库目录（训练/推理消费的成品）
DERIVED = DATA_DIR / 'derived'            # 共享面板与对齐矩阵

# ---- 标签口径 ----
# 标签 = 未来 10 日 vwap 收益的全 A 截面排序映射到 [0,1]（Y_RANK），与 Sigmoid 输出头配套。
# 训练与评价均为全 A 单一因子；池内评价为同一因子在该指数成分内重排。
PANEL_PATH = DERIVED / 'panel.npz'   # 占位，DTAG 定义后重绑定（见下）
LABELS_PATH = DERIVED / 'labels.npz'
MODEL_DIR = DATA_DIR / 'models_weights'
DRM_MODEL_DIR = DATA_DIR / 'models_weights'          # DRM 权重与 alpha 权重同目录（全市场共享）

# ---- 早停判据 DA_VAL_METRIC ----
# 离线元训练的模型选择判据。wmse=定稿默认（轨迹验证的加权 MSE）；
# ic=平均 Pearson IC | rankic=平均 Spearman | topq=打分前 val_top_frac 股票的标签平均分位，
# 均为实验变体。DA 线与 MASTER 线共用此开关，变体产物挂 VTAG 后缀。
DA_VAL_METRIC = os.environ.get('DA_VAL_METRIC', 'wmse')
assert DA_VAL_METRIC in ('wmse', 'ic', 'rankic', 'topq'), f'未知 DA_VAL_METRIC={DA_VAL_METRIC}'
# ---- 架构变体开关（环境变量注入，默认即定稿）----
# DA_DMODEL：MASTER 骨干隐藏维（默认 64）。DA_BETA：门控 softmax 温度（默认 5.0）。
# DA_SEED：随机种子（默认 42）。DA_GAMMA：加权 MSE 的 γ（默认 5.0，研报区间 5~10 的下端点）。
_DM = int(os.environ.get('DA_DMODEL', 64))
_BT = float(os.environ.get('DA_BETA', 5.0))
_SD = int(os.environ.get('DA_SEED', 42))
_GM = float(os.environ.get('DA_GAMMA', 5.0))     # 加权 MSE 多头端权重陡度 γ
# DA_DROP：两个注意力模块的 dropout（默认 0.5，官方 MASTER 值）。
_DP = float(os.environ.get('DA_DROP', 0.5))
# ---- 标签适配器几何 DA_H ----
# logit=定稿默认（H 仿射作用在 logit 域，构造上不出界）；affine=研报字面的 y 域平直仿射（实验变体）。
# ---- 训练损失 DA_LOSS ----
# wmse=定稿默认（加权 MSE，w=Sigmoid(γ(ŷ−0.5))）；ic=−截面 Pearson | rankwmse=秩空间加权 MSE，
# 均为已测试并否决的实验变体（产物 _lic / _lrw 后缀隔离）。
# ---- ABCD 头部区分度实验（2026-09 新增，均为实验变体，产物按后缀隔离）----
# prank  =A 头部加权 pairwise 排序损失（_prk）：头部内正样本×采样负样本，边界对加权；
# listnet=B ListNet 式 listwise 交叉熵（_lst）：softmax(z) 对 softmax(y) 对齐头部；
# focal  =C wMSE + focal 顶部分类辅助（_foc）：复用 Sigmoid 输出做"是否 top10%"二分类；
# hybrid =D wMSE + pairwise 混合（_hyb）：在定稿损失上加排序项，最低风险对照。
DA_LOSS = os.environ.get('DA_LOSS', 'wmse')
assert DA_LOSS in ('wmse', 'ic', 'rankwmse', 'prank', 'listnet', 'focal', 'hybrid'), \
    f'未知 DA_LOSS={DA_LOSS}（wmse|ic|rankwmse|prank|listnet|focal|hybrid）'
DA_H = os.environ.get('DA_H', 'logit')
assert DA_H in ('logit', 'affine'), f'未知 DA_H={DA_H}（logit|affine）'
# ---- DRM 参数组 DA_DRM ----
# report=定稿默认（窗 40 日、GAT dropout 0.3，正式研报参数组）；deck=路演 deck 参数组
# （窗 20 日、dropout 0.5，实验变体）。deck 变体的 DRM 整链产物挂 _drmdk 后缀，主线共享件零覆盖。
# ---- wMSE 权重挂靠 DA_W ----
# pred=定稿默认（权重挂预测值 ŷ，研报公式字面）；label=权重挂真实标签（实验变体，_wlab 后缀）。
DA_W = os.environ.get('DA_W', 'pred')
assert DA_W in ('pred', 'label'), f'未知 DA_W={DA_W}（pred|label）'
DA_DRM = os.environ.get('DA_DRM', 'report')
assert DA_DRM in ('report', 'deck'), f'未知 DA_DRM={DA_DRM}（report|deck）'
# ---- 训练模式包 DA_OPT ----
# base=定稿默认；ft=三项训练改造合体的实验变体（DRM 暖启动年 epoch0 过验证、plateau 退火、
# 可学习内层 lr）。ft 时 DRM 链与 alpha 产物均挂 ft 后缀；base 与主线逐位一致。
DA_OPT = os.environ.get('DA_OPT', 'base')
assert DA_OPT in ('base', 'ft'), f'未知 DA_OPT={DA_OPT}（base|ft）'

# ---- 事件驱动更新 DA_EVENT（2026-09 新增：Step2 改调度）----
# 1=读 data/derived/switch_alarm_daily.csv（diag/step5 风格切换检测器定稿输出），把报警段
# 起点并入在线更新节点：报警当日立即元更新+内层适应（不等 20 日排期），且该节点内层适应
# 学习率放大 DA_EVENT_TRUST 倍（信任域调制）。只影响在线阶段（update）。
DA_EVENT = os.environ.get('DA_EVENT', '0')
assert DA_EVENT in ('0', '1'), f'未知 DA_EVENT={DA_EVENT}（0|1）'
# ---- 快慢双层 DA_FAST（2026-09 新增：Step3）----
# 1=在线打分层叠加日频线性修正器（model_core/fast_layer.py）：用最近已实现标签滚动拟合
# score→标签的截面映射（含 10 维 DRM 暴露），强收缩先验（默认收缩到数据估计的 30%~50%）。
# 只影响在线阶段（update）。
DA_FAST = os.environ.get('DA_FAST', '0')
assert DA_FAST in ('0', '1'), f'未知 DA_FAST={DA_FAST}（0|1）'
# ---- 输入层清理 DA_MCLEAN（2026-09 新增：Step3）----
# 1=市场状态进门控前先做「状态依赖半衰期 EWMA + 120 维 PC 去噪」（model_core/mstate_clean.py）。
# 改变门控输入 → 需重跑离线元训练与在线更新（离线/在线两侧自动一致）。
DA_MCLEAN = os.environ.get('DA_MCLEAN', '0')
assert DA_MCLEAN in ('0', '1'), f'未知 DA_MCLEAN={DA_MCLEAN}（0|1）'
_ch = ([] if DA_DRM == 'report' else ['drmdk']) + ([] if DA_OPT == 'base' else ['ft'])
DTAG = ('_' + ''.join(_ch)) if _ch else ''                        # DRM 链产物后缀（跨 alpha 变体共享）
_LOSS_TAG = {'wmse': [], 'ic': ['lic'], 'rankwmse': ['lrw'],
             'prank': ['prk'], 'listnet': ['lst'], 'focal': ['foc'], 'hybrid': ['hyb']}[DA_LOSS]
_parts = _LOSS_TAG + ([] if DA_H == 'logit' else ['haff']) + ([] if DA_DRM == 'report' else ['drmdk']) \
         + ([] if DA_W == 'pred' else ['wlab']) + ([] if DA_OPT == 'base' else ['ft']) \
         + ([] if DA_EVENT == '0' else ['evt']) + ([] if DA_FAST == '0' else ['fast']) \
         + ([] if DA_MCLEAN == '0' else ['mcl']) \
         + ([] if _DM == 64 else [f'd{_DM}']) + ([] if _BT == 5.0 else [f'b{_BT:g}']) \
         + ([] if _GM == 5.0 else [f'g{_GM:g}']) \
         + ([] if _DP == 0.5 else [f'dp{int(round(_DP * 100))}']) \
         + ([] if _SD == 42 else [f's{_SD}'])
ATAG = ('_' + ''.join(_parts)) if _parts else ''                  # 架构/种子变体后缀
ARCH_STAMP = '' if not ATAG else ATAG[1:] + '_'                   # model_version 变体戳

VTAG = '' if DA_VAL_METRIC == 'wmse' else f'_{DA_VAL_METRIC}'    # 产物文件/目录后缀
VAL_STAMP = ('' if DA_VAL_METRIC == 'wmse' else f'{DA_VAL_METRIC}stop_') + ARCH_STAMP   # 版本戳：判据+架构

# DRM 链产物路径（随 DTAG 分叉；report=主线原名，deck=_drmdk 后缀）
PANEL_PATH = DERIVED / f'panel{DTAG}.npz'
EXPO_PATH = DERIVED / f'expo{DTAG}.dat'
MSTATE_PATH = DERIVED / f'market_state{DTAG}.npz'
EXPO_PQ_DIR = DERIVED / f'deep_risk_exposure_daily{DTAG}'
DRM_CKPT = 'drm' + DTAG                                           # ckpt 前缀：drm_2019.pt / drm_drmdk_2019.pt

SCORE_DIR = IN_DIR / f'meta_master_score{VTAG}{ATAG}'      # DA 元增量因子
MASTER_SCORE_DIR = IN_DIR / f'master_score{VTAG}{ATAG}'    # 静态 MASTER 因子

# 新增功能路径（2026-09）：
# 事件驱动报警序列（diag/step5 风格切换检测器定稿输出，随交付体提供；DA_EVENT=1 时读取）
SWITCH_ALARM_CSV = DERIVED / 'switch_alarm_daily.csv'
# 快慢双层 raw sidecar（DA_FAST=1 时落盘原始分，断点续跑重建修正器滚动统计用）
FAST_RAW_DIR = MODEL_DIR / f'fast_raw{VTAG}{ATAG}'
_et = ([] if DA_VAL_METRIC == 'wmse' else [DA_VAL_METRIC]) + ([] if not ATAG else [ATAG[1:]])
EVAL_TAG = ('-' + '_'.join(_et)) if _et else ''   # 评价目录后缀：eval{TAG}_{pool}

# 原始表目录：随交付体自带（data/raw_tables/，7 张表约 1.3G），从零重建数据时由
# prepare 阶段读取；日常训练/评价只用 data/derived/ 产物，不触碰此目录。
# 转内网数据源时只需改 input_data_update/ 各 build_* 的读取层。
RAW_TABLES = DATA_DIR / 'raw_tables'

CFG = dict(
    seed=_SD,                            # 随机种子（由 DA_SEED 注入）
    window=40,                       # 个股输入时序窗（40 日 × 70 维）
    mad_k=5.0,                       # 量价比值截面去极值阈：中位数 ± 5×MAD
    label_h=11,                      # 标签：t+1 建仓 → t+11 平仓（持有 10 日）
    eps=1e-12,

    # ---- 数据层 / DRM 构建 ----
    data_start=2009,                 # 行情/Barra 起点（2013 首个 DRM 训练窗需 2009-2012 数据）
    horizon=20,                      # DRM 标签跨度（未来 20 日）+ VALID 窗口完整性检查
    barra_ffill_limit=5,             # Barra 暴露时间轴前向填充上限（新鲜度 ≤5 日）
    excl_st=True,                    # VALID 排除 ST（PIT [进入,撤销) 左闭右开）
    min_listed_days=60,              # 剔除新股：上市满 60 个交易日才进 alpha 池/标签排序池
                                     # （只作用于 MASTER_POOL 与标签，不改 VALID，DRM 不受影响）

    # ---- DRM 深度风险模型（隐式 GAT + 双支 GRU，K=10）----
    n_price=5, n_barra=10,           # DRM 输入 = 5 量价比值 + 10 Barra CNE5 = 15 维（n,40,15）
    drm_k=10,                        # 潜在风险因子数（双支各出 K/2=5）
    drm_hidden=64, drm_gru_layers=2, # 双支 GRU 隐藏维与层数
    drm_window=40 if DA_DRM == 'report' else 20,          # DRM 输入回看窗（report=40 / deck=20）
    drm_gat_dropout=0.3 if DA_DRM == 'report' else 0.5,   # GAT 注意力 dropout（report=0.3 / deck=0.5）
    drm_lam_vif=0.01,                # 损失 VIF 迹正则权重 λ（压因子间共线性）
    drm_lr=8e-4, drm_max_epoch=100,
    drm_patience=20 if DA_OPT == 'base' else 5,   # 早停耐心（以验证集 R² 为准；ft 变体取 5）
    plateau_stall=3, plateau_factor=0.5,   # ft 变体的 plateau 退火：验证停滞 3 轮 → lr 减半
    drm_train_years=4, drm_val_frac=0.10,  # 4 年滚动训练窗，尾部 10% 留作验证
    drm_warm_start=True,             # DRM 年度滚动的暖启动：每年从上一年权重继续训练，
                                     # 保隐因子身份跨年连续。与 da_warm_start 相互独立
    drm_first_expo_year=2013,        # 暴露产出起始年（覆盖 alpha 2019+ OOS）

    # ---- MASTER 骨干 ----
    m_dmodel=_DM, m_tnhead=4, m_snhead=2, m_dropout=_DP,   # 隐藏维 / 时序头数 / 截面头数 / dropout
    m_gamma=_GM,                     # 加权 MSE 的 γ：w=Sigmoid(γ(ŷ−0.5))，放大多头端权重
    m_lr=8e-5, m_patience=10, m_max_epoch=100,  # MASTER 静态预训练（静态因子线与暖启动来源）

    # ---- 市场状态门控（183 维 = 63 指数态 + 120 DRM 衍生态）----
    gate_mode='dual',                # dual=双门控（定稿）| single | none（后两者消融用）
    gate_beta=_BT,                   # softmax 温度 β（越小特征取舍越尖锐）
    gate_clip=5.0,                   # 门控输入 robust-z 截断阈（抑制 amp 族重尾）

    # ---- DoubleAdapt 元学习 ----
    da_tau=10.0,                     # 适配器门控 softmax 温度（大 τ 近似各头均匀混合、保截面秩）
    da_heads=8, da_label_hid=32,     # 适配器头数 / 标签门控隐藏维
    da_psi_lr=0.01,                  # 数据适配器 ψ 学习率
    da_reg=0.05, da_sigma=0.167,     # REINFORCE 标签正则：coef=(L_φ−L_q)/da_sigma + da_reg；
                                     # da_sigma=2σ² 按 [0,1] 排序标签的尺度实例化（σ=1/√12）
    da_inner_lr=1e-3,                # 内层适应步长（1 步继承状态 Adam）
    da_outer_lr=1e-3,                # 外层 φ 离线元训练学习率
    da_online_lr=1e-3,               # 外层 φ 在线阶段学习率
    da_grad_clip=3.0,                # 梯度裁剪范数上限
    da_incr=20,                      # 在线元更新节拍（每 20 个交易日一次）
    da_support=20, da_query=20,      # 元任务 support/query 段长度（交易日）
    da_max_epoch=100, da_patience=10 if DA_OPT == 'base' else 5,  # 离线元训练轮数上限与早停耐心

    # ---- ABCD 头部损失实验超参（仅 DA_LOSS=prank/listnet/focal/hybrid 时生效；调参改这里）----
    rk_head_q=0.8,                   # A/D pairwise：正样本标签分位阈（y≥0.8 = 头部 20%）
    rk_topq=0.9,                     # A/D pairwise：G1 边界（y≥0.9 的正样本额外加权，对应 G1/G2 区分）
    rk_k=16,                         # A/D pairwise：每个正样本采样的负样本数
    rk_tau=0.1,                      # A/D pairwise：softplus 温度（z=logit(pred) 空间，越小推得越狠）
    rk_wtop=2.0,                     # A/D pairwise：y≥rk_topq 正样本的额外权重（w=1+rk_wtop）
    ln_tau_q=0.1,                    # B listwise：目标分布 softmax(y/τq) 温度（越小质量越压头部）
    ln_tau_z=0.1,                    # B listwise：模型分布 softmax(z/τz) 温度
    focal_topq=0.9,                  # C focal：顶部分类边界（y≥0.9 为正类）
    focal_gamma=2.0,                 # C focal：聚焦参数 γ（难样本加权）
    focal_alpha=0.25,                # C focal：正类权重 α
    focal_lambda=0.1,                # C：focal 辅助项相对 wMSE 的权重（wMSE≈0.04 量级对齐）
    hb_lambda=0.1,                   # D：pairwise 项相对 wMSE 的权重

    # ---- 事件驱动更新 / 快慢双层 / 输入层清理（DA_EVENT / DA_FAST / DA_MCLEAN，默认关闭）----
    event=DA_EVENT == '1',           # 事件驱动更新节点（报警段起点即更新，不等 20 日排期）
    event_trust=float(os.environ.get('DA_EVENT_TRUST', 2.0)),   # 事件节点内层 lr 放大倍数（信任域）
    event_gap=int(os.environ.get('DA_EVENT_GAP', 3)),           # 节点最小间隔（可训日；过近则合并）
    fast=DA_FAST == '1',             # 快慢双层：日频线性修正器
    fast_window=int(os.environ.get('DA_FAST_W', 40)),           # 滚动标签日数（统计窗口）
    fast_min_days=int(os.environ.get('DA_FAST_MIND', 20)),      # 最少标签日数（不足不修正）
    fast_min_rows=int(os.environ.get('DA_FAST_MINR', 20000)),   # 最少样本行数（不足不修正）
    fast_sh_a=float(os.environ.get('DA_FAST_SHA', 0.3)),        # 截距收缩系数（向 0.5）
    fast_sh_s=float(os.environ.get('DA_FAST_SHS', 0.5)),        # score 斜率收缩系数（向 1）
    fast_sh_z=float(os.environ.get('DA_FAST_SHZ', 0.3)),        # 风格系数收缩系数（向 0）
    fast_cap=float(os.environ.get('DA_FAST_CAP', 0.10)),        # 单股修正上限（rank 单位）
    mclean=DA_MCLEAN == '1',         # 输入层清理：状态依赖 EWMA + 120 维 PC 去噪
    mclean_hl=float(os.environ.get('DA_MCLEAN_HL', 5.0)),       # 平时 EWMA 半衰期（交易日）
    mclean_hl_alarm=float(os.environ.get('DA_MCLEAN_HLA', 1.5)),  # 报警期 EWMA 半衰期（交易日）
    mclean_pc_k=int(os.environ.get('DA_MCLEAN_K', 20)),         # 120 维保留主成分数

    # ---- 滚动区间 ----
    da_off_end=2017,                 # 离线元训练任务截至年
    da_val_year=2018,                # 轨迹验证年（离线早停选点）
    da_online_y0=2019, da_online_y1=2026,  # 在线 OOS 区间（面板末日 2026-07-20，2026 为半年样本）
    da_pretrain_year=2019,           # 暖启动用的预训练 MASTER 年份（仅 da_warm_start=True 时用）
    da_warm_start=False,             # DoubleAdapt 离线元训练的启动方式：False=冷启动（φ 随机初始化，
                                     # 定稿）/ True=暖启动（载入 da_pretrain_year 的预训练 MASTER 权重）。
                                     # 与 drm_warm_start 相互独立

    # ---- 回测/评价口径（evaluate/factor_eval.py 消费，详见 指标计算口径.md）----
    bt_groups=10,                    # 十分组（多头 = G1 最高分组等权满仓）
    val_top_frac=0.10,               # topq 判据的多头端宽度（与回测十分组一致）
    bt_fee_side=0.0015,              # 单边 1.5‰（双边千三）
    bt_min_periods=20,               # 双周频不足 20 期（约 0.77 年）不年化，标 YTD
    bt_ppy=26,                       # 双周频年化基数（多空/换手，每年约 26 期）
    bt_ann_days=243,                 # 日频年化基数（A 股年均交易日；调仓双周、收益风险按日频核算）
    ic_ann_weeks=52,                 # 周频 IC 年化基数 √52（与回测 ppy=26 刻意不同，勿混用）
)


# 三池评价：全 A 训练出的单一因子，分别在三个指数 PIT 成分内独立排序/回测
POOLS = {'csi300': '000300.SH', 'csi500': '000905.SH', 'csi1000': '000852.SH'}

# 全市场评价：股票池 = 全 A 可投（panel.MASTER_POOL，无成分过滤），基准 = 中证800。
# RankIC/多空不依赖基准口径，可与研报公开值直接对标。
ALLMKT = ('allmkt', '000906.SH')


def device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')

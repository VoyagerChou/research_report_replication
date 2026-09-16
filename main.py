# -*- coding: utf-8 -*-
"""编排入口：按子命令依次调起数据准备、训练、在线更新与评价各模块。

Meta Master alpha 模型 = MASTER 骨干 + 183 维市场状态门控 + 加权 MSE +
[0,1] 截面排序标签 + DoubleAdapt 元增量。三步流程：
  1. 数据准备：input_data_update/prepare_local_data.py 从本地原始表构造共享面板
  2. 离线元训练：train/double_adapt_train.py（φ 冷启动 + 学适配器，2018 轨迹验证早停）
  3. 在线元增量：update/update.py（每 20 日 meta-update→适应→打分落库）

用法：
  python main.py prepare   # 仅准备数据（面板 + 回测辅助矩阵）
  python main.py train     # 仅离线元训练
  python main.py update    # 仅在线元增量
  python main.py all       # 全流程（prepare→train→update，不含评价）
  python main.py eval      # 评价 DoubleAdapt 因子（周频RankIC + 双周多头回测）
  python main.py master    # 静态 MASTER 因子线：逐年训练 + 逐日打分落库（可另一张卡并行）
  python main.py eval master   # 评价静态 MASTER 因子
  python main.py align     # 研报口径对齐回测（分年对比表；口径见 指标计算口径.md）
  python main.py compare       # 横向对照表（全部已跑评价，纯读盘）
  python main.py drmcheck      # DRM 验收诊断

标签口径：未来 10 日 vwap 收益的全 A 截面排序映射 [0,1]；全 A 训练 + 四池映射评价。
实验变体经 DA_* 环境变量注入（默认即定稿主线），见 config.py 开头说明。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if stage in ('prepare', 'all'):
        from input_data_update import prepare_local_data
        prepare_local_data.main()

    if stage in ('train', 'all'):
        from train import double_adapt_train
        double_adapt_train.main()

    if stage in ('update', 'all'):
        from update import update
        update.main()

    if stage in ('master',):                      # 静态 MASTER 因子线（可与 DA 并行跑在另一张卡）
        from update import master_score
        yrs = [int(a) for a in sys.argv[2:] if a.isdigit()]
        master_score.main(yrs or None)

    if stage in ('eval',):                        # 单列：评价不并入 all（回测重、按需跑）
        from evaluate import factor_eval          # `python main.py eval master` 评静态MASTER线
        line = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ('da', 'master') else 'da'
        factor_eval.main(line)

    if stage in ('align',):                       # 研报口径对齐回测（重建版，自包含）
        from evaluate import report_align
        report_align.main()

    if stage in ('compare',):                     # 横向对照表(因子线×训练池×评价池,纯读盘)
        from evaluate import compare
        compare.main()

    if stage in ('drmcheck',):                    # DRM 验收诊断（同日R² deep vs barra + 自相关）
        from evaluate import diag_drm
        diag_drm.main()



if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""离线元训练编排：`python main.py train`。

薄壳：载入共享面板(panel.npz) → DoubleAdaptTrainer.train_offline（模型核心在 model_core/meta_train.py）。
产物：data/models_weights/da_offline{VTAG}{ATAG}.pt
（φ=MASTER骨干+门控 / ψ=DataAdapter / opt / gn / meta，断点续跑已内建在 train_offline）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.prepare_data import PanelData
from model_core.meta_train import DoubleAdaptTrainer


def main(quiet=False):
    print('== 离线元训练 ==', flush=True)
    panel = PanelData()                      # 自检：TECH/EXPO memmap 通道正确性
    trainer = DoubleAdaptTrainer(panel)
    ckpt = trainer.train_offline(quiet=quiet)
    print(f'离线元训练完成：{ckpt}', flush=True)
    return ckpt


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""ABCD 四种损失的连续实验工作流（云端一键：train → update → eval × 4 变体）。

对每个损失变体依次执行（子进程注入 DA_LOSS，产物按 config.py 的后缀机制自动隔离，
与原版 meta_master_score 完全分开、原版因子不受影响）：

  1. train  离线元训练   → data/models_weights/da_offline{tag}.pt
             （已存在则跳过；训练中断有 da_offline_resume{tag}.pt 内建续跑）
  2. update 在线元增量   → data/factor_values/meta_master_score{tag}/（2019–2026 逐年分片
             + da_online_state{tag}_{year}.pt 年界状态；中断后重跑自动从最近年末状态续跑）
  3. eval   评估：
     a) 若 evaluate/factor_eval.py 存在 → `python main.py eval`（项目自带评估，四池口径）
        —— 注意：该文件在本机交付体中曾被清空，不存在时自动跳过并告警；
     b) 始终 → `python evaluate_head.py`（自包含头部指标：RankIC/十分位/G1−G2/
        top10 overlap/NDCG@10%/TopQ，逐日+分年+汇总）
  4. 收尾   `python evaluate_head.py --compare` 输出横向对照表
             （含 ref_original = 原版 wmse 因子参照行）

用法：
  python run_loss_abcd.py                          # 全部 4 个变体，train+update+eval 连跑
  python run_loss_abcd.py --losses prank hybrid    # 只跑指定变体
  python run_loss_abcd.py --stages update eval     # 只跑后两段（离线已训完时）
  python run_loss_abcd.py --stages eval            # 只重跑评估
  python run_loss_abcd.py --dry-run                # 只打印计划与产物路径，不执行
  python run_loss_abcd.py --continue-on-error      # 单变体失败不阻断后续
  python run_loss_abcd.py --no-ref                 # 不评估原版参照行

预计耗时（单变体，视云 GPU）：离线元训练 ~2–6h，在线增量 ~0.5–1.5h，评估 ~2min；
4 变体串行约 10–30h。全程断点可续（重复执行本脚本即可）。
日志：logs/loss_abcd/<运行时间戳>/<loss>_<stage>.log
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOSSES = ['prank', 'listnet', 'focal', 'hybrid']       # A / B / C / D
RUN_TS = time.strftime('%Y%m%d_%H%M%S')
LOG_DIR = ROOT / 'logs' / 'loss_abcd' / RUN_TS
REF_NAME = 'ref_original'


def resolve(loss):
    """用子进程按 config.py 的实际逻辑解析该变体的产物路径（ATAG/SCORE_DIR/ckpt）。"""
    env = dict(os.environ)
    env['DA_LOSS'] = loss
    code = ("import config;"
            "print(config.ATAG);print(config.SCORE_DIR);"
            "print(config.MODEL_DIR / f'da_offline{config.VTAG}{config.ATAG}.pt')")
    out = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True, check=True)
    atag, score_dir, ckpt = out.stdout.splitlines()[:3]
    return atag.strip(), Path(score_dir.strip()), Path(ckpt.strip())


def run_cmd(args, env, log_path, echo=True):
    """跑子进程：输出同时进控制台与日志文件。返回退出码。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(f'\n===== {time.strftime("%H:%M:%S")} | cmd: {" ".join(str(a) for a in args)} =====\n')
        p = subprocess.Popen([str(a) for a in args], cwd=str(ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, encoding='utf-8', errors='replace')
        for line in p.stdout:
            if echo:
                print(line, end='')
            lf.write(line)
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--losses', nargs='+', default=LOSSES, choices=LOSSES)
    ap.add_argument('--stages', nargs='+', default=['train', 'update', 'eval'],
                    choices=['train', 'update', 'eval'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--no-ref', action='store_true')
    args = ap.parse_args()

    print(f'== ABCD 损失实验工作流 ==  变体: {args.losses}  阶段: {args.stages}')
    print(f'日志目录: {LOG_DIR}')

    # ---- 解析各变体路径（与 config 实际计算一致）----
    plan = {}
    for loss in args.losses:
        atag, score_dir, ckpt = resolve(loss)
        plan[loss] = dict(atag=atag, score_dir=score_dir, ckpt=ckpt)
        print(f'  [{loss}] ATAG={atag or "(空=主线)"} ckpt={ckpt.name} score_dir={score_dir}')

    ref_score_dir = ROOT / 'data' / 'factor_values' / 'meta_master_score'
    has_ref = ref_score_dir.exists() and not args.no_ref
    has_main_eval = (ROOT / 'evaluate' / 'factor_eval.py').exists()
    if not has_main_eval:
        print('  [提示] evaluate/factor_eval.py 不存在 → 跳过 `main.py eval`（项目自带评估），'
              '仅跑自包含头部指标 evaluate_head.py')

    if args.dry_run:
        print('\n--dry-run：以下是执行计划（未执行）——')
        for loss in args.losses:
            for stage in args.stages:
                print(f'  [{loss}] {stage}')
        print(f'  参照行: {"有" if has_ref else "无"} | main.py eval: {"有" if has_main_eval else "跳过"}')
        return

    failures = []
    t_all = time.time()
    for loss in args.losses:
        p = plan[loss]
        env = dict(os.environ)
        env['DA_LOSS'] = loss
        env['PYTHONIOENCODING'] = 'utf-8'
        print(f'\n########## 变体 [{loss}] ##########', flush=True)

        for stage in args.stages:
            t0 = time.time()
            log = LOG_DIR / f'{loss}_{stage}.log'
            if stage == 'train':
                if p['ckpt'].exists():
                    print(f'[{loss}/train] {p["ckpt"].name} 已存在，跳过')
                    continue
                print(f'[{loss}/train] 离线元训练 → {p["ckpt"].name}')
                rc = run_cmd([sys.executable, 'main.py', 'train'], env, log)
            elif stage == 'update':
                print(f'[{loss}/update] 在线元增量 → {p["score_dir"]}')
                rc = run_cmd([sys.executable, 'main.py', 'update'], env, log)
            else:  # eval
                rc = 0
                if has_main_eval:
                    print(f'[{loss}/eval] 项目自带评估 main.py eval')
                    rc = run_cmd([sys.executable, 'main.py', 'eval'], env, log)
                print(f'[{loss}/eval] 头部指标 evaluate_head.py')
                rc2 = run_cmd([sys.executable, 'evaluate_head.py',
                               '--score-dir', str(p['score_dir']), '--name', loss], env, log)
                rc = rc or rc2
            dt = (time.time() - t0) / 60
            print(f'[{loss}/{stage}] 退出码 {rc}，耗时 {dt:.1f} min', flush=True)
            if rc != 0:
                failures.append((loss, stage, rc))
                if not args.continue_on_error:
                    print(f'!! [{loss}/{stage}] 失败（rc={rc}），中止。'
                          f'修复后重跑本脚本即可（断点续跑）。日志: {log}')
                    sys.exit(rc)

        # 每个变体完成后立刻把参照行也备好（只算一次）
        if 'eval' in args.stages and has_ref and not (LOG_DIR / '.ref_done').exists():
            run_cmd([sys.executable, 'evaluate_head.py', '--score-dir',
                     str(ref_score_dir), '--name', REF_NAME], env, LOG_DIR / f'{REF_NAME}.log')
            (LOG_DIR / '.ref_done').touch()

    # ---- 收尾：横向对照 ----
    if 'eval' in args.stages:
        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        run_cmd([sys.executable, 'evaluate_head.py', '--compare'], env, LOG_DIR / 'compare.log')

    print(f'\n== 工作流结束 ==  总耗时 {(time.time() - t_all) / 60:.1f} min')
    if failures:
        print(f'失败项: {failures}（修复后重跑本脚本即可，断点续跑）')
        sys.exit(1)
    print('全部变体完成。对照表: results/loss_abcd/summary.csv')


if __name__ == '__main__':
    main()

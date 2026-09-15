#!/usr/bin/env bash
# =============================================================================
# Meta Master · Step2+Step3 一键实验脚本（顺序执行：跑完一个变体自动跑下一个）
#
# 变体与步骤（每个变体：train → update → eval）：
#   1) evtfastmcl  DA_EVENT=1 DA_FAST=1 DA_MCLEAN=1   组合：事件驱动+快慢双层+输入清理
#   2) evt         DA_EVENT=1                          单因子：事件驱动更新+信任域
#   3) fast        DA_FAST=1                           单因子：快慢双层
#   4) mcl         DA_MCLEAN=1                         单因子：输入层清理
#
# 用法：
#   bash run_step23.sh            # 顺序跑全部 4 个变体（无人值守）
#   bash run_step23.sh evt fast   # 只跑指定变体（按脚本内固定顺序）
#   nohup bash run_step23.sh > logs/step23/runner.out 2>&1 &    # 后台挂机跑
#
# 断点续跑：重跑本脚本会自动跳过已完成步骤（train 见 ckpt 跳过 / update 按年续跑）。
# 产物隔离：因子   data/factor_values/meta_master_score_<tag>/
#           状态   data/models_weights/da_online_state_<tag>_*.pt（DA_FAST 另有 fast_raw_<tag>/）
#           评价   data/eval/eval_da/summary_<tag>.csv（factor_eval 固定写 summary.csv，脚本代为另存）
# 失败处理：某变体某步骤失败 → 跳过该变体剩余步骤，继续下一个变体；结束时汇总失败项。
# =============================================================================
set -eo pipefail

cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

LOG=logs/step23
mkdir -p "$LOG"

echo "==================== Step2+Step3 一键实验 ===================="
echo "开始时间: $(date '+%F %T')"
python -V

# ---- 前置检查：新增/修改文件是否就位 ----
missing=0
for f in model_core/event_schedule.py model_core/mstate_clean.py model_core/fast_layer.py \
         data/derived/switch_alarm_daily.csv; do
  if [ ! -f "$f" ]; then echo "[错误] 缺少新增文件: $f"; missing=1; fi
done
if ! grep -q "DA_EVENT" config.py; then echo "[错误] config.py 不是新版（缺 DA_EVENT）"; missing=1; fi
if ! grep -q "event_schedule" update/update.py; then echo "[错误] update/update.py 不是新版（缺 event_schedule）"; missing=1; fi
if [ "$missing" -ne 0 ]; then echo "请先上传新增/修改文件后再运行。"; exit 1; fi

# ---- 变体定义（顺序即执行顺序）----
VARIANTS=(
  "evtfastmcl|DA_EVENT=1 DA_FAST=1 DA_MCLEAN=1|组合：事件驱动+快慢双层+输入清理"
  "evt|DA_EVENT=1|单因子：事件驱动更新+信任域"
  "fast|DA_FAST=1|单因子：快慢双层"
  "mcl|DA_MCLEAN=1|单因子：输入层清理"
)

WANT=("$@")
if [ "${#WANT[@]}" -gt 0 ]; then
  for w in "${WANT[@]}"; do
    ok=0
    for v in "${VARIANTS[@]}"; do
      IFS='|' read -r t _ _ <<< "$v"
      if [ "$t" = "$w" ]; then ok=1; fi
    done
    if [ "$ok" -ne 1 ]; then echo "[错误] 未知变体: $w（可选: evtfastmcl evt fast mcl）"; exit 1; fi
  done
fi

should_run() {
  local t="$1"
  if [ "${#WANT[@]}" -eq 0 ]; then return 0; fi
  for w in "${WANT[@]}"; do
    if [ "$w" = "$t" ]; then return 0; fi
  done
  return 1
}

run_stage() {  # run_stage <tag> <stage> <env...>
  local tag="$1"; local stage="$2"; shift 2
  echo ""
  echo "=============================================================="
  echo "  [$tag] $stage   环境: $*"
  echo "=============================================================="
  env "$@" python -u main.py "$stage" 2>&1 | tee "$LOG/${tag}_${stage}.log"
}

T0=$(date +%s)
FAILED=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag envs desc <<< "$v"
  if ! should_run "$tag"; then continue; fi
  echo ""
  echo "################################################################"
  echo "#  变体 [$tag]  $desc"
  echo "################################################################"
  vfail=0
  for stage in train update eval; do
    # shellcheck disable=SC2086
    if ! run_stage "$tag" "$stage" $envs; then
      echo "[失败] [$tag] $stage —— 跳过该变体剩余步骤，继续下一个变体"
      FAILED+=("$tag/$stage")
      vfail=1
      break
    fi
  done
  if [ "$vfail" -eq 0 ] && [ -f data/eval/eval_da/summary.csv ]; then
    cp data/eval/eval_da/summary.csv "data/eval/eval_da/summary_${tag}.csv"
    echo "  [eval] 评价汇总已另存 -> data/eval/eval_da/summary_${tag}.csv"
  fi
done

T1=$(date +%s)
echo ""
echo "==================== 运行结束（用时 $(( (T1 - T0) / 60 )) 分钟）===================="
echo "阶段日志: $LOG/<变体>_<步骤>.log"

# ---- 变体评价对比（失败不影响已完成产物）----
python - <<'PYEOF' || echo "[提示] 对比汇总失败（不影响已完成产物）"
import glob
import pandas as pd
rows = []
for f in sorted(glob.glob('data/eval/eval_da/summary_*.csv')):
    d = pd.read_csv(f)
    d.insert(0, 'variant', f.split('summary_')[-1].replace('.csv', ''))
    rows.append(d)
if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv('logs/step23/summary_compare.csv', index=False)
    print('')
    print('=== 变体评价对比（已存 logs/step23/summary_compare.csv）===')
    print(out.to_string(index=False))
else:
    print('未找到 data/eval/eval_da/summary_*.csv')
PYEOF

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo ""
  echo "[警告] 以下阶段失败，请查看对应日志: ${FAILED[*]}"
  exit 1
fi
echo "全部变体完成。"

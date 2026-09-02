#!/bin/bash
#
# Paper-final synthesis launcher.
#
# Runs synth/synth.py over the 1,200-entry manifest in two cells, then
# returns once both finish:
#
#   base   no mask refinement — the k=0 ablation cell of Table 3
#   m4_xu  margin=4 dilation + dual-tag union — the paper-final cell:
#          baseline_src_only (λ=0) / Swap (λ=3) / Ours (λ=7) / λ-sweep points
#
# Sharding: NUM_SHARDS parallel python processes per cell, one per GPU.
#
# Env:
#   PYTHON      python interpreter (default: 'python')
#   NUM_SHARDS  parallel shards per cell (default: 8 = one per B200 GPU)
#   GPU_BASE    first physical GPU id (default: 0)
#   LAMBDAS     comma-separated paper-λ subset (default: 3,5,7,9,11,13)
#   QWEN_PY     interpreter with Qwen3-ForcedAligner installed. Not used by
#               synthesis (in-synthesis FA evaluation is off); it is the eval
#               stage -- `python -m eval mer_la_lid` -- that needs it.
#
# Usage:
#   bash scripts/run_synth.sh
#   NUM_SHARDS=4 GPU_BASE=4 bash scripts/run_synth.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_BASE="${GPU_BASE:-0}"
LAMBDAS="${LAMBDAS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="synth/synth.py"
SETUP="layers_set_8-12__max__a2t"

# Cell descriptors: tag -> phrase-mask flags
declare -A CELL_FLAGS
CELL_FLAGS[base]=""
CELL_FLAGS[m4_xu]="--phrase-mask-margin 4 --phrase-mask-xlang-union true"

run_cell() {
  local cell="$1" flags="$2" outsuffix="l8-12_${cell}"
  local log_dir="synth_${outsuffix}_logs"
  mkdir -p "$log_dir"
  echo
  echo "================================================================"
  echo "[synth/$cell] out=synth_${outsuffix} flags='${flags}' lambdas='${LAMBDAS:-<default>}'"
  echo "================================================================"

  local pids=()
  for ((s=0; s<NUM_SHARDS; s++)); do
    local gpu=$((GPU_BASE + s))
    local log="$log_dir/shard${s}.log"
    local lambda_arg=()
    [[ -n "$LAMBDAS" ]] && lambda_arg=(--lambdas "$LAMBDAS")
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
      --num-shards "$NUM_SHARDS" --gpu-shard "$s" \
      --shard-name "GPU${gpu}" \
      --attention-setup "$SETUP" \
      --out-suffix "$outsuffix" \
      "${lambda_arg[@]}" \
      $flags \
      > "$log" 2>&1 &
    pids+=($!)
    echo "[synth/$cell] launched shard=$s gpu=$gpu pid=$! log=$log"
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[synth/$cell] pid=$pid FAILED"
      fail=1
    fi
  done
  if [[ $fail -ne 0 ]]; then
    echo "[synth/$cell] one or more shards failed. See $log_dir/shard*.log" >&2
  fi
  echo "[synth/$cell] DONE"
  return $fail
}

run_cell base    "${CELL_FLAGS[base]}"
run_cell m4_xu   "${CELL_FLAGS[m4_xu]}"

echo
echo "[synth] ALL CELLS DONE"
echo "    k=0 ablation            -> synth_l8-12_base/<eid>/"
echo "    no_CS / Swap / Ours     -> synth_l8-12_m4_xu/<eid>/"

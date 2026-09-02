#!/bin/bash
#
# Multi-GPU launcher for the MER + LA + per-phrase LID eval module
# (eval/_mer_la_lid.py, dispatched via `python -m eval mer_la_lid`).
#
# Required env:
#   SYNTH_DIR     synth directory (e.g. 'synth_l8-12_m4_xu')
# Optional env:
#   NUM_SHARDS    default 8
#   GPU_BASE      default 0
#   EVAL_DIR      override the eval output dir (default: 'eval_<synth-suffix>')
#   PYTHON        python interpreter (default: 'python')
#   QWEN_PY       interpreter with Qwen3-ForcedAligner installed (FA span pass)
#   DRY_RUN=1     print plan and exit

set -euo pipefail

PYTHON="${PYTHON:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_BASE="${GPU_BASE:-0}"
SYNTH_DIR="${SYNTH_DIR:?SYNTH_DIR required (e.g. SYNTH_DIR=synth_l8-12_m4_xu)}"
EVAL_DIR="${EVAL_DIR:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$SYNTH_DIR" in
  synth)        log_suffix="eval";;
  synth_*)      log_suffix="eval_${SYNTH_DIR#synth_}";;
  *)            log_suffix="eval_${SYNTH_DIR}";;
esac
LOG_DIR="${EVAL_DIR:-$log_suffix}_logs"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval] DRY: would launch $NUM_SHARDS shards on GPU $GPU_BASE..$((GPU_BASE+NUM_SHARDS-1))"
  echo "[eval] DRY: synth_dir=$SYNTH_DIR  eval_dir=${EVAL_DIR:-(auto)}  logs=$LOG_DIR"
  exit 0
fi
mkdir -p "$LOG_DIR"

EVAL_DIR_ARG=()
[[ -n "$EVAL_DIR" ]] && EVAL_DIR_ARG=(--eval-dir "$EVAL_DIR")

QWEN_ARG=()
[[ -n "${QWEN_PY:-}" ]] && QWEN_ARG=(--fa-aligner-python "$QWEN_PY")

# --- CPU thread budgeting ----------------------------------------------------
# Without these caps each python process spawns one OpenMP thread per logical
# CPU and they contend heavily; capping per-shard to (cores/shards - 2) keeps
# Whisper.generate() fast.
TOTAL_CORES="${TOTAL_CORES:-$(nproc)}"
PER_SHARD_THREADS="${PER_SHARD_THREADS:-$(( TOTAL_CORES / NUM_SHARDS - 2 ))}"
[[ $PER_SHARD_THREADS -lt 1 ]] && PER_SHARD_THREADS=1
export OMP_NUM_THREADS="$PER_SHARD_THREADS"
export MKL_NUM_THREADS="$PER_SHARD_THREADS"
export OPENBLAS_NUM_THREADS="$PER_SHARD_THREADS"
export NUMEXPR_NUM_THREADS="$PER_SHARD_THREADS"
export TOKENIZERS_PARALLELISM=false

echo "[eval] synth=$SYNTH_DIR num_shards=$NUM_SHARDS gpu_base=$GPU_BASE logs=$LOG_DIR per_shard_threads=$PER_SHARD_THREADS"

pids=()
for ((s=0; s<NUM_SHARDS; s++)); do
  gpu=$((GPU_BASE + s))
  log="$LOG_DIR/shard${s}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m eval mer_la_lid \
    --num-shards "$NUM_SHARDS" \
    --gpu-shard "$s" \
    --device-index 0 \
    --shard-name "GPU${gpu}" \
    --synth-dir "$SYNTH_DIR" \
    "${EVAL_DIR_ARG[@]}" \
    "${QWEN_ARG[@]}" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "[eval] launched shard=$s gpu=$gpu pid=$! log=$log"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[eval] pid=$pid FAILED"
    fail=1
  fi
done
[[ $fail -ne 0 ]] && { echo "[eval] One or more shards failed. See $LOG_DIR/shard*.log" >&2; exit 1; }

echo "[eval] All shards done. Aggregating ..."
"$PYTHON" -m eval mer_la_lid \
  --only-aggregate \
  --synth-dir "$SYNTH_DIR" \
  "${EVAL_DIR_ARG[@]}" \
  > "$LOG_DIR/aggregate.log" 2>&1
echo "[eval] DONE."

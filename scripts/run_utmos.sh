#!/bin/bash
#
# Multi-GPU UTMOS launcher (eval/_utmos.py, dispatched via `python -m eval utmos`).
# UTMOS is reference-free and depends only on the synth wavs.
#
# Required env:
#   SYNTH_DIR     synth directory (e.g. 'synth_l8-12_m4_xu')
# Optional env:
#   UTMOS_DIR     output dir name (default: 'utmos_<synth-suffix>')
#   NUM_SHARDS    default 8
#   GPU_BASE      default 0
#   PYTHON        python interpreter (default: 'python')

set -euo pipefail

PYTHON="${PYTHON:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_BASE="${GPU_BASE:-0}"
SYNTH_DIR="${SYNTH_DIR:?SYNTH_DIR required (e.g. SYNTH_DIR=synth_l8-12_m4_xu)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$SYNTH_DIR" in
  synth)    suffix="";;
  synth_*)  suffix="${SYNTH_DIR#synth_}";;
  *)        suffix="$SYNTH_DIR";;
esac
UTMOS_DIR="${UTMOS_DIR:-utmos${suffix:+_$suffix}}"
LOG_DIR="${UTMOS_DIR}_logs"
mkdir -p "$LOG_DIR" "$UTMOS_DIR"

echo "[utmos] synth=$SYNTH_DIR utmos_dir=$UTMOS_DIR num_shards=$NUM_SHARDS gpu_base=$GPU_BASE"

pids=()
for ((s=0; s<NUM_SHARDS; s++)); do
  gpu=$((GPU_BASE + s))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m eval utmos \
    --synth-dir "$SYNTH_DIR" \
    --utmos-dir "$UTMOS_DIR" \
    --num-shards "$NUM_SHARDS" \
    --gpu-shard "$s" \
    --device-index 0 \
    --shard-name "GPU${gpu}" \
    > "$LOG_DIR/shard${s}.log" 2>&1 &
  pids+=($!)
  echo "[utmos] launched shard=$s gpu=$gpu pid=$! log=$LOG_DIR/shard${s}.log"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then echo "[utmos] pid=$pid FAILED"; fail=1; fi
done
[[ $fail -ne 0 ]] && { echo "[utmos] One or more shards failed. See $LOG_DIR/shard*.log" >&2; exit 1; }

echo "[utmos] Merging per-shard JSONLs ..."
merged="${UTMOS_DIR}/utmos.jsonl"
: > "$merged"
for ((s=0; s<NUM_SHARDS; s++)); do
  shard_jsonl="${UTMOS_DIR}/utmos.shard${s}.jsonl"
  [[ -f "$shard_jsonl" ]] && cat "$shard_jsonl" >> "$merged"
done
echo "[utmos] DONE  merged $(wc -l < "$merged") rows -> $merged"

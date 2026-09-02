#!/bin/bash
#
# Attention-backend overhead benchmark launcher (paper appendix: Runtime Cost of Attention Extraction).
#
# Runs analysis/bench_attention_backend.py on ONE GPU, three sequential
# conditions:
#
#   A  baseline (no-CS)  attn_implementation=sdpa
#   B  baseline (no-CS)  attn_implementation=eager
#   C  LCG "Ours" (λ=7, margin=4, xlang-union, layers_set_8-12__max__a2t)
#      attn_implementation=eager (the only backend that returns attentions)
#
# Outputs: _artifacts/bench/bench_backend_out/{A_baseline_sdpa,B_baseline_eager,C_lcg_eager}/*.wav
#          .../results.jsonl + summary.json + bench.log
#
# Env:
#   PYTHON             python interpreter (default: 'python'; activate the
#                      OmniVoice env first, or point PYTHON at its interpreter)
#   GPU                physical GPU id (default: 0)
#   DIRECTIONS         manifest setups to sample (default: J_to_E,K_to_E)
#   NUM_PER_DIRECTION  entries per direction (default: 25)
#   ENTRY_OFFSET       skip the first N entries per direction (default: 0)
#   WARMUP             timed-excluded warmup utterances per condition (default: 2)
#   CONDITIONS         condition subset, in order (default: A,B,C)
#   OUT_ROOT           output root (default: _artifacts/bench/bench_backend_out)
#
# Reproducing the reported numbers. The paper/appendix table comes from
#   ENTRY_OFFSET=25 OUT_ROOT=_artifacts/bench/bench_backend_out_run2 \
#       bash analysis/run_bench_backend.sh
# (the first 25 entries per direction were consumed by an earlier pilot run;
# the offset keeps the timed set disjoint from it). The defaults below start
# at offset 0 and therefore time a *different* 50-utterance subset.
#
# Usage:
#   bash analysis/run_bench_backend.sh
#   GPU=7 NUM_PER_DIRECTION=25 bash analysis/run_bench_backend.sh
#   bash analysis/run_bench_backend.sh --dry-run        # plan only, no model load

set -euo pipefail

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DIRECTIONS="${DIRECTIONS:-J_to_E,K_to_E}"
NUM_PER_DIRECTION="${NUM_PER_DIRECTION:-25}"
ENTRY_OFFSET="${ENTRY_OFFSET:-0}"
WARMUP="${WARMUP:-2}"
CONDITIONS="${CONDITIONS:-A,B,C}"
OUT_ROOT="${OUT_ROOT:-_artifacts/bench/bench_backend_out}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="analysis/bench_attention_backend.py"
LOG_DIR="$OUT_ROOT"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bench.log"

echo "================================================================"
echo "[bench_backend] gpu=$GPU directions=$DIRECTIONS n/dir=$NUM_PER_DIRECTION offset=$ENTRY_OFFSET"
echo "[bench_backend] warmup=$WARMUP conditions=$CONDITIONS out=$OUT_ROOT"
echo "[bench_backend] log=$LOG"
echo "================================================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" \
  --directions "$DIRECTIONS" \
  --num-per-direction "$NUM_PER_DIRECTION" \
  --entry-offset "$ENTRY_OFFSET" \
  --warmup "$WARMUP" \
  --conditions "$CONDITIONS" \
  --out-root "$OUT_ROOT" \
  "$@" 2>&1 | tee "$LOG"

echo "[bench_backend] DONE"
echo "    per-utterance -> $OUT_ROOT/results.jsonl"
echo "    summary       -> $OUT_ROOT/summary.json"

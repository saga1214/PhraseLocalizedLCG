#!/bin/bash
#
# Cross-lingual attention-probe analysis (Sec. 4.1, Fig. 3 of the paper).
#
# Runs analysis/attention_probe_dataset.py over a grid of 65 layer-range
# setups × {mean, max} head reductions for the audio→text attention map,
# under two conditions:
#   (A) baseline   --lang-override none   (true language tag)
#   (B) cross      --lang-override swap   (inverted language tag)
# Phrase ground-truth comes from forced alignment on the TRUE language in both.
#
# Sample-sharding splits each condition across NUM_SAMPLE_SHARDS parallel
# jobs; total GPUs used = 2 × NUM_SAMPLE_SHARDS. After all shards finish,
# analysis/merge_xlang_probe.py stitches the per-shard outputs into a single
# metrics.json + summary.csv per condition.
#
# Env overrides:
#   PYTHON              python interpreter (default: 'python')
#   QWEN_PY             interpreter with Qwen3-ForcedAligner installed
#   NUM_SAMPLE_SHARDS   samples per condition split across N GPUs (default 4)
#   GPU_BASE            first physical GPU id (default 0)
#   N                   --num-samples per dataset (default 100)
#   NUM_LAYERS          number of LLM layers (default 28)
#   OUT_BASE            top-level output dir
#   RUN_ONLY=A|B|both   default 'both'
#   SKIP_MERGE=1        leave per-shard outputs unmerged
#   ADD_VNORM=1         add value-norm-weighted (a2t_vnorm) twin setups
#   ADD_VNORMFX=1       add full ||W_O^h v_j||-weighted (a2t_vnormfx) twins

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
QWEN_PY="${QWEN_PY:-python}"
NUM_SAMPLE_SHARDS="${NUM_SAMPLE_SHARDS:-4}"
GPU_BASE="${GPU_BASE:-0}"
RUN_TS=$(date +%Y%m%d_%H%M%S)
N="${N:-100}"
OUT_BASE="${OUT_BASE:-_artifacts/probe/attention_probe_xlang__${RUN_TS}}"
FA_BATCH_SIZE="${FA_BATCH_SIZE:-0}"
NUM_LAYERS="${NUM_LAYERS:-28}"
RUN_ONLY="${RUN_ONLY:-both}"
SKIP_MERGE="${SKIP_MERGE:-0}"

# 65 layer ranges × {mean, max} = 130 setups
RANGES=()
for i in $(seq 0 $((NUM_LAYERS - 1))); do
  RANGES+=("layer_${i}")
done
RANGES+=(
  "layers_8-13"  "layers_8-12"  "layers_9-13"  "layers_12-13"
  "layers_11-13" "layers_7-13"  "layers_8-14"
  "layers_10-14" "layers_10-15" "layers_8-15" "layers_8-18"
)
RANGES+=(
  "layers_set_7-8"   "layers_set_7-9"   "layers_set_7-12"  "layers_set_7-13"
  "layers_set_8-9"   "layers_set_8-12"  "layers_set_8-13"
  "layers_set_9-12"  "layers_set_9-13"
  "layers_set_12-13"
  "layers_set_7-8-9"    "layers_set_7-8-12"   "layers_set_7-8-13"
  "layers_set_7-9-12"   "layers_set_7-9-13"   "layers_set_7-12-13"
  "layers_set_8-9-12"   "layers_set_8-9-13"   "layers_set_8-12-13"
  "layers_set_9-12-13"
  "layers_set_7-8-9-12"   "layers_set_7-8-9-13"   "layers_set_7-8-12-13"
  "layers_set_7-9-12-13"  "layers_set_8-9-12-13"
  "layers_set_7-8-9-12-13"
)

SETUPS=""
for rng in "${RANGES[@]}"; do
  SETUPS="${SETUPS}${rng}__mean__a2t,${rng}__max__a2t,"
done
SETUPS="${SETUPS%,}"

# --- Value-norm-weighted attention variants (Kobayashi et al., EMNLP 2020) ---
# Since attention weights are >= 0, ||alpha_ij * v_j|| == alpha_ij * ||v_j||;
# the *__a2t_vnorm setups reweight the audio->text attention block by the
# key-side value norms before the argmax. *__a2t_vnormfx additionally folds in
# the per-head o_proj slice: ||alpha_ij * W_O^h v_j||.
# VNORM_RANGES = the top raw-a2t comparator ranges (layers_set_8-12 anchor +
# the strongest mid-layer index-set/range ensembles from the grid above).
# Enable with e.g.:
#   ADD_VNORM=1 bash analysis/run_attention_probe.sh
#   ADD_VNORM=1 ADD_VNORMFX=1 bash analysis/run_attention_probe.sh
ADD_VNORM="${ADD_VNORM:-0}"
ADD_VNORMFX="${ADD_VNORMFX:-0}"
VNORM_RANGES=(
  "layers_set_8-12" "layers_set_8-13" "layers_set_9-12" "layers_set_12-13"
  "layers_8-13" "layers_set_7-8-9-12-13"
)
if [[ "$ADD_VNORM" == "1" ]]; then
  for rng in "${VNORM_RANGES[@]}"; do
    SETUPS="${SETUPS},${rng}__mean__a2t_vnorm,${rng}__max__a2t_vnorm"
  done
fi
if [[ "$ADD_VNORMFX" == "1" ]]; then
  for rng in "${VNORM_RANGES[@]}"; do
    SETUPS="${SETUPS},${rng}__mean__a2t_vnormfx,${rng}__max__a2t_vnormfx"
  done
fi

NUM_SETUPS=$(echo "$SETUPS" | tr "," "\n" | wc -l)
mkdir -p "$OUT_BASE"

run_shard() {
  # $1=condition (baseline|cross), $2=override (none|swap),
  # $3=sample_shard, $4=gpu_id, $5=out_dir
  local cond="$1" override="$2" sshard="$3" gpu="$4" out_dir="$5"
  mkdir -p "$out_dir"
  echo "[xlang/$cond/shard$sshard] gpu=$gpu override=$override sample-shard=$sshard/$NUM_SAMPLE_SHARDS N=$N setups=$NUM_SETUPS out=$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" analysis/attention_probe_dataset.py \
    --datasets "en,ko" \
    --num-samples "$N" \
    --num-sample-shards "$NUM_SAMPLE_SHARDS" \
    --sample-shard "$sshard" \
    --fa-batch-size "$FA_BATCH_SIZE" \
    --attn-implementation eager \
    --lang-override "$override" \
    --edit-aligner-python "$QWEN_PY" \
    --setups "$SETUPS" \
    --output-dir "$out_dir" \
    > "$out_dir/run.log" 2>&1
  local rc=$?
  echo "[xlang/$cond/shard$sshard] DONE (exit=$rc). See $out_dir/{metrics.json,summary.csv,run.log}"
  return $rc
}

pids=()
labels=()
condition_dirs_A=()
condition_dirs_B=()
gpu_idx=$GPU_BASE

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "A" ]]; then
  CONDITION_DIR_A="${OUT_BASE}/baseline"
  mkdir -p "$CONDITION_DIR_A"
  for ((s=0; s<NUM_SAMPLE_SHARDS; s++)); do
    out_dir="${CONDITION_DIR_A}/shard${s}"
    run_shard baseline none "$s" "$gpu_idx" "$out_dir" &
    pids+=($!)
    labels+=("baseline/shard$s gpu=$gpu_idx")
    condition_dirs_A+=("$out_dir")
    gpu_idx=$((gpu_idx + 1))
  done
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "B" ]]; then
  CONDITION_DIR_B="${OUT_BASE}/cross"
  mkdir -p "$CONDITION_DIR_B"
  for ((s=0; s<NUM_SAMPLE_SHARDS; s++)); do
    out_dir="${CONDITION_DIR_B}/shard${s}"
    run_shard cross swap "$s" "$gpu_idx" "$out_dir" &
    pids+=($!)
    labels+=("cross/shard$s gpu=$gpu_idx")
    condition_dirs_B+=("$out_dir")
    gpu_idx=$((gpu_idx + 1))
  done
fi

echo "[xlang] Launched ${#pids[@]} parallel jobs."
fail=0
for j in "${!pids[@]}"; do
  if ! wait "${pids[$j]}"; then
    echo "[xlang] FAILED: ${labels[$j]} (pid=${pids[$j]})"
    fail=1
  fi
done
[[ $fail -ne 0 ]] && { echo "[xlang] At least one shard failed. Check $OUT_BASE/*/shard*/run.log" >&2; exit 1; }

if [[ "$SKIP_MERGE" == "1" ]]; then
  echo "[xlang] SKIP_MERGE=1 — leaving per-shard outputs as-is."
  exit 0
fi

echo "[xlang] Merging shard outputs ..."
[[ ${#condition_dirs_A[@]} -gt 0 ]] && "$PYTHON" analysis/merge_xlang_probe.py \
    --input-dirs "${condition_dirs_A[@]}" \
    --output-dir "$OUT_BASE/baseline/merged"
[[ ${#condition_dirs_B[@]} -gt 0 ]] && "$PYTHON" analysis/merge_xlang_probe.py \
    --input-dirs "${condition_dirs_B[@]}" \
    --output-dir "$OUT_BASE/cross/merged"
echo "[xlang] ALL DONE."

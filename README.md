# Phrase-Localized Language-Contrastive Guidance

**Training-Free Localized Accent Control for Code-Switching Text-to-Speech**

[![arXiv](https://img.shields.io/badge/arXiv-2609.01016-b31b1b)](https://arxiv.org/abs/2609.01016)
[![EMNLP](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-b31b1b)](https://2026.emnlp.org/)
[![Demo](https://img.shields.io/badge/🔊-Audio%20Demo-blue)](https://saga1214.github.io/PhraseLocalizedLCG/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Reference implementation and benchmark for **Phrase-Localized Language-Contrastive
Guidance (LCG)** — a training-free, inference-time control framework for
discrete-diffusion TTS that removes cross-lingual accent leakage on
intra-sentential code-switching input.

When a foreign phrase is embedded in a sentence, zero-shot TTS models pronounce
it with the *matrix* language's accent, because classifier-free guidance applies
a single language condition to the whole utterance. LCG adds a third,
phrase-language-conditioned branch and gates it with a frame-level mask derived
from the model's own self-attention, so each region is guided by its own
language. No fine-tuning, no auxiliary aligner.

🔊 **[Listen to audio samples](https://saga1214.github.io/PhraseLocalizedLCG/)**

---

## Highlights

Across 12 directional language pairs (macro-average over 1,200 utterances):

| | Baseline | **+ LCG (λ=7)** |
|---|---|---|
| Mixed Error Rate ↓ | 0.564 | **0.445** |
| Embedded-phrase language accuracy ↑ | 0.233 | **0.518** |
| Embedded-phrase LID confidence ↑ | 0.247 | **0.588** |
| Speaker similarity ↑ | 0.973 | 0.969 |

Native listeners preferred LCG over the unguided baseline in **75.5%** of blind
A/B trials on embedded-phrase nativeness, with global quality MOS statistically
unchanged.

## Repository layout

```
benchmark/     1,200-utterance code-switching corpus + schema docs
examples/      5 monolingual reference voice prompts
synth/         LCG sampling engine (3-branch CFG, attention mask extraction)
eval/          MER / LA / LID / speaker similarity / UTMOS
aggregate/     per-direction and per-system summary tables
analysis/      attention probe, backend benchmark, external-system runner
results/       stored summaries backing the paper's appendix tables
docs/          source of the audio demo page
scripts/       multi-GPU launchers
```

## Installation

Two environments are used. The **main** environment runs synthesis and all
objective metrics. The **forced-alignment** environment hosts only
Qwen3-ForcedAligner, kept separate because its `transformers` / `flash-attn`
requirements conflict with the main one.

Reference environment: Python 3.12.13, CUDA 12.8, `torch==2.8.0+cu128`,
`torchaudio==2.8.0+cu128`, `transformers==5.8.1`, `numpy==2.4.5`,
`librosa==0.11.0`, `accelerate==1.13.0`, `openai-whisper==20250625`,
`jiwer==3.1.0`, `speechbrain==1.1.0`.

```bash
git clone https://github.com/saga1214/PhraseLocalizedLCG
cd PhraseLocalizedLCG

conda create -n lcg python=3.12 -y && conda activate lcg
pip install uv

# torch/torchaudio: match the CUDA wheel index to your driver
uv pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install -e .
```

Installing this package pulls in the OmniVoice backbone pinned to the exact
upstream commit used for every number in the paper
(`k2-fsa/OmniVoice@6a3f23d`). LCG does not patch that package: the three-branch
CFG and the attention-based mask extraction live in `synth/` and read attentions
by calling `model.llm(..., output_attentions=True)` externally.

<details>
<summary>Forced-alignment environment (needed for phrase-level metrics)</summary>

```bash
conda create -n qwen3-asr python=3.12 -y && conda activate qwen3-asr
pip install uv
uv pip install -U qwen-asr
uv pip install -U flash-attn --no-build-isolation

export QWEN_PY="$(conda run -n qwen3-asr which python)"
```

Forced alignment is **required** for the phrase-level metrics (LA_e, LID_e,
MER_e): there is no proportional-span fallback, and a wav whose spans cannot be
aligned is recorded as a failure rather than scored. MER, SIM and UTMOS do not
need this environment.
</details>

The first synthesis call downloads the OmniVoice 0.81B checkpoint from the
Hugging Face Hub.

## Reproducing the benchmark

**1. Synthesize.**

```bash
bash scripts/run_synth.sh                 # full λ sweep, 8 GPUs
LAMBDAS=7 bash scripts/run_synth.sh       # just the "Ours" configuration
```

This runs `synth/synth.py` over the benchmark with the paper's final mask
refinement (margin `k=4`, dual-tag union → suffix `m4_xu`) and the attention
layer ensemble `{8, 12}` with head-max pooling in the audio→text direction.
Each entry directory receives:

| File | System |
|---|---|
| `baseline_src_only.wav` | unguided baseline (λ=0) |
| `cs_attention_plc__..._lambda3.wav` | Swap (λ=3) |
| `cs_attention_plc__..._lambda7.wav` | **Ours** (λ=7) |
| `..._lambda{5,9,11,13}.wav` | sweep points for Fig. 4 |

**2. Evaluate.**

```bash
python -m eval all --synth-dir synth_l8-12_m4_xu --num-shards 8 --gpu-shard 0
```

Individual metrics: `mer_la_lid`, `matrix_lid`, `sim`, `utmos`.
`scripts/run_eval.sh` and `scripts/run_utmos.sh` provide multi-GPU launchers.

**3. Aggregate.**

```bash
python aggregate/aggregate.py --synth-dir synth_l8-12_m4_xu
```

## Key hyperparameters

| Parameter | Value | Flag |
|---|---|---|
| Language-steering scale λ | 7 (Ours), 3 (Swap), 0 (baseline) | `--lambdas` |
| Global text guidance γ | 2.0 | — |
| Denoising steps T | 32 | — |
| Attention layers | {8, 12}, head-max, audio→text | `--attention-setup` |
| Margin dilation k | 4 (effective radius 10) | `--phrase-mask-margin` |
| Dual-tag union | on | `--phrase-mask-xlang-union` |

## Additional experiments

| Experiment | Code | Stored summary |
|---|---|---|
| Value-norm-weighted attention probe | `analysis/attention_probe*.py`, `analysis/run_attention_probe.sh` | `results/attention_probe_summary.csv` |
| Attention-backend runtime cost (SDPA vs eager vs LCG) | `analysis/bench_attention_backend.py`, `analysis/run_bench_backend.sh` | `results/bench_backend_summary.json` |
| External system reference (MOSS-TTS-v1.5) | `analysis/moss_tts_runner.py` | `results/mosstts_per_utterance.csv` |

The paper additionally reports an evaluation on human-authored code-switching
transcripts (Appendix E). That corpus is distributed under terms that do not
permit redistribution, so no data or preparation code for it is included here;
the results are reported in the paper.

## License

Code in this repository is released under **Apache-2.0** (see `LICENSE`).

The reference voice prompts in `examples/` are excerpts from public monolingual
corpora and follow the terms of their source projects — see
[`examples/README.md`](examples/README.md) for per-file attribution. Note that
`ko_ref.wav` comes from the KSS Dataset, which is **non-commercial**.

## Citation

Paper: [arXiv:2609.01016](https://arxiv.org/abs/2609.01016) (accepted to
**EMNLP 2026, Main Conference**; the proceedings entry will be added once
published).

```bibtex
@misc{lee2026phraselocalizedlanguagecontrastiveguidancetrainingfree,
  title         = {Phrase-Localized Language-Contrastive Guidance: Training-Free Localized Accent Control for Code-Switching Text-to-Speech},
  author        = {Che Hyun Lee and Sangkwon Park and Donghun Kang and Dongwook Lee and Youngho Cho and Heeseung Kim and Sungroh Yoon},
  year          = {2026},
  eprint        = {2609.01016},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.01016},
}
```

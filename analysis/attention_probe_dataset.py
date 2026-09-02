#!/usr/bin/env python3
# Copyright 2026  Xiaomi Corp.
#
# See ../LICENSE for clarification regarding multiple authors

"""Multi-sample attention probe on LJSpeech (EN) and KSS (KO).

For each dataset (loaded via HuggingFace ``datasets``):
  - The first utterance is the reference voice prompt.
  - The next N (default 100) utterances are evaluated.

For each evaluation sample:
  - The text is split on whitespace; every word / 어절 is treated as a phrase.
  - The model synthesizes audio from the text (eager forward, attention
    captured per step).
  - The Qwen3 forced aligner runs on the synthesized audio to derive
    per-word audio-token GT spans (one mask per word + a ``__all__`` union).
  - For each captured step and each setup, the per-audio argmax text index
    is computed once (reusing ``cs_attention_probe`` 's a2t pipeline) and
    indexed against each word's text mask to build a predicted per-word
    audio mask. We then compute IoU / Recall / Precision / F1 against the
    FA-derived GT.
  - The utterance-level score for a setup is the **word-mean over steps**.
  - Across N utterances we report mean / std / quartiles and rank setups.

Only the ``a2t`` direction (audio query → text key) is supported in this
script, plus its value-norm-weighted variants (Kobayashi et al., EMNLP 2020):
``a2t_vnorm`` multiplies the attention block by the key-side value norms
``||v_j||`` before the argmax (since alpha >= 0,
``||alpha_ij v_j|| = alpha_ij ||v_j||``), and ``a2t_vnormfx`` uses the full
``||W_O^h v_j||`` with per-head o_proj slices. Direction is enforced by the
setup id suffix (``...__a2t`` / ``...__a2t_vnorm`` / ``...__a2t_vnormfx``);
value norms are only captured when at least one requested setup needs them.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in (_ROOT, _ROOT / "synth", _THIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
from omnivoice.models.omnivoice import _combine_text, _gumbel_sample  # noqa: E402

from dlm_sampling_experiments import (  # noqa: E402
    _alloc_decode_batch,
    _align_qwen3_forced_batch,
    _prepare_decode_branch,
    _resolve_qwen_aligner_language,
    add_shared_model_and_sampling_args,
    build_generation_config,
    collect_phrase_char_spans,
    decode_tokens,
    ensure_dir,
    get_best_device,
    make_voice_prompt,
    prepare_single_item,
    resolve_dtype,
    set_seed,
    str2bool,
    write_json,
)
from attention_probe import (  # noqa: E402
    _binary_set_metrics,
    _build_schedule,
    _layer_slice,
    _length_of_tokens,
    _omnivoice_forward_with_attentions,
    _omnivoice_forward_with_attentions_and_vnorms,
    _phrase_text_mask,
    _predict_with_log_probs_simple,
    _sanitize_for_json,
    _ValueNormCapture,
    build_phrase_audio_gt_from_fa,
    map_phrase_to_text_token_range,
)

logger = logging.getLogger(__name__)


# 22 setup grid (option A):
#  - 7 single layers (early -> last)
#  - 4 multi-layer ranges (Whisper baseline, last_4, mid-late ensemble, all)
#  - each layer/range tried with both mean and max head reductions
# All a2t direction (t2a excluded based on prior experiments).
DEFAULT_SETUPS: list[str] = [
    # ----- Single layer × {mean, max} -----
    "layer_3__mean__a2t",   "layer_3__max__a2t",
    "layer_6__mean__a2t",   "layer_6__max__a2t",
    "layer_9__mean__a2t",   "layer_9__max__a2t",
    "layer_12__mean__a2t",  "layer_12__max__a2t",
    "layer_15__mean__a2t",  "layer_15__max__a2t",
    "layer_18__mean__a2t",  "layer_18__max__a2t",
    "layer_24__mean__a2t",  "layer_24__max__a2t",
    # ----- Range × {mean, max} -----
    "last_1__mean__a2t",          "last_1__max__a2t",            # Whisper-style baseline
    "last_4__mean__a2t",          "last_4__max__a2t",            # last-4 ensemble
    "layers_10-15__mean__a2t",    "layers_10-15__max__a2t",      # mid-late ensemble
    "all__mean__a2t",             "all__max__a2t",               # global
]
METRIC_KEYS: tuple[str, ...] = ("iou", "recall", "precision", "f1")


# Direction tokens: plain audio→text plus its value-norm-weighted variants
# (Kobayashi et al., EMNLP 2020). The ``a2t_*`` weighting name (after the
# ``a2t_`` prefix) must match a key produced by ``_ValueNormCapture.collect``.
SUPPORTED_DIRECTIONS: tuple[str, ...] = ("a2t", "a2t_vnorm", "a2t_vnormfx")


def _parse_setup(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Setup must be 'layer_range__head_reduce__direction', got {name!r}"
        )
    lr, hr, dr = parts
    if dr not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            "This script only supports the a2t direction and its value-norm-"
            f"weighted variants {SUPPORTED_DIRECTIONS[1:]} (got {name!r})."
        )
    if hr not in ("sum", "mean", "max"):
        raise ValueError(f"Unknown head_reduce: {hr!r}")
    return lr, hr, dr


def _weightings_needed(setups: list[tuple[str, str, str]]) -> list[str]:
    """Value-norm weighting names required by ``setups`` (e.g. ['vnorm']).

    Empty when every setup uses the plain ``a2t`` direction, in which case no
    v_proj capture hooks are registered (zero overhead).
    """
    return sorted(
        {dr[len("a2t_"):] for _, _, dr in setups if dr.startswith("a2t_")}
    )


# ---------------------------------------------------------------------------
# Instrumented decoding: capture per-step per-setup text-idx-per-audio.
# (FA GT is built *after* synthesis, so the loop just produces the int arrays
# and we compute metrics offline.)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _decode_capture_text_idx(
    model: OmniVoice,
    item,
    gen_config: OmniVoiceGenerationConfig,
    text_token_offset: int,
    text_token_end: int,
    target_start: int,
    setups: list[tuple[str, str, str]],
    max_steps: int = -1,
    vnorm_capture: Optional[_ValueNormCapture] = None,
) -> dict[str, Any]:
    weightings = _weightings_needed(setups)
    if weightings and vnorm_capture is None:
        raise ValueError(
            f"Setups request value-norm weighting(s) {weightings} but no "
            "vnorm_capture was provided."
        )
    if "vnormfx" in weightings and not vnorm_capture.capture_fx:
        raise ValueError(
            "a2t_vnormfx setups require a vnorm_capture built with "
            "capture_fx=True."
        )
    device = model.device
    mask_id = model.config.audio_mask_id
    num_codebook = model.config.num_audio_codebook
    t_len = item.target_len
    sample_tokens = torch.full(
        (1, num_codebook, t_len), mask_id, dtype=torch.long, device=device
    )

    cond_branch = _prepare_decode_branch(model, item, gen_config, branch_type="cond")
    uncond_branch = _prepare_decode_branch(
        model, item, gen_config, branch_type="uncond"
    )
    branches = [cond_branch, uncond_branch]
    batch_input_ids, batch_audio_mask, batch_attention_mask = _alloc_decode_batch(
        model, branches
    )
    for idx, branch in enumerate(branches):
        start = branch.target_start
        batch_input_ids[idx, :, start : start + t_len] = sample_tokens[0]

    schedule = _build_schedule(
        total_mask=t_len * num_codebook,
        num_step=gen_config.num_step,
        t_shift=gen_config.t_shift,
    )
    layer_ids = torch.arange(num_codebook, device=device).view(1, -1, 1)
    capture_limit = (
        gen_config.num_step if max_steps <= 0 else min(max_steps, gen_config.num_step)
    )

    captured_step_indices: list[int] = []
    per_step_text_idx: dict[str, list[np.ndarray]] = {
        f"{lr}__{hr}__{dr}": [] for lr, hr, dr in setups
    }

    for step in range(gen_config.num_step):
        if int((sample_tokens == mask_id).sum().item()) == 0:
            break
        capture = step < capture_limit
        if capture:
            if weightings:
                (
                    audio_logits,
                    attentions,
                    value_norms,
                ) = _omnivoice_forward_with_attentions_and_vnorms(
                    model,
                    input_ids=batch_input_ids,
                    audio_mask=batch_audio_mask,
                    attention_mask=batch_attention_mask,
                    vnorm_capture=vnorm_capture,
                )
            else:
                audio_logits, attentions = _omnivoice_forward_with_attentions(
                    model,
                    input_ids=batch_input_ids,
                    audio_mask=batch_audio_mask,
                    attention_mask=batch_attention_mask,
                )
                value_norms = None
            audio_logits = audio_logits.to(torch.float32)
        else:
            audio_logits = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)
            attentions = None
            value_norms = None

        c_logits = audio_logits[
            0:1, :, cond_branch.target_start : cond_branch.target_start + t_len, :
        ]
        u_logits = audio_logits[
            1:2, :, uncond_branch.target_start : uncond_branch.target_start + t_len, :
        ]
        pred_tokens, confidence = _predict_with_log_probs_simple(
            model, c_logits, u_logits, gen_config
        )

        if attentions is not None and capture:
            stacked = torch.stack(attentions, dim=0)  # (L, B, H, S, S)
            cond_attn = stacked[:, 0]  # (L, H, S, S)
            audio_to_text = cond_attn[
                :,
                :,
                target_start : target_start + t_len,
                text_token_offset:text_token_end,
            ].contiguous()
            # Kobayashi value-norm weighting, computed once per step and
            # shared across all *_vnorm* setups:
            #   weighted[l, h, q, k] = attn[l, h, q, k] * ||v_k||
            # where ||v_k|| is the key-side norm for the same text-token
            # window (cond branch = batch index 0), broadcast over the
            # query/audio axis. No renormalization is needed for the argmax.
            weighted_a2t: dict[str, torch.Tensor] = {}
            if value_norms is not None:
                a2t_f32 = audio_to_text.to(torch.float32)
                for wname in weightings:
                    vnorm_text = torch.stack(
                        [n[0] for n in value_norms[wname]], dim=0
                    )[:, :, text_token_offset:text_token_end].to(
                        device=a2t_f32.device
                    )  # (L, H, T_text)
                    if tuple(vnorm_text.shape) != (
                        a2t_f32.size(0),
                        a2t_f32.size(1),
                        a2t_f32.size(3),
                    ):
                        raise RuntimeError(
                            f"Value-norm shape {tuple(vnorm_text.shape)} does "
                            "not align with the attention block "
                            f"{tuple(a2t_f32.shape)}"
                        )
                    weighted_a2t[wname] = a2t_f32 * vnorm_text[:, :, None, :]
            for lr, hr, dr in setups:
                setup_name = f"{lr}__{hr}__{dr}"
                src = (
                    audio_to_text
                    if dr == "a2t"
                    else weighted_a2t[dr[len("a2t_"):]]
                )
                sub = _layer_slice(src, lr)
                if hr == "sum":
                    agg = sub.sum(dim=1).sum(dim=0)
                elif hr == "mean":
                    agg = sub.mean(dim=1).mean(dim=0)
                elif hr == "max":
                    agg = sub.amax(dim=1).amax(dim=0)
                else:
                    raise ValueError(f"Unknown head_reduce: {hr}")
                text_idx = (
                    agg.argmax(dim=-1).detach().to(torch.int64).cpu().numpy()
                )  # (T_audio,)
                per_step_text_idx[setup_name].append(text_idx)
            captured_step_indices.append(int(step))

        # token sampling (same as iterative_decode)
        base_scores = confidence - (layer_ids * gen_config.layer_penalty_factor)
        if gen_config.position_temperature > 0.0:
            base_scores = _gumbel_sample(base_scores, gen_config.position_temperature)
        mask_positions = sample_tokens == mask_id
        selection_scores = base_scores.masked_fill(~mask_positions, -float("inf"))
        k_step = schedule[step]
        if k_step > 0 and torch.any(mask_positions):
            _, topk_idx = torch.topk(selection_scores.flatten(), k_step)
            flat_sample = sample_tokens.flatten()
            flat_pred = pred_tokens.flatten()
            flat_sample[topk_idx] = flat_pred[topk_idx]
            sample_tokens = flat_sample.view_as(sample_tokens)
        for idx, branch in enumerate(branches):
            start = branch.target_start
            batch_input_ids[idx, :, start : start + t_len] = sample_tokens[0]

    return {
        "tokens": sample_tokens.squeeze(0).detach().cpu(),
        "captured_step_indices": captured_step_indices,
        "per_step_text_idx": {
            k: np.stack(v, axis=0) for k, v in per_step_text_idx.items() if v
        },
    }


# ---------------------------------------------------------------------------
# Offline metric computation (after FA produces phrase audio GT).
# ---------------------------------------------------------------------------


def _per_word_metrics_per_step(
    per_step_text_idx: dict[str, np.ndarray],  # (S, T_audio) int
    phrase_text_masks: dict[str, np.ndarray],
    phrase_audio_gt_masks: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Compute per-setup × per-word × per-metric × per-step values via
    fancy-index lookup on the captured text-idx arrays.
    """
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for setup_name, ti_arr in per_step_text_idx.items():
        S, T_audio = ti_arr.shape
        per_phrase: dict[str, dict[str, list[float]]] = {}
        for phrase, tmask in phrase_text_masks.items():
            gt = phrase_audio_gt_masks[phrase]
            if not tmask.any() or not gt.any():
                per_phrase[phrase] = {
                    k: [float("nan")] * S for k in METRIC_KEYS
                }
                continue
            # pred[s, a] = tmask[ti_arr[s, a]]
            pred = tmask[ti_arr]  # (S, T_audio) bool
            inter = np.logical_and(pred, gt[None, :]).sum(axis=-1)
            union = np.logical_or(pred, gt[None, :]).sum(axis=-1)
            pred_n = pred.sum(axis=-1).astype(np.float64)
            gt_n = float(gt.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
                recall = (
                    inter.astype(np.float64) / gt_n if gt_n > 0
                    else np.full(S, np.nan)
                )
                precision = np.where(
                    pred_n > 0, inter / np.maximum(pred_n, 1), np.nan
                )
                denom = recall + precision
                f1 = np.where(
                    denom > 0,
                    2 * recall * precision / np.where(denom == 0, 1, denom),
                    np.nan,
                )
            per_phrase[phrase] = {
                "iou": [float(x) for x in iou.tolist()],
                "recall": [float(x) for x in recall.tolist()],
                "precision": [float(x) for x in precision.tolist()],
                "f1": [float(x) for x in f1.tolist()],
            }
        out[setup_name] = per_phrase
    return out


def _word_mean_per_step(
    sample_metrics: dict[str, dict[str, dict[str, list[float]]]],
    exclude_phrases: tuple[str, ...] = ("__all__",),
) -> dict[str, dict[str, list[float]]]:
    """For each setup, compute the per-step mean across all words (skip
    ``__all__`` and NaNs). Returns dict[setup] -> dict[metric] -> list."""
    out: dict[str, dict[str, list[float]]] = {}
    for setup, per_phrase in sample_metrics.items():
        first_traj = next(iter(per_phrase.values()))["iou"]
        S = len(first_traj)
        per_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
        for s in range(S):
            for metric in METRIC_KEYS:
                vals = [
                    per_phrase[w][metric][s]
                    for w in per_phrase
                    if w not in exclude_phrases
                ]
                vals = [v for v in vals if not math.isnan(v)]
                per_metric[metric].append(
                    float(np.mean(vals)) if vals else float("nan")
                )
        out[setup] = per_metric
    return out


def _trajectory_summary(traj: list[float]) -> dict[str, float]:
    arr = np.array(
        [x for x in traj if not (isinstance(x, float) and math.isnan(x))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {
            "first": float("nan"),
            "last": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# HuggingFace datasets loader helpers
# ---------------------------------------------------------------------------


def _load_dataset_hf(hf_id: str, cache_dir: Optional[str], split: str = "train"):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(hf_id, split=split, **kwargs)


def _audio_array_from_entry(entry) -> tuple[np.ndarray, int]:
    audio = entry.get("audio")

    # Case 1: newest HF datasets returns a torchcodec ``AudioDecoder`` object.
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        # data may be a torch tensor of shape (channels, n_samples) or (n_samples,).
        if hasattr(data, "numpy"):
            arr = data.numpy()
        else:
            arr = np.asarray(data)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] <= 4 else arr[:, 0]
        sr = int(samples.sample_rate)
        return arr.astype(np.float32, copy=False), sr

    # Case 2: HF-style dict { "array", "sampling_rate" }
    if isinstance(audio, dict) and "array" in audio:
        return (
            np.asarray(audio["array"], dtype=np.float32),
            int(audio["sampling_rate"]),
        )

    # Case 3: dict pointing at a file path
    if isinstance(audio, dict) and audio.get("path"):
        arr, sr = sf.read(audio["path"])
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return np.asarray(arr, dtype=np.float32), int(sr)

    # Case 4: bare path string
    if isinstance(audio, str):
        arr, sr = sf.read(audio)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return np.asarray(arr, dtype=np.float32), int(sr)

    raise ValueError(
        f"Could not extract audio from entry (audio type={type(audio)}). "
        f"Supported: torchcodec AudioDecoder, dict with 'array'/'path', or "
        f"path string."
    )


_TEXT_COLUMN_PREFERENCE = (
    "normalized_text",   # LJSpeech (some versions)
    "text",              # LJSpeech / generic
    "original_script",   # KSS (raw form — keeps digits/abbreviations as written)
    "expanded_script",   # KSS fallback (digits spelled out in Korean)
    "transcript",        # other common naming
    "sentence",
)


def _entry_text(entry) -> str:
    for key in _TEXT_COLUMN_PREFERENCE:
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError(
        f"No text-like column found. Tried keys={list(_TEXT_COLUMN_PREFERENCE)}; "
        f"row has keys={list(entry.keys())}"
    )


def _make_voice_prompt_from_array(
    model: OmniVoice,
    args: argparse.Namespace,
    wav_array: np.ndarray,
    sr: int,
    ref_text: str,
):
    """Save the in-memory waveform to a temp wav so we can reuse the
    standard ``make_voice_prompt`` code path. The temp file persists for the
    lifetime of the process (we keep its handle on args so it doesn't get
    GC'd or unlinked mid-run)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav = wav_array
    if wav.ndim > 1:
        wav = wav[:, 0]
    sf.write(tmp.name, wav.astype(np.float32), int(sr))
    args.ref_audio = tmp.name
    args.ref_text = ref_text
    args._ref_tmp_handle = tmp  # keep alive
    return make_voice_prompt(model, args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Dataset-level attention probe (LJSpeech EN + KSS KO). "
            "Treats each word/어절 as a phrase and measures per-setup "
            "IoU/Recall/Precision/F1 against FA-derived word audio spans."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_model_and_sampling_args(p)
    p.add_argument(
        "--datasets",
        type=str,
        default="en,ko",
        help="Comma-separated subset of {en, ko}.",
    )
    p.add_argument(
        "--lang-override",
        type=str,
        choices=["none", "en", "ko", "swap"],
        default="none",
        help="Override the model's language tag without affecting FA/GT alignment. "
        "'none' = use dataset-true tag (en for LJSpeech, ko for KSS). "
        "'en'/'ko' = force that tag for all samples. "
        "'swap' = invert (LJ rows get ko, KSS rows get en).",
    )
    p.add_argument("--lj-hf-id", type=str, default="Inku004/lj-speech")
    p.add_argument(
        "--lj-split", type=str, default="full",
        help="Split name for the LJSpeech HF dataset (Inku004/lj-speech uses 'full').",
    )
    p.add_argument("--kss-hf-id", type=str, default="Bingsu/KSS_Dataset")
    p.add_argument(
        "--kss-split", type=str, default="train",
        help="Split name for the KSS HF dataset.",
    )
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument(
        "--num-sample-shards",
        type=int,
        default=1,
        help="Total number of sample shards. When > 1, only samples whose "
        "(idx-1) %% num_sample_shards == sample_shard are processed; other "
        "samples are skipped. Use with --save-per-sample True (auto-set) so "
        "examples/merge_xlang_probe.py can stitch the shards back together.",
    )
    p.add_argument(
        "--sample-shard",
        type=int,
        default=0,
        help="This invocation's sample-shard index in [0, num_sample_shards). "
        "Ignored when --num-sample-shards=1.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If >0, only capture attention metrics for the first N decoding "
        "steps (sampling still completes).",
    )
    p.add_argument("--attn-implementation", type=str, default="eager")
    p.add_argument(
        "--setups",
        type=str,
        default=",".join(DEFAULT_SETUPS),
        help="Comma-separated setup ids 'layer_range__head_reduce__direction'. "
        "Supported directions: 'a2t' plus its value-norm-weighted variants "
        "'a2t_vnorm' (attn * ||v_j||) and 'a2t_vnormfx' (attn * ||W_O^h v_j||; "
        "Kobayashi et al., EMNLP 2020).",
    )
    # FA aligner args (same names as elsewhere in the project)
    p.add_argument(
        "--edit-aligner-model", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B"
    )
    p.add_argument("--edit-aligner-conda-env", type=str, default=None)
    p.add_argument("--edit-aligner-conda-executable", type=str, default="conda")
    p.add_argument(
        "--edit-aligner-python", type=str, default=None,
        help="Path to a python interpreter with Qwen3-ForcedAligner installed. "
             "Falls back to the current interpreter when omitted.",
    )
    p.add_argument("--edit-aligner-cuda-visible-devices", type=str, default=None)
    p.add_argument("--edit-aligner-timeout-sec", type=float, default=0.0)
    p.add_argument("--edit-aligner-language", type=str, default=None)
    p.add_argument("--edit-aligner-device", type=str, default=None)
    p.add_argument(
        "--edit-aligner-dtype",
        type=str,
        choices=["auto", "float16", "float32", "bfloat16"],
        default="bfloat16",
    )
    p.add_argument("--edit-aligner-attn-implementation", type=str, default=None)
    p.add_argument("--edit-aligner-margin-sec", type=float, default=0.0)
    p.add_argument(
        "--phrase-ignore-case",
        type=str2bool,
        default=False,
        help="Case-sensitive matching by default for word identity.",
    )
    p.add_argument(
        "--save-per-sample",
        type=str2bool,
        default=False,
        help="If True, dump per-sample raw metrics in metrics.json (large).",
    )
    p.add_argument(
        "--save-synthesized-audio",
        type=str2bool,
        default=False,
        help="If True, save each synthesized wav alongside metrics (for "
        "debugging FA failures).",
    )
    p.add_argument(
        "--fa-batch-size",
        type=int,
        default=0,
        help="Mini-batch size for the forced-aligner subprocess. 0 = one batch "
        "(all samples in a single subprocess call). Larger batches amortize the "
        "model load cost; 0 is fastest if all samples fit in worker memory.",
    )
    p.add_argument(
        "--allow-fa-batch-fallback",
        type=str2bool,
        default=True,
        help="If the batch FA subprocess fails (e.g. running against an old "
        "worker), fall back to per-sample single calls (slower but works).",
    )
    return p


def main():
    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)
    args = build_parser().parse_args()
    set_seed(args.seed)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis") / "outputs" / "attention_probe_dataset"
    )
    ensure_dir(out_dir)

    device = args.device or get_best_device()
    if args.edit_aligner_device is None:
        args.edit_aligner_device = device
    dtype = resolve_dtype(args.dtype, device)
    logger.info(
        "Loading model=%s device=%s dtype=%s attn_impl=%s",
        args.model, device, dtype, args.attn_implementation,
    )
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    gen_config = build_generation_config(args)

    setups = [_parse_setup(s.strip()) for s in args.setups.split(",") if s.strip()]
    if not setups:
        raise ValueError("--setups list is empty")
    setup_names = [f"{lr}__{hr}__{dr}" for lr, hr, dr in setups]
    logger.info("Setups: %s", setup_names)

    # Register v_proj value-norm hooks only when a setup actually needs them
    # (a2t_vnorm / a2t_vnormfx); plain-a2t runs are untouched.
    weightings = _weightings_needed(setups)
    vnorm_capture: Optional[_ValueNormCapture] = None
    if weightings:
        logger.info(
            "Value-norm weighting requested by setups (%s); enabling capture.",
            weightings,
        )
        vnorm_capture = _ValueNormCapture(
            model, capture_fx=("vnormfx" in weightings)
        )

    datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets_to_run:
        if d not in ("en", "ko"):
            raise ValueError(f"--datasets must be subset of en,ko; got {d}")

    per_dataset_results: dict[str, dict[str, Any]] = {}

    for ds_tag in datasets_to_run:
        if ds_tag == "en":
            hf_id = args.lj_hf_id
            language = "en"
            split = args.lj_split
        else:
            hf_id = args.kss_hf_id
            language = "ko"
            split = args.kss_split

        logger.info(
            "Loading dataset (%s) from %s [split=%s] ...", ds_tag, hf_id, split
        )
        ds = _load_dataset_hf(hf_id, args.hf_cache_dir, split=split)
        logger.info("Dataset size: %d", len(ds))
        if len(ds) < args.num_samples + 1:
            raise RuntimeError(
                f"Dataset too small: {len(ds)} < {args.num_samples + 1}"
            )

        ref_entry = ds[0]
        ref_audio, ref_sr = _audio_array_from_entry(ref_entry)
        ref_text = _entry_text(ref_entry)
        logger.info("Ref idx=0 (%s) | text=%r", ds_tag, ref_text[:80])

        args.language = language
        voice_prompt = _make_voice_prompt_from_array(
            model, args, ref_audio, ref_sr, ref_text
        )
        if voice_prompt is None:
            raise RuntimeError(f"Failed to build voice prompt for {ds_tag}")

        per_sample_word_mean: list[dict[str, dict[str, list[float]]]] = []
        per_sample_raw: list[dict[str, Any]] = []
        sample_failures: list[dict[str, Any]] = []

        # ---------- Phase 1: synthesize + capture attention for every sample ----------
        # We collect everything needed for the offline metric calc; FA is deferred
        # so it can be batched in Phase 2.
        pending: list[dict[str, Any]] = []
        aligner_lang = _resolve_qwen_aligner_language(
            args.edit_aligner_language or args.language
        )
        frame_rate = model.audio_tokenizer.config.frame_rate
        # Resolve model language tag for this dataset under --lang-override.
        # FA / aligner_lang above already captured the TRUE language so GT
        # alignment is untouched; only the model's <|lang_start|> token swaps.
        if args.lang_override == "none":
            model_lang = language
        elif args.lang_override == "swap":
            model_lang = "ko" if language == "en" else "en"
        else:
            model_lang = args.lang_override
        logger.info(
            "Lang tag: dataset_true=%s -> model_lang=%s (override=%s); FA stays on %s",
            language, model_lang, args.lang_override, aligner_lang,
        )
        # Compute the per-shard sample indices once so we can log + record them.
        if args.num_sample_shards < 1:
            raise ValueError("--num-sample-shards must be >= 1")
        if not (0 <= args.sample_shard < args.num_sample_shards):
            raise ValueError(
                f"--sample-shard must be in [0, {args.num_sample_shards}); "
                f"got {args.sample_shard}"
            )
        shard_sample_indices = [
            i for i in range(1, 1 + args.num_samples)
            if (i - 1) % args.num_sample_shards == args.sample_shard
        ]
        logger.info(
            "Sample shard %d/%d: processing %d/%d samples (indices first/last = %s/%s)",
            args.sample_shard, args.num_sample_shards,
            len(shard_sample_indices), args.num_samples,
            shard_sample_indices[0] if shard_sample_indices else None,
            shard_sample_indices[-1] if shard_sample_indices else None,
        )
        for idx in shard_sample_indices:
            entry = ds[idx]
            try:
                text = _entry_text(entry).strip()
                if not text:
                    raise ValueError("empty text")
                words = [w for w in text.split() if w.strip()]
                if not words:
                    raise ValueError("no words after split")

                set_seed(args.seed + idx)
                item = prepare_single_item(
                    model, text, args, voice_prompt, language=model_lang
                )
                full_text_for_tokens = _combine_text(text=text, ref_text=item.ref_text)
                phrase_char_spans, _, missing_in_full = collect_phrase_char_spans(
                    full_text_for_tokens,
                    words,
                    ignore_case=bool(args.phrase_ignore_case),
                )
                phrase_char_spans_fa, _, _missing_in_fa = collect_phrase_char_spans(
                    text, words, ignore_case=bool(args.phrase_ignore_case),
                )
                if not phrase_char_spans:
                    raise ValueError(
                        f"No phrase char span found; missing={missing_in_full!r}"
                    )

                cond_branch = _prepare_decode_branch(
                    model, item, gen_config, branch_type="cond"
                )
                target_start = cond_branch.target_start
                t_len = item.target_len

                style_text = ""
                if gen_config.denoise and item.ref_audio_tokens is not None:
                    style_text += "<|denoise|>"
                lang_str = item.lang if item.lang else "None"
                instruct_str = item.instruct if item.instruct else "None"
                style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
                style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"
                style_token_len = int(
                    model.text_tokenizer(style_text, return_tensors="pt").input_ids.size(-1)
                )
                wrapped_full_text = f"<|text_start|>{full_text_for_tokens}<|text_end|>"
                wrapped_text_token_len = _length_of_tokens(
                    wrapped_full_text, model.text_tokenizer
                )
                text_token_offset = style_token_len
                text_token_end = text_token_offset + wrapped_text_token_len

                phrase_text_masks: dict[str, np.ndarray] = {}
                union_text = np.zeros(wrapped_text_token_len, dtype=bool)
                for word, char_spans in phrase_char_spans.items():
                    ranges = []
                    for cs in char_spans:
                        p_start, p_end = map_phrase_to_text_token_range(
                            full_text_for_tokens, cs, model.text_tokenizer
                        )
                        p_start = max(0, min(wrapped_text_token_len, p_start))
                        p_end = max(p_start + 1, min(wrapped_text_token_len, p_end))
                        ranges.append((p_start, p_end))
                    m = _phrase_text_mask(ranges, wrapped_text_token_len)
                    phrase_text_masks[word] = m
                    union_text |= m
                phrase_text_masks["__all__"] = union_text

                cap = _decode_capture_text_idx(
                    model,
                    item,
                    gen_config,
                    text_token_offset=text_token_offset,
                    text_token_end=text_token_end,
                    target_start=target_start,
                    setups=setups,
                    max_steps=args.max_steps,
                    vnorm_capture=vnorm_capture,
                )
                out_audio = decode_tokens(model, cap["tokens"], item.ref_rms, gen_config)
                if bool(args.save_synthesized_audio):
                    sf.write(
                        out_dir / f"{ds_tag}_idx{idx:04d}.wav",
                        out_audio,
                        model.sampling_rate,
                    )

                pending.append(
                    {
                        "idx": idx,
                        "text": text,
                        "num_words": len(words),
                        "phrase_text_masks": phrase_text_masks,
                        "phrase_char_spans_fa": phrase_char_spans_fa,
                        "per_step_text_idx": cap["per_step_text_idx"],
                        "out_audio": out_audio.astype(np.float32, copy=False),
                        "t_len": int(t_len),
                    }
                )
                if len(pending) % 10 == 0 or len(pending) == args.num_samples:
                    logger.info(
                        "[%s phase1] synthesized %d/%d (skipped %d)",
                        ds_tag,
                        len(pending),
                        args.num_samples,
                        len(sample_failures),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s phase1] sample idx=%d failed: %s", ds_tag, idx, e)
                sample_failures.append({"idx": idx, "phase": "synth", "reason": str(e)})

        if not pending:
            logger.warning("[%s] No samples synthesized; skipping FA + metric.", ds_tag)

        # ---------- Phase 2: batch FA ----------
        # Either one giant batch (fa_batch_size <= 0) or fixed-size mini-batches
        # so progress is visible and a partial failure doesn't cost everything.
        bs = int(args.fa_batch_size)
        if bs <= 0:
            mini_batches = [pending] if pending else []
        else:
            mini_batches = [
                pending[i : i + bs] for i in range(0, len(pending), bs)
            ]
        for b_idx, batch in enumerate(mini_batches):
            samples_for_fa = [
                (
                    (entry["out_audio"], model.sampling_rate),
                    entry["text"],
                    aligner_lang,
                )
                for entry in batch
            ]
            try:
                items_per_sample, errors = _align_qwen3_forced_batch(
                    args=args,
                    samples=samples_for_fa,
                    allow_fallback=bool(args.allow_fa_batch_fallback),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s phase2] batch %d FA fully failed (%s); recording all as failed.",
                    ds_tag, b_idx, e,
                )
                items_per_sample = [[]] * len(batch)
                errors = [str(e)] * len(batch)
            for entry, items, err in zip(batch, items_per_sample, errors):
                entry["raw_aligned_items"] = items
                entry["fa_error"] = err
            logger.info(
                "[%s phase2] batch %d/%d done (%d samples, %d FA errors)",
                ds_tag, b_idx + 1, len(mini_batches), len(batch),
                sum(1 for e in errors if e is not None),
            )

        # ---------- Phase 3: per-sample metric computation ----------
        kept = 0
        for entry in pending:
            idx = entry["idx"]
            try:
                if entry.get("fa_error"):
                    raise ValueError(f"FA error: {entry['fa_error']}")
                raw_items = entry.get("raw_aligned_items") or []
                if not raw_items:
                    raise ValueError("FA returned no items")
                phrase_audio_gt_masks = build_phrase_audio_gt_from_fa(
                    aligned_items=raw_items,
                    phrases=list(entry["phrase_text_masks"].keys() - {"__all__"}),
                    phrase_char_spans_fa=entry["phrase_char_spans_fa"],
                    fa_text=entry["text"],
                    audio_token_len=entry["t_len"],
                    frame_rate=frame_rate,
                )
                metrics = _per_word_metrics_per_step(
                    per_step_text_idx=entry["per_step_text_idx"],
                    phrase_text_masks=entry["phrase_text_masks"],
                    phrase_audio_gt_masks=phrase_audio_gt_masks,
                )
                word_mean = _word_mean_per_step(metrics)
                per_sample_word_mean.append(word_mean)
                if bool(args.save_per_sample) or args.num_sample_shards > 1:
                    per_sample_raw.append(
                        {
                            "idx": idx,
                            "text": entry["text"],
                            "num_words": entry["num_words"],
                            "word_mean_per_step": word_mean,
                        }
                    )
                kept += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s phase3] sample idx=%d failed: %s", ds_tag, idx, e)
                sample_failures.append({"idx": idx, "phase": "metric", "reason": str(e)})
        logger.info(
            "[%s] phase3 done | kept=%d / requested=%d (failures: %d)",
            ds_tag, kept, args.num_samples, len(sample_failures),
        )

        # ---- Aggregate across samples ----
        aggregate: dict[str, dict[str, Any]] = {}
        for setup in setup_names:
            per_metric_utt: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
            traj_matrix: dict[str, list[list[float]]] = {k: [] for k in METRIC_KEYS}
            for sample in per_sample_word_mean:
                if setup not in sample:
                    continue
                for metric in METRIC_KEYS:
                    traj = sample[setup][metric]
                    traj_matrix[metric].append(traj)
                    cleaned = [v for v in traj if not math.isnan(v)]
                    if cleaned:
                        per_metric_utt[metric].append(float(np.mean(cleaned)))
            agg_setup: dict[str, Any] = {}
            for metric, values in per_metric_utt.items():
                arr = np.array(values, dtype=np.float64)
                if arr.size == 0:
                    agg_setup[f"{metric}_mean"] = float("nan")
                    agg_setup[f"{metric}_std"] = float("nan")
                    agg_setup[f"{metric}_p25"] = float("nan")
                    agg_setup[f"{metric}_p50"] = float("nan")
                    agg_setup[f"{metric}_p75"] = float("nan")
                    agg_setup[f"{metric}_n"] = 0
                else:
                    agg_setup[f"{metric}_mean"] = float(arr.mean())
                    agg_setup[f"{metric}_std"] = float(arr.std())
                    agg_setup[f"{metric}_p25"] = float(np.percentile(arr, 25))
                    agg_setup[f"{metric}_p50"] = float(np.percentile(arr, 50))
                    agg_setup[f"{metric}_p75"] = float(np.percentile(arr, 75))
                    agg_setup[f"{metric}_n"] = int(arr.size)
            # per-step utterance-mean trajectory
            for metric, mats in traj_matrix.items():
                if not mats:
                    agg_setup[f"{metric}_per_step_mean"] = []
                    agg_setup[f"{metric}_step_summary"] = _trajectory_summary([])
                    continue
                max_S = max(len(t) for t in mats)
                full = np.full((len(mats), max_S), np.nan)
                for r, t in enumerate(mats):
                    full[r, : len(t)] = t
                per_step_mean = np.nanmean(full, axis=0).tolist()
                agg_setup[f"{metric}_per_step_mean"] = [
                    float(x) if not math.isnan(x) else float("nan")
                    for x in per_step_mean
                ]
                agg_setup[f"{metric}_step_summary"] = _trajectory_summary(
                    agg_setup[f"{metric}_per_step_mean"]
                )
            aggregate[setup] = agg_setup

        per_dataset_results[ds_tag] = {
            "hf_id": hf_id,
            "language": language,
            "ref_idx": 0,
            "ref_text": ref_text,
            "num_eval_requested": args.num_samples,
            "num_eval_completed": kept,
            "num_eval_failed": len(sample_failures),
            "sample_indices_processed": list(shard_sample_indices),
            "sample_failures": sample_failures[:50],
            "per_setup_aggregate": aggregate,
            **(
                {"per_sample_raw": per_sample_raw}
                if (bool(args.save_per_sample) or args.num_sample_shards > 1)
                else {}
            ),
        }

    if vnorm_capture is not None:
        vnorm_capture.remove()

    # ---- Cross-dataset ranking ----
    cross_rank: list[dict[str, Any]] = []
    if len(datasets_to_run) >= 1:
        for setup in setup_names:
            f1_vals: dict[str, float] = {}
            iou_vals: dict[str, float] = {}
            for ds in datasets_to_run:
                agg = per_dataset_results[ds]["per_setup_aggregate"].get(setup, {})
                f1_vals[ds] = float(agg.get("f1_mean", float("nan")))
                iou_vals[ds] = float(agg.get("iou_mean", float("nan")))
            valid_f1 = [v for v in f1_vals.values() if not math.isnan(v)]
            valid_iou = [v for v in iou_vals.values() if not math.isnan(v)]
            cross_rank.append({
                "setup": setup,
                "per_dataset_f1_mean": f1_vals,
                "per_dataset_iou_mean": iou_vals,
                "min_f1_mean": min(valid_f1) if valid_f1 else float("nan"),
                "avg_f1_mean": float(np.mean(valid_f1)) if valid_f1 else float("nan"),
                "min_iou_mean": min(valid_iou) if valid_iou else float("nan"),
                "avg_iou_mean": float(np.mean(valid_iou)) if valid_iou else float("nan"),
            })
        cross_rank.sort(
            key=lambda r: (-r["min_f1_mean"] if not math.isnan(r["min_f1_mean"]) else float("inf"))
        )

    final = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "setups": setup_names,
        "datasets": datasets_to_run,
        "per_dataset": per_dataset_results,
        "ranked_by_min_f1": cross_rank,
    }
    # Dump strict JSON: sanitize NaN/Inf to None, and ask the encoder to
    # *reject* any leftover NaN (allow_nan=False) so a missed-sanitize path
    # surfaces immediately instead of producing a file that other parsers
    # silently refuse.
    import json as _json  # local alias to avoid shadowing
    sanitized = _sanitize_for_json(final)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as _f:
        _json.dump(
            sanitized,
            _f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    logger.info("Saved %s", out_dir / "metrics.json")

    # ---- Human-readable summary ----
    csv_lines = [
        "setup,"
        + ",".join(
            f"{ds}_{m}_mean" for ds in datasets_to_run for m in METRIC_KEYS
        )
    ]
    for setup in setup_names:
        row = [setup]
        for ds in datasets_to_run:
            agg = per_dataset_results[ds]["per_setup_aggregate"].get(setup, {})
            for m in METRIC_KEYS:
                v = agg.get(f"{m}_mean", float("nan"))
                # Empty cell for NaN — standard CSV convention; pandas /
                # spreadsheet apps treat empty as missing automatically.
                row.append(f"{v:.4f}" if (v is not None and not math.isnan(v)) else "")
        csv_lines.append(",".join(row))
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")
    logger.info("Saved %s", out_dir / "summary.csv")


if __name__ == "__main__":
    main()

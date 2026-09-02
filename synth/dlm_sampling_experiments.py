#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""LCG sampling engine for the Phrase-Localized LCG benchmark.

CLI subcommand ``cs-attention-plc-langdir`` (alias ``cs-attention-plc``)
performs code-switching synthesis with:
  • a runtime attention-derived phrase mask, and
  • a language-direction contrastive guidance term ``λ·(c_phr - c_src)``
    overlaid on standard CFG inside the masked region.

The repository wraps this engine through ``synth/synth.py``, which calls
``run_cs_attention_plc`` directly.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import (
    _combine_text,
    _get_time_steps,
    _gumbel_sample,
    _tokenize_with_nonverbal_tags,
)

logger = logging.getLogger(__name__)

NONVERBAL_TAG_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


@dataclass
class TaskItem:
    text: str
    target_len: int
    lang: Optional[str]
    instruct: Optional[str]
    ref_text: Optional[str]
    ref_audio_tokens: Optional[torch.Tensor]
    ref_rms: Optional[float]


@dataclass(frozen=True)
class DecodeBranchSpec:
    input_ids: torch.Tensor
    audio_mask: torch.Tensor
    target_start: int
    total_len: int


@dataclass
class SampleOutput:
    tokens: torch.Tensor


@dataclass
class ForcedAlignEditSpan:
    opcode_index: int
    source_char_span: tuple[int, int]
    source_token_span: tuple[int, int]
    time_span_sec: tuple[float, float]
    aligned_items: list[dict[str, Any]]


@dataclass(frozen=True)
class TextUnit:
    text: str
    start: int
    end: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float32 if device.startswith("cpu") else torch.float16
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_generation_config(args: argparse.Namespace) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        denoise=args.denoise,
        preprocess_prompt=args.preprocess_prompt,
        postprocess_output=args.postprocess_output,
        audio_chunk_duration=args.audio_chunk_duration,
        audio_chunk_threshold=args.audio_chunk_threshold,
    )


def make_voice_prompt(model: OmniVoice, args: argparse.Namespace):
    if args.ref_audio is None:
        return None
    return model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        preprocess_prompt=args.preprocess_prompt,
    )


def prepare_single_item(
    model: OmniVoice,
    text: str,
    args: argparse.Namespace,
    voice_prompt,
    duration: Optional[float] = None,
    speed: Optional[float] = None,
    language: Optional[str] = None,
) -> TaskItem:
    task = model._preprocess_all(
        text=text,
        language=args.language if language is None else language,
        voice_clone_prompt=voice_prompt,
        ref_audio=None,
        ref_text=None,
        instruct=args.instruct,
        preprocess_prompt=args.preprocess_prompt,
        speed=args.speed if speed is None else speed,
        duration=args.duration if duration is None else duration,
    )
    assert task.batch_size == 1, "Expected single-item task"
    return TaskItem(
        text=task.texts[0],
        target_len=task.target_lens[0],
        lang=task.langs[0],
        instruct=task.instructs[0],
        ref_text=task.ref_texts[0],
        ref_audio_tokens=task.ref_audio_tokens[0],
        ref_rms=task.ref_rms[0],
    )


def _task_with_target_len(item: TaskItem, target_len: int) -> TaskItem:
    return TaskItem(
        text=item.text,
        target_len=target_len,
        lang=item.lang,
        instruct=item.instruct,
        ref_text=item.ref_text,
        ref_audio_tokens=item.ref_audio_tokens,
        ref_rms=item.ref_rms,
    )


def _prepare_decode_branch(
    model: OmniVoice,
    item: TaskItem,
    gen_config: OmniVoiceGenerationConfig,
    branch_type: str,
) -> DecodeBranchSpec:
    prepared = model._prepare_inference_inputs(
        text=item.text,
        num_target_tokens=item.target_len,
        ref_text=item.ref_text,
        ref_audio_tokens=item.ref_audio_tokens,
        lang=item.lang,
        instruct=item.instruct,
        denoise=gen_config.denoise,
    )
    cond_input_ids = prepared["input_ids"]
    cond_audio_mask = prepared["audio_mask"]
    c_len = cond_input_ids.size(2)
    t_len = item.target_len
    if branch_type == "cond":
        return DecodeBranchSpec(
            input_ids=cond_input_ids,
            audio_mask=cond_audio_mask,
            target_start=c_len - t_len,
            total_len=c_len,
        )
    if branch_type == "uncond":
        return DecodeBranchSpec(
            input_ids=cond_input_ids[..., -t_len:],
            audio_mask=cond_audio_mask[..., -t_len:],
            target_start=0,
            total_len=t_len,
        )
    raise ValueError(f"Unsupported branch_type: {branch_type}")


def _alloc_decode_batch(
    model: OmniVoice,
    branches: list[DecodeBranchSpec],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = model.device
    max_len = max(branch.total_len for branch in branches)
    batch_input_ids = torch.full(
        (len(branches), model.config.num_audio_codebook, max_len),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=device,
    )
    batch_audio_mask = torch.zeros((len(branches), max_len), dtype=torch.bool, device=device)
    batch_attention_mask = torch.zeros(
        (len(branches), 1, max_len, max_len),
        dtype=torch.bool,
        device=device,
    )
    for i, branch in enumerate(branches):
        batch_input_ids[i, :, : branch.total_len] = branch.input_ids[0]
        batch_audio_mask[i, : branch.total_len] = branch.audio_mask[0]
        batch_attention_mask[i, :, : branch.total_len, : branch.total_len] = True
        if max_len > branch.total_len:
            diag = torch.arange(branch.total_len, max_len, device=device)
            batch_attention_mask[i, :, diag, diag] = True
    return batch_input_ids, batch_audio_mask, batch_attention_mask


def _filter_top_k(log_probs: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = max(1, math.ceil(ratio * log_probs.shape[-1]))
    val, ind = log_probs.topk(k, dim=-1)
    filtered = torch.full_like(log_probs, float("-inf"))
    filtered.scatter_(-1, ind, val)
    return filtered


def _predict_with_log_probs(
    model: OmniVoice,
    c_logits: torch.Tensor,
    u_logits: torch.Tensor,
    gen_config: OmniVoiceGenerationConfig,
    contrastive_logits: Optional[torch.Tensor] = None,
    contrastive_guidance_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if contrastive_logits is None or contrastive_guidance_scale == 0.0:
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)
    else:
        guided_logits = c_logits
        if gen_config.guidance_scale != 0:
            guided_logits = guided_logits + gen_config.guidance_scale * (c_logits - u_logits)
        guided_logits = guided_logits + contrastive_guidance_scale * (
            c_logits - contrastive_logits
        )
        log_probs = F.log_softmax(guided_logits, dim=-1)

    log_probs[..., model.config.audio_mask_id] = -float("inf")

    if gen_config.class_temperature > 0.0:
        filtered = _filter_top_k(log_probs, ratio=0.1)
        sampled_logits = _gumbel_sample(filtered, gen_config.class_temperature)
        pred_tokens = sampled_logits.argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)

    confidence_scores = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence_scores, log_probs



def _predict_with_log_probs_langdir(
    model: OmniVoice,
    c_src_logits: torch.Tensor,
    c_phr_logits: torch.Tensor,
    u_logits: torch.Tensor,
    gen_config: OmniVoiceGenerationConfig,
    phrase_weight: torch.Tensor,
    lang_strength: float,
    lang_direction_base: str = "src",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lang-direction CFG variant.

    Baseline CFG everywhere uses the source-lang cond branch:

        baseline = c_src + g (c_src - u)

    Phrase positions add an extra ``lang_strength`` * direction term, weighted
    by ``phrase_weight`` (0/1 hard or [0, 1] soft mask). The direction is
    selected by ``lang_direction_base``:

      - "src"    (default, paper formulation):  direction = c_phr - c_src
        Pure language identity push (text & speaker cancel out).
        final = baseline + w * lang_strength * (c_phr - c_src)

      - "uncond" (alternative formulation):     direction = c_phr - u
        Standard-CFG-style push from unconditional toward phrase-lang cond.
        Mixes language direction with extra CFG strength on phrase positions
        (mathematically equivalent to lang_direction_base="src" with an
        additional w·lang_strength·(c_src - u) term).
        final = baseline + w * lang_strength * (c_phr - u)
    """
    g = gen_config.guidance_scale
    if g != 0:
        baseline = c_src_logits + g * (c_src_logits - u_logits)
    else:
        baseline = c_src_logits
    if phrase_weight.dim() == 2:
        w = phrase_weight.unsqueeze(0).unsqueeze(-1)
    elif phrase_weight.dim() == 3:
        w = phrase_weight.unsqueeze(-1)
    else:
        raise ValueError("phrase_weight must have shape (C, T) or (B, C, T)")
    w = w.to(dtype=baseline.dtype, device=baseline.device)
    if lang_direction_base == "uncond":
        direction = c_phr_logits - u_logits
    elif lang_direction_base == "src":
        direction = c_phr_logits - c_src_logits
    else:
        raise ValueError(f"Unknown lang_direction_base: {lang_direction_base!r}")
    mixed_logits = baseline + w * float(lang_strength) * direction
    log_probs = F.log_softmax(mixed_logits, dim=-1)
    log_probs[..., model.config.audio_mask_id] = -float("inf")

    if gen_config.class_temperature > 0.0:
        filtered = _filter_top_k(log_probs, ratio=0.1)
        sampled_logits = _gumbel_sample(filtered, gen_config.class_temperature)
        pred_tokens = sampled_logits.argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)

    confidence_scores = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence_scores, log_probs



# ---------------------------------------------------------------------------
# Attention-derived phrase audio mask (used by cs-attention-plc)
# ---------------------------------------------------------------------------


ATTENTION_SETUPS: dict[str, dict[str, str]] = {
    "last_1__mean__a2t": {
        "direction": "a2t",
        "layer_range": "last_1",
        "head_reduce": "mean",
    },
    "layers_10-15__max__a2t": {
        "direction": "a2t",
        "layer_range": "layers_10-15",
        "head_reduce": "max",
    },
    "layer_3__max__t2a": {
        "direction": "t2a",
        "layer_range": "layer_3",
        "head_reduce": "max",
    },
    # ---- Added after dataset-level probe ----
    "layer_8__max__a2t": {
        "direction": "a2t",
        "layer_range": "layer_8",
        "head_reduce": "max",
    },
    "layer_12__max__a2t": {
        "direction": "a2t",
        "layer_range": "layer_12",
        "head_reduce": "max",
    },
    "layers_set_8-12__mean__a2t": {
        "direction": "a2t",
        "layer_range": "layers_set_8-12",
        "head_reduce": "mean",
    },
    "layers_set_8-12__max__a2t": {
        "direction": "a2t",
        "layer_range": "layers_set_8-12",
        "head_reduce": "max",
    },
    "layers_8-12__max__a2t": {
        "direction": "a2t",
        "layer_range": "layers_8-12",
        "head_reduce": "max",
    },
}


def _layer_slice_for_attn(
    attn: torch.Tensor, layer_range: str
) -> torch.Tensor:
    """Select layers from a (L, H, R, C) attention tensor by symbolic range.

    Supported names: ``last_1``, ``last_4``, ``all``, ``layer_<i>``,
    ``layers_<a>-<b>``.
    """
    L = attn.size(0)
    if layer_range == "all":
        return attn
    # ``last_N`` for any integer N.
    if layer_range.startswith("last_"):
        n = max(1, int(layer_range.split("_", 1)[1]))
        return attn[max(0, L - n) : L]
    # ``layers_set_<i>-<j>-<k>-...`` — non-contiguous index set.
    if layer_range.startswith("layers_set_"):
        rest = layer_range[len("layers_set_"):]
        idxs = [int(x) for x in rest.split("-") if x != ""]
        idxs = sorted({max(0, min(L - 1, i)) for i in idxs})
        index = torch.tensor(idxs, dtype=torch.long, device=attn.device)
        return attn.index_select(0, index)
    if layer_range.startswith("layer_"):
        idx = int(layer_range.split("_", 1)[1])
        idx = max(0, min(L - 1, idx))
        return attn[idx : idx + 1]
    if layer_range.startswith("layers_"):
        a_str, b_str = layer_range.split("_", 1)[1].split("-", 1)
        a = max(0, min(L - 1, int(a_str)))
        b = max(a, min(L - 1, int(b_str)))
        return attn[a : b + 1]
    raise ValueError(f"Unknown layer_range: {layer_range}")


@torch.inference_mode()
def _omnivoice_forward_with_attentions(
    model: OmniVoice,
    input_ids: torch.Tensor,
    audio_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Forward that mirrors ``OmniVoice.forward`` (using the model's own
    ``_prepare_embed_inputs`` and ``audio_heads``) but routes the underlying
    LLM call through ``output_attentions=True``.
    """
    if input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(0)
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    llm_out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = llm_out.last_hidden_state
    batch_size, seq_len, _ = hidden.shape
    logits_flat = model.audio_heads(hidden)
    audio_logits = logits_flat.view(
        batch_size,
        seq_len,
        model.config.num_audio_codebook,
        model.config.audio_vocab_size,
    ).permute(0, 2, 1, 3)
    return audio_logits, llm_out.attentions


def _phrase_audio_mask_from_attention(
    cond_attn: torch.Tensor,
    setup_name: str,
    text_token_offset: int,
    text_token_end: int,
    target_start: int,
    t_len: int,
    phrase_text_mask: np.ndarray,
    *,
    topk: int = 1,
    soft: bool = False,
) -> np.ndarray:
    """Apply the chosen attention setup to the cond-branch attention and
    return a per-audio-token mask.

    Default (``soft=False, topk=1``): bool array; True = the audio token's
    argmax-attended text token is inside a phrase span.

    ``topk > 1`` (still hard): bool array; True = ANY of the audio token's
    top-k attended text tokens lies inside a phrase span. Boundary tokens
    whose attention is split between phrase and base text recover.

    ``soft=True`` (overrides ``topk``): float array in [0, 1] holding the
    probability mass on phrase text tokens: ``p_phr[a] = Σ_{t ∈ phrase} agg[a, t]``.
    Overrides ``topk`` (the full distribution is used). PLC blend functions
    accept this continuous weight directly (see ``_predict_with_log_probs_*``).

    ``cond_attn`` has shape (L, H, S, S). ``phrase_text_mask`` is over the
    wrapped textual region length ``text_token_end - text_token_offset``.
    """
    setup = ATTENTION_SETUPS[setup_name]
    direction = setup["direction"]
    layer_range = setup["layer_range"]
    head_reduce = setup["head_reduce"]

    if direction == "a2t":
        sliced = cond_attn[
            :,
            :,
            target_start : target_start + t_len,
            text_token_offset:text_token_end,
        ].contiguous()  # (L, H, T_audio, T_text)
    elif direction == "t2a":
        sliced = cond_attn[
            :,
            :,
            text_token_offset:text_token_end,
            target_start : target_start + t_len,
        ].contiguous()  # (L, H, T_text, T_audio)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    sub = _layer_slice_for_attn(sliced, layer_range)
    if head_reduce == "sum":
        agg = sub.sum(dim=1).sum(dim=0)
    elif head_reduce == "mean":
        agg = sub.mean(dim=1).mean(dim=0)
    elif head_reduce == "max":
        agg = sub.amax(dim=1).amax(dim=0)
    else:
        raise ValueError(f"Unknown head_reduce: {head_reduce}")

    # Reorient to (T_audio, T_text). For t2a we transpose so the rest of the
    # code path is direction-agnostic.
    if direction == "t2a":
        agg = agg.transpose(0, 1).contiguous()
    # agg now: (T_audio, T_text)

    phrase_text_mask_t = torch.from_numpy(
        phrase_text_mask.astype(np.bool_, copy=False)
    ).to(device=agg.device)

    if soft:
        # Continuous mask: sum of probability mass on phrase text tokens.
        # No need to re-normalize -- `mean`/`max` over heads/layers preserves
        # the row sum of `agg` (when head_reduce='sum' it does not, but for
        # the [0,1]-blend semantics in PLC we clamp to [0,1]).
        weight = agg.to(torch.float32) * phrase_text_mask_t.to(torch.float32)[None, :]
        p_phr = weight.sum(dim=-1)
        p_phr = p_phr.clamp_(0.0, 1.0)
        return p_phr.detach().cpu().numpy().astype(np.float32, copy=False)

    k = max(1, int(topk))
    if k == 1:
        text_idx_per_audio = agg.argmax(dim=-1).detach().to(torch.int64).cpu().numpy()
        return phrase_text_mask[text_idx_per_audio].astype(bool)

    # topk > 1 (hard union): True if ANY of the top-k attended text tokens is in phrase.
    k = min(k, agg.size(-1))
    topk_idx = agg.topk(k, dim=-1).indices.detach().to(torch.int64).cpu().numpy()  # (T_audio, k)
    selected = phrase_text_mask[topk_idx]  # (T_audio, k) bool
    return selected.any(axis=-1).astype(bool)


def _dilate_mask_1d(mask: np.ndarray, margin: int) -> np.ndarray:
    """Symmetric *graded* dilation of the phrase mask over ``margin`` rounds.

    ``margin`` (the paper's ``k``) is the number of expansion ROUNDS, not the
    final radius. Round ``s = 1..k`` widens the current mask by +-s frames, so
    the radii accumulate into the triangular number

        r_k = sum_{s=1..k} s = k (k + 1) / 2

    i.e. the result equals a single max-filter of radius ``r_k``:
    ``out[a] = max_{|d| <= r_k} mask[a + d]``. Concretely k=2 -> +-3 frames and
    the paper-final k=4 -> +-10 frames. This graded schedule is what produced
    every reported number; the ablation grid k in {0, 2, 4} therefore probes
    effective radii {0, 3, 10}.

    Bool mask: logical OR over the neighborhood (binary dilation).
    Float mask: elementwise max over the neighborhood (soft-mask max-filter).
    ``margin <= 0`` returns ``mask`` unchanged.
    """
    if margin <= 0 or mask.size == 0:
        return mask
    is_bool = mask.dtype == np.bool_
    n = mask.shape[0]
    if is_bool:
        out = mask.copy()
        combine = np.logical_or
    else:
        out = mask.astype(np.float32, copy=True)
        combine = np.maximum
    for s in range(1, int(margin) + 1):
        if s >= n:
            break
        shifted_right = np.zeros_like(out)
        shifted_right[s:] = out[:-s] if is_bool else out[:-s]
        shifted_left = np.zeros_like(out)
        shifted_left[:-s] = out[s:] if is_bool else out[s:]
        out = combine(combine(out, shifted_right), shifted_left)
    return out


def _build_phrase_text_mask(
    phrase_token_ranges: list[tuple[int, int]],
    text_token_len: int,
) -> np.ndarray:
    """Build a bool mask over the wrapped textual region for the union of
    every phrase's BPE token range."""
    m = np.zeros(int(text_token_len), dtype=bool)
    for r_start, r_end in phrase_token_ranges:
        a = max(0, min(text_token_len, int(r_start)))
        b = max(a, min(text_token_len, int(r_end)))
        m[a:b] = True
    return m


def _build_phrase_audio_gt_from_fa(
    aligned_items,
    phrases: list[str],
    phrase_char_spans_fa: dict[str, list[tuple[int, int]]],
    fa_text: str,
    audio_token_len: int,
    frame_rate: float,
) -> dict[str, np.ndarray]:
    """Build per-phrase audio-token bool masks from forced-aligner output.

    Mirror of ``examples/cs_attention_probe.build_phrase_audio_gt_from_fa``:
    for each phrase, mark every audio token whose time falls inside a FA item
    whose char span overlaps the phrase's char span in ``fa_text``. Returns a
    dict keyed by phrase plus ``__all__`` for the union.
    """
    aligned = _align_items_to_source_chars(fa_text.strip(), list(aligned_items))
    item_ranges: list[tuple[int, int, int, int]] = []
    for item, (c_start, c_end) in aligned:
        t_start = float(_align_item_start(item))
        t_end = float(_align_item_end(item))
        a_start = max(0, int(np.floor(t_start * frame_rate)))
        a_end = min(audio_token_len, int(np.ceil(t_end * frame_rate)))
        if a_end > a_start:
            item_ranges.append((a_start, a_end, int(c_start), int(c_end)))

    masks: dict[str, np.ndarray] = {}
    union = np.zeros(audio_token_len, dtype=bool)
    for phrase in phrases:
        m = np.zeros(audio_token_len, dtype=bool)
        for pc_start, pc_end in phrase_char_spans_fa.get(phrase, []):
            for a_start, a_end, c_start, c_end in item_ranges:
                if c_end <= pc_start or c_start >= pc_end:
                    continue
                m[a_start:a_end] = True
        masks[phrase] = m
        union |= m
    masks["__all__"] = union
    return masks


def _phrase_mask_set_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> dict[str, float]:
    """Scalar IoU / Recall / Precision / F1 between two 1-D bool masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    pred_n = int(pred_b.sum())
    gt_n = int(gt_b.sum())
    iou = inter / union if union > 0 else float("nan")
    recall = inter / gt_n if gt_n > 0 else float("nan")
    precision = inter / pred_n if pred_n > 0 else float("nan")
    if (
        not math.isnan(recall)
        and not math.isnan(precision)
        and (recall + precision) > 0
    ):
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = float("nan")
    return {"iou": iou, "recall": recall, "precision": precision, "f1": f1}


def _length_of_tokens_for_text(text: str, tokenizer) -> int:
    return int(_tokenize_with_nonverbal_tags(text, tokenizer).size(-1))


def _map_phrase_char_span_to_token_range(
    full_text_for_tokens: str,
    phrase_char_span: tuple[int, int],
    tokenizer,
) -> tuple[int, int]:
    """Map a phrase's char span inside ``full_text_for_tokens`` (= ref + ' ' + text)
    to a token range inside the wrapped sequence
    ``<|text_start|>{full_text_for_tokens}<|text_end|>``.

    The returned indices are measured from the start of the wrapped token
    sequence, so they line up with the textual region in cond-branch
    input_ids.
    """
    s, e = phrase_char_span
    s = max(0, min(len(full_text_for_tokens), s))
    e = max(s, min(len(full_text_for_tokens), e))
    pre_wrapped = f"<|text_start|>{full_text_for_tokens[:s]}"
    incl_wrapped = f"<|text_start|>{full_text_for_tokens[:e]}"
    p_start = _length_of_tokens_for_text(pre_wrapped, tokenizer)
    p_end = _length_of_tokens_for_text(incl_wrapped, tokenizer)
    if p_end <= p_start:
        p_end = p_start + 1
    return p_start, p_end


def _build_schedule(
    total_mask: int, num_step: int, t_shift: float
) -> list[int]:
    if total_mask <= 0:
        return [0 for _ in range(num_step)]
    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step + 1,
        t_shift=t_shift,
    ).tolist()
    rem = total_mask
    sched: list[int] = []
    for step in range(num_step):
        if step == num_step - 1:
            num = rem
        else:
            num = min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        sched.append(int(num))
        rem -= int(num)
    return sched


@torch.inference_mode()
def iterative_decode(
    model: OmniVoice,
    item: TaskItem,
    gen_config: OmniVoiceGenerationConfig,
) -> SampleOutput:
    """Vanilla iterative masked-diffusion decode (CFG over cond + uncond).

    Used by ``run_cs_attention_plc`` to produce the matrix-only
    ``baseline_src_only.wav`` companion. The LCG path itself goes through
    ``iterative_decode_with_attention_plc``.
    """
    device = model.device
    mask_id = model.config.audio_mask_id
    num_codebook = model.config.num_audio_codebook
    t_len = item.target_len

    sample_tokens = torch.full(
        (1, num_codebook, t_len), mask_id, dtype=torch.long, device=device,
    )

    cond_branch = _prepare_decode_branch(model, item, gen_config, branch_type="cond")
    uncond_branch = _prepare_decode_branch(model, item, gen_config, branch_type="uncond")
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

    for step in range(gen_config.num_step):
        if int((sample_tokens == mask_id).sum().item()) == 0:
            break
        logits = model(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)
        c_logits = logits[0:1, :,
                          cond_branch.target_start : cond_branch.target_start + t_len, :]
        u_logits = logits[1:2, :,
                          uncond_branch.target_start : uncond_branch.target_start + t_len, :]
        pred_tokens, confidence, _ = _predict_with_log_probs(
            model, c_logits, u_logits, gen_config,
        )

        base_scores = confidence - (layer_ids * gen_config.layer_penalty_factor)
        if gen_config.position_temperature > 0.0:
            base_scores = _gumbel_sample(base_scores, gen_config.position_temperature)
        mask_positions = sample_tokens == mask_id
        selection_scores = base_scores.masked_fill(~mask_positions, -float("inf"))
        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(selection_scores.flatten(), k)
            flat_sample = sample_tokens.flatten()
            flat_sample[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens = flat_sample.view_as(sample_tokens)

        for idx, branch in enumerate(branches):
            start = branch.target_start
            batch_input_ids[idx, :, start : start + t_len] = sample_tokens[0]

    return SampleOutput(tokens=sample_tokens.squeeze(0).detach().cpu())


@torch.inference_mode()
def decode_tokens(
    model: OmniVoice,
    tokens: torch.Tensor,
    ref_rms: Optional[float],
    gen_config: OmniVoiceGenerationConfig,
) -> np.ndarray:
    return model._decode_and_post_process(tokens=tokens, rms=ref_rms, gen_config=gen_config)


def resize_tokens_time(tokens: torch.Tensor, new_len: int) -> torch.Tensor:
    if tokens.size(1) == new_len:
        return tokens.clone()
    if new_len <= 0:
        raise ValueError("new_len must be > 0")
    idx = torch.linspace(
        0,
        tokens.size(1) - 1,
        steps=new_len,
        device=tokens.device,
    ).round().long().clamp(0, tokens.size(1) - 1)
    return tokens.index_select(1, idx)


def make_span_from_center(total_len: int, center_ratio: float, width_ratio: float) -> tuple[int, int]:
    center = int(round(center_ratio * total_len))
    width = max(1, int(round(width_ratio * total_len)))
    start = max(0, center - width // 2)
    end = min(total_len, start + width)
    if end <= start:
        end = min(total_len, start + 1)
    return start, end


def changed_span_in_target(src_text: str, tgt_text: str) -> Optional[tuple[int, int]]:
    matcher = difflib.SequenceMatcher(a=src_text, b=tgt_text)
    starts: list[int] = []
    ends: list[int] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        starts.append(j1)
        ends.append(j2)
    if not starts:
        return None
    s, e = min(starts), max(ends)
    if e <= s:
        e = min(len(tgt_text), s + 1)
    return s, e


def map_char_span_to_token_span(
    text_len: int,
    token_len: int,
    span: tuple[int, int],
    margin_ratio: float = 0.0,
) -> tuple[int, int]:
    if token_len <= 0:
        return 0, 0
    if text_len <= 0:
        return 0, token_len
    s_char, e_char = span
    s = int(math.floor(s_char / max(1, text_len) * token_len))
    e = int(math.ceil(e_char / max(1, text_len) * token_len))
    s = max(0, min(token_len - 1, s))
    e = max(s + 1, min(token_len, e))
    m = int(round(margin_ratio * token_len))
    s = max(0, s - m)
    e = min(token_len, e + m)
    if e <= s:
        e = min(token_len, s + 1)
    return s, e


def _prefix_char_weights(text: str, estimator) -> list[float]:
    prefix = [0.0]
    get_char_weight = getattr(estimator, "_get_char_weight", None)
    if not callable(get_char_weight):
        for _ in text:
            prefix.append(prefix[-1] + 1.0)
        return prefix

    for ch in text:
        w = float(get_char_weight(ch))
        if w < 0:
            w = 0.0
        prefix.append(prefix[-1] + w)
    return prefix


def _char_index_to_token_index(
    prefix_weights: list[float], idx: int, token_len: int
) -> int:
    if token_len <= 0:
        return 0
    n_char = len(prefix_weights) - 1
    idx = max(0, min(n_char, idx))
    total = prefix_weights[-1]
    if total <= 1e-8:
        ratio = idx / max(1, n_char)
    else:
        ratio = prefix_weights[idx] / total
    pos = int(round(ratio * token_len))
    return max(0, min(token_len, pos))


def build_voice_prompt(
    model: OmniVoice,
    ref_audio: str,
    ref_text: Optional[str],
    preprocess_prompt: bool,
):
    return model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        preprocess_prompt=preprocess_prompt,
    )


def _normalize_text_for_metric(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-z\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


def char_error_rate(ref: str, hyp: str) -> float:
    ref_n = _normalize_text_for_metric(ref)
    hyp_n = _normalize_text_for_metric(hyp)
    if len(ref_n) == 0:
        return float("nan")
    dist = _levenshtein_distance(ref_n, hyp_n)
    return float(dist / len(ref_n))


def safe_transcribe_audio(model: OmniVoice, audio: np.ndarray) -> tuple[Optional[str], Optional[str]]:
    try:
        wav = audio.astype(np.float32, copy=False)
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]
        text = model.transcribe((wav, model.sampling_rate))
        return text, None
    except Exception as e:
        return None, str(e)


def find_all_phrase_spans(
    text: str, phrase: str, ignore_case: bool = True
) -> list[tuple[int, int]]:
    q = phrase.strip()
    if len(q) == 0:
        return []
    src = text.lower() if ignore_case else text
    tgt = q.lower() if ignore_case else q
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = src.find(tgt, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(tgt)))
        start = idx + len(tgt)
    return spans


def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = m.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans_sorted[0]]
    for s, e in spans_sorted[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def parse_delimited_list(value: Optional[str], delimiter: str = "||") -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(delimiter) if len(x.strip()) > 0]


def unique_preserve_order(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def collect_phrase_char_spans(
    text: str,
    phrases: list[str],
    ignore_case: bool = True,
) -> tuple[dict[str, list[tuple[int, int]]], list[tuple[int, int]], list[str]]:
    per_phrase: dict[str, list[tuple[int, int]]] = {}
    missing: list[str] = []
    merged_all: list[tuple[int, int]] = []
    for phrase in phrases:
        spans = find_all_phrase_spans(text, phrase, ignore_case=ignore_case)
        if spans:
            per_phrase[phrase] = spans
            merged_all.extend(spans)
        else:
            per_phrase[phrase] = []
            missing.append(phrase)
    return per_phrase, merge_spans(merged_all), missing


def build_soft_phrase_weight(
    text: str,
    token_len: int,
    num_codebook: int,
    char_spans: list[tuple[int, int]],
    estimator,
    boundary_tokens: float = 2.0,
    margin_tokens: int = 0,
    phrase_target_lens: Optional[list[int]] = None,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Build a soft (sigmoid) phrase weight envelope over the audio token canvas.

    Returns (weight, expanded_token_spans):
      weight: float tensor of shape (num_codebook, token_len) in [0, 1].
              1.0 -> use phrase-lang condition; 0.0 -> source-lang.
      expanded_token_spans: per-phrase (start, end) tuples after margin expansion.

    Position estimation pipeline:
      1) char_spans -> raw token_spans via `_prefix_char_weights` proportional split.
      2) If `phrase_target_lens` is given, *re-center* each token_span on its
         original midpoint and resize length to the monolingual estimator's prediction
         (so we use char-weight only for *position*, monolingual estimator for *length*).
      3) Optional ±margin_tokens expansion.
      4) Sigmoid envelope with boundary_tokens controlling transition softness.
    """
    if token_len <= 0:
        raise ValueError("token_len must be > 0")
    prefix = _prefix_char_weights(text, estimator)
    raw_spans = [
        _weighted_char_span_to_token_span(prefix, token_len, span) for span in char_spans
    ]
    if phrase_target_lens is not None:
        if len(phrase_target_lens) != len(char_spans):
            raise ValueError(
                "phrase_target_lens length must match char_spans length"
            )
        recentered: list[tuple[int, int]] = []
        for (s, e), tgt_len in zip(raw_spans, phrase_target_lens):
            if tgt_len <= 0:
                recentered.append((s, e))
                continue
            mid = (s + e) / 2.0
            half = tgt_len / 2.0
            ns = max(0, int(round(mid - half)))
            ne = min(token_len, int(round(mid + half)))
            ne = max(ns + 1, ne)
            recentered.append((ns, ne))
        raw_spans = recentered
    raw_spans = merge_spans(raw_spans)
    if margin_tokens > 0:
        expanded = [
            (max(0, s - margin_tokens), min(token_len, e + margin_tokens))
            for s, e in raw_spans
        ]
        expanded = merge_spans(expanded)
    else:
        expanded = raw_spans

    pos = torch.arange(token_len, dtype=torch.float32)
    weight = torch.zeros(token_len, dtype=torch.float32)
    tau = max(1e-3, float(boundary_tokens))
    for s, e in expanded:
        left = torch.sigmoid((pos - s + 0.5) / tau)
        right = torch.sigmoid((e - 0.5 - pos) / tau)
        weight = torch.maximum(weight, left * right)
    weight = weight.unsqueeze(0).expand(num_codebook, -1).contiguous()
    return weight, expanded


def build_phrase_region_masks_weight(
    text: str,
    token_len: int,
    num_codebook: int,
    estimator,
    phrase_char_spans: dict[str, list[tuple[int, int]]],
    margin_ratio: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, list[list[int]]]]:
    region_masks: dict[str, torch.Tensor] = {}
    token_spans_payload: dict[str, list[list[int]]] = {}
    merged_all: list[tuple[int, int]] = []
    for phrase, spans in phrase_char_spans.items():
        if not spans:
            continue
        region_mask, token_spans, _expanded = build_local_edit_mask_from_char_spans(
            text=text,
            token_len=token_len,
            num_codebook=num_codebook,
            char_spans=spans,
            estimator=estimator,
            margin_ratio=margin_ratio,
        )
        region_masks[phrase] = region_mask
        token_spans_payload[phrase] = [list(x) for x in token_spans]
        merged_all.extend(spans)

    if merged_all:
        region_mask, token_spans, _expanded = build_local_edit_mask_from_char_spans(
            text=text,
            token_len=token_len,
            num_codebook=num_codebook,
            char_spans=merge_spans(merged_all),
            estimator=estimator,
            margin_ratio=margin_ratio,
        )
        region_masks["__all_phrases__"] = region_mask
        token_spans_payload["__all_phrases__"] = [list(x) for x in token_spans]

    return region_masks, token_spans_payload


def build_phrase_region_masks_forced(
    aligned_items,
    source_text: str,
    source_token_len: int,
    output_token_len: int,
    num_codebook: int,
    frame_rate: float,
    margin_sec: float,
    phrase_char_spans: dict[str, list[tuple[int, int]]],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, list[list[int]]],
    dict[str, list[dict[str, Any]]],
]:
    region_masks: dict[str, torch.Tensor] = {}
    token_spans_payload: dict[str, list[list[int]]] = {}
    detail_payload: dict[str, list[dict[str, Any]]] = {}
    merged_all: list[tuple[int, int]] = []

    for phrase, spans in phrase_char_spans.items():
        if not spans:
            continue
        region_mask, token_spans, details = build_forced_aligner_mask_from_char_spans(
            aligned_items=aligned_items,
            source_text=source_text,
            char_spans=spans,
            source_token_len=source_token_len,
            output_token_len=output_token_len,
            num_codebook=num_codebook,
            frame_rate=frame_rate,
            margin_sec=margin_sec,
        )
        region_masks[phrase] = region_mask
        token_spans_payload[phrase] = [list(x) for x in token_spans]
        detail_payload[phrase] = [asdict(x) for x in details]
        merged_all.extend(spans)

    if merged_all:
        region_mask, token_spans, details = build_forced_aligner_mask_from_char_spans(
            aligned_items=aligned_items,
            source_text=source_text,
            char_spans=merge_spans(merged_all),
            source_token_len=source_token_len,
            output_token_len=output_token_len,
            num_codebook=num_codebook,
            frame_rate=frame_rate,
            margin_sec=margin_sec,
        )
        region_masks["__all_phrases__"] = region_mask
        token_spans_payload["__all_phrases__"] = [list(x) for x in token_spans]
        detail_payload["__all_phrases__"] = [asdict(x) for x in details]

    return region_masks, token_spans_payload, detail_payload


def build_region_masks_from_token_spans(
    token_len: int,
    num_codebook: int,
    spans_by_name: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[list[int]]]]:
    region_masks: dict[str, torch.Tensor] = {}
    payload: dict[str, list[list[int]]] = {}
    merged_all: list[tuple[int, int]] = []
    for name, spans in spans_by_name.items():
        valid_spans = merge_spans(spans)
        if not valid_spans:
            continue
        region_mask = torch.zeros((num_codebook, token_len), dtype=torch.bool)
        for s, e in valid_spans:
            region_mask[:, s:e] = True
        region_masks[name] = region_mask
        payload[name] = [list(x) for x in valid_spans]
        merged_all.extend(valid_spans)
    if merged_all:
        merged_all = merge_spans(merged_all)
        region_mask = torch.zeros((num_codebook, token_len), dtype=torch.bool)
        for s, e in merged_all:
            region_mask[:, s:e] = True
        region_masks["__all_phrases__"] = region_mask
        payload["__all_phrases__"] = [list(x) for x in merged_all]
    return region_masks, payload


def _is_latin_char(ch: str) -> bool:
    code = ord(ch)
    return (
        (0x0041 <= code <= 0x005A)
        or (0x0061 <= code <= 0x007A)
        or (0x00C0 <= code <= 0x024F)
        or (0x1E00 <= code <= 0x1EFF)
        or (0xFF21 <= code <= 0xFF3A)
        or (0xFF41 <= code <= 0xFF5A)
    )


def _is_hangul_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
        or 0xAC00 <= code <= 0xD7AF
        or 0xD7B0 <= code <= 0xD7FF
    )


LATIN_SCRIPT_LANGUAGE_IDS = {
    "de",
    "dut",
    "en",
    "es",
    "fr",
    "it",
    "nl",
    "pt",
    "ro",
}


def _char_matches_language_script(ch: str, language: Optional[str]) -> bool:
    if language is None:
        return False
    lang = str(language).strip().lower()
    if len(lang) == 0:
        return False
    if lang == "ko":
        return _is_hangul_char(ch)
    if lang in LATIN_SCRIPT_LANGUAGE_IDS:
        return _is_latin_char(ch)
    return False


def find_language_script_spans(text: str, language: Optional[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    idx = 0
    joiners = {"'", "-", "’"}
    while idx < len(text):
        ch = text[idx]
        is_match = _char_matches_language_script(ch, language)
        if start is None:
            if is_match:
                start = idx
        else:
            keep_joiner = (
                ch in joiners
                and idx + 1 < len(text)
                and _char_matches_language_script(text[idx + 1], language)
            )
            if (not is_match) and (not keep_joiner):
                spans.append((start, idx))
                start = None
                continue
        idx += 1
    if start is not None:
        spans.append((start, len(text)))
    return spans


def infer_language_from_text_script(
    text: str,
    fallback_language: Optional[str],
) -> Optional[str]:
    has_hangul = any(_is_hangul_char(ch) for ch in text)
    has_latin = any(_is_latin_char(ch) for ch in text)
    if has_hangul and not has_latin:
        return "ko"
    if has_latin and not has_hangul:
        if fallback_language is not None and str(fallback_language).strip().lower() in (
            LATIN_SCRIPT_LANGUAGE_IDS
        ):
            return fallback_language
        return "en"
    return fallback_language


def join_source_segments(segments: list[str]) -> str:
    text = " ".join(seg.strip() for seg in segments if seg and seg.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text.strip()


def remove_char_spans(text: str, spans: list[tuple[int, int]]) -> str:
    merged = merge_spans(spans)
    if not merged:
        return text.strip()
    pieces: list[str] = []
    cursor = 0
    for s, e in merged:
        pieces.append(text[cursor:s])
        cursor = e
    pieces.append(text[cursor:])
    stripped = "".join(pieces)
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"([(\[{])\s+", r"\1", stripped)
    stripped = re.sub(r"\s+([)\]}])", r"\1", stripped)
    return stripped.strip()


def build_cs_variant_suffix(
    edit_language: Optional[str],
    contrastive_language: Optional[str] = None,
    contrastive_scale: float = 0.0,
) -> str:
    def _norm(x: Optional[str]) -> str:
        if x is None or len(str(x).strip()) == 0:
            return "none"
        return re.sub(r"[^0-9a-zA-Z_-]+", "_", str(x)).strip("_") or "unknown"

    suffix = f"lang_{_norm(edit_language)}"
    if contrastive_language is not None and contrastive_scale != 0.0:
        scale_str = f"{float(contrastive_scale):.3f}".rstrip("0").rstrip(".")
        scale_str = scale_str.replace("-", "m").replace(".", "p")
        suffix += f"__cfg_vs_{_norm(contrastive_language)}__scale_{scale_str}"
    return suffix


def _weighted_char_span_to_token_span(
    prefix_weights: list[float],
    token_len: int,
    span: tuple[int, int],
) -> tuple[int, int]:
    s = _char_index_to_token_index(prefix_weights, span[0], token_len)
    e = _char_index_to_token_index(prefix_weights, span[1], token_len)
    s = max(0, min(token_len - 1, s))
    e = max(s + 1, min(token_len, e))
    return s, e


def _text_units_for_alignment_diff(text: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    last_end = 0
    for m in NONVERBAL_TAG_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            units.extend(_text_units_for_alignment_diff_plain(segment, offset=last_end))
        units.append(TextUnit(text=m.group(0).lower(), start=m.start(), end=m.end()))
        last_end = m.end()
    if last_end < len(text):
        units.extend(_text_units_for_alignment_diff_plain(text[last_end:], offset=last_end))
    if units:
        return units
    return [TextUnit(text=ch, start=i, end=i + 1) for i, ch in enumerate(text) if not ch.isspace()]


def _text_units_for_alignment_diff_plain(text: str, offset: int = 0) -> list[TextUnit]:
    units: list[TextUnit] = []
    start: Optional[int] = None
    buf: list[str] = []
    for idx, ch in enumerate(text):
        if ch.isalnum():
            if start is None:
                start = idx
            buf.append(ch.lower())
        else:
            if start is not None:
                units.append(
                    TextUnit(
                        text="".join(buf),
                        start=offset + start,
                        end=offset + idx,
                    )
                )
                start = None
                buf = []
    if start is not None:
        units.append(
            TextUnit(
                text="".join(buf),
                start=offset + start,
                end=offset + len(text),
            )
        )
    return units


def _forced_aligner_opcodes(source_text: str, edited_text: str):
    src_units = _text_units_for_alignment_diff(source_text)
    tgt_units = _text_units_for_alignment_diff(edited_text)
    matcher = difflib.SequenceMatcher(
        a=[u.text for u in src_units],
        b=[u.text for u in tgt_units],
    )
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        src_start = src_units[i1].start if i1 < len(src_units) else len(source_text)
        src_end = src_units[i2 - 1].end if i2 > i1 else src_start
        tgt_start = tgt_units[j1].start if j1 < len(tgt_units) else len(edited_text)
        tgt_end = tgt_units[j2 - 1].end if j2 > j1 else tgt_start
        out.append((tag, src_start, src_end, tgt_start, tgt_end))
    return out


def _estimate_edit_segment_tokens(
    text: str,
    estimator,
    token_per_weight: float,
    frame_rate: float,
    nonverbal_tag_duration_sec: float,
) -> int:
    total_tokens = 0
    last_end = 0
    for m in NONVERBAL_TAG_PATTERN.finditer(text):
        if m.start() > last_end:
            seg = text[last_end : m.start()]
            total_tokens += int(
                round(estimator.calculate_total_weight(seg) * token_per_weight)
            )
        total_tokens += int(round(nonverbal_tag_duration_sec * frame_rate))
        last_end = m.end()
    if last_end < len(text):
        seg = text[last_end:]
        total_tokens += int(round(estimator.calculate_total_weight(seg) * token_per_weight))
    return total_tokens


def _resolve_qwen_aligner_language(language: Optional[str]) -> str:
    if language is None or len(str(language).strip()) == 0:
        raise ValueError(
            "Forced aligner requires a language. Pass --edit-aligner-language "
            "or set --language to one of the Qwen3-ForcedAligner languages."
        )
    raw = str(language).strip()
    by_code = {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
    }
    by_name = {v.lower(): v for v in by_code.values()}
    key = raw.lower().replace("_", "-")
    if key in by_code:
        return by_code[key]
    if key in by_name:
        return by_name[key]
    raise ValueError(
        f"Unsupported Qwen3-ForcedAligner language: {language!r}. "
        "Use one of: Chinese, English, Cantonese, French, German, Italian, "
        "Japanese, Korean, Portuguese, Russian, Spanish."
    )

def _run_qwen3_forced_aligner_subprocess(
    args: argparse.Namespace,
    audio: Any,
    text: str,
    language: str,
) -> list[dict[str, Any]]:
    worker = Path(__file__).with_name("qwen3_forced_aligner_worker.py")
    if not worker.exists():
        raise FileNotFoundError(f"Missing forced aligner worker: {worker}")

    cache_tmp_root = Path.home() / ".cache" / "omnivoice"
    ensure_dir(cache_tmp_root)
    with tempfile.TemporaryDirectory(
        prefix="qwen_align_",
        dir=str(cache_tmp_root),
    ) as tmp:
        tmp_dir = Path(tmp)
        if isinstance(audio, tuple):
            wav_path = tmp_dir / "source.wav"
            wav, sr = audio
            sf.write(wav_path, np.asarray(wav, dtype=np.float32), int(sr))
            audio_payload = str(wav_path)
        else:
            audio_payload = str(audio)

        input_path = tmp_dir / "input.json"
        output_path = tmp_dir / "output.json"
        payload = {
            "audio": audio_payload,
            "text": text,
            "language": language,
            "model": args.edit_aligner_model,
            "device": args.edit_aligner_device,
            "dtype": args.edit_aligner_dtype,
            "attn_implementation": args.edit_aligner_attn_implementation,
        }
        write_json(input_path, payload)

        if args.edit_aligner_python:
            cmd = [
                args.edit_aligner_python,
                str(worker),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        elif args.edit_aligner_conda_env:
            cmd = [
                args.edit_aligner_conda_executable,
                "run",
                "-n",
                args.edit_aligner_conda_env,
                "python",
                str(worker),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        else:
            # Fall back to the current interpreter when neither an explicit
            # python binary nor a conda env is given.
            cmd = [
                sys.executable,
                str(worker),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]

        logger.info("Running Qwen3 forced aligner subprocess: %s", " ".join(cmd))
        timeout = args.edit_aligner_timeout_sec
        env = os.environ.copy()
        if args.edit_aligner_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.edit_aligner_cuda_visible_devices
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            text=True,
            capture_output=True,
            timeout=None if timeout <= 0 else timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Qwen3 forced aligner subprocess failed "
                f"(exit={completed.returncode}).\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        with output_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if "items" not in result or not isinstance(result["items"], list):
            raise RuntimeError(f"Invalid forced aligner worker output: {result}")
        return result["items"]


def _align_qwen3_forced(
    args: argparse.Namespace,
    audio: Any,
    text: str,
    language: str,
):
    return _run_qwen3_forced_aligner_subprocess(
        args=args,
        audio=audio,
        text=text,
        language=language,
    )


def _run_qwen3_forced_aligner_batch_subprocess(
    args: argparse.Namespace,
    samples: list[tuple[Any, str, str]],
) -> tuple[list[list[dict[str, Any]]], list[Optional[str]]]:
    """Run Qwen3 forced alignment for *many* samples with a single subprocess
    call (the worker loads the model exactly once).

    ``samples`` is a list of ``(audio, text, language)`` triples where
    ``audio`` is either a path string or an in-memory tuple ``(wav, sr)`` that
    we'll dump into a tmp wav. Returns ``(items_per_sample, sample_errors)``
    -- both lists are parallel to the input; an entry of ``sample_errors`` is
    ``None`` for success or an error string for that sample.
    """
    worker = Path(__file__).with_name("qwen3_forced_aligner_worker.py")
    if not worker.exists():
        raise FileNotFoundError(f"Missing forced aligner worker: {worker}")
    if not samples:
        return [], []

    cache_tmp_root = Path.home() / ".cache" / "omnivoice"
    ensure_dir(cache_tmp_root)
    with tempfile.TemporaryDirectory(
        prefix="qwen_align_batch_",
        dir=str(cache_tmp_root),
    ) as tmp:
        tmp_dir = Path(tmp)
        samples_payload: list[dict[str, Any]] = []
        for i, (audio, text, language) in enumerate(samples):
            if isinstance(audio, tuple):
                wav_path = tmp_dir / f"sample_{i:04d}.wav"
                wav, sr = audio
                sf.write(wav_path, np.asarray(wav, dtype=np.float32), int(sr))
                audio_payload: Any = str(wav_path)
            else:
                audio_payload = str(audio)
            samples_payload.append(
                {"audio": audio_payload, "text": text, "language": language}
            )

        input_path = tmp_dir / "input.json"
        output_path = tmp_dir / "output.json"
        request = {
            "samples": samples_payload,
            "model": args.edit_aligner_model,
            "device": args.edit_aligner_device,
            "dtype": args.edit_aligner_dtype,
            "attn_implementation": args.edit_aligner_attn_implementation,
        }
        write_json(input_path, request)

        if args.edit_aligner_python:
            cmd = [
                args.edit_aligner_python,
                str(worker),
                "--input", str(input_path),
                "--output", str(output_path),
            ]
        elif args.edit_aligner_conda_env:
            cmd = [
                args.edit_aligner_conda_executable,
                "run", "-n", args.edit_aligner_conda_env,
                "python", str(worker),
                "--input", str(input_path),
                "--output", str(output_path),
            ]
        else:
            # Fall back to the current interpreter when neither an explicit
            # python binary nor a conda env is given.
            cmd = [
                sys.executable,
                str(worker),
                "--input", str(input_path),
                "--output", str(output_path),
            ]

        logger.info(
            "Running Qwen3 batch FA: %d samples (model loaded once)",
            len(samples_payload),
        )
        timeout = args.edit_aligner_timeout_sec
        env = os.environ.copy()
        if args.edit_aligner_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.edit_aligner_cuda_visible_devices
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            text=True,
            capture_output=True,
            timeout=None if timeout <= 0 else timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Qwen3 batch forced aligner failed "
                f"(exit={completed.returncode}).\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        with output_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        items_per = result.get("items_per_sample")
        if not isinstance(items_per, list):
            raise RuntimeError(
                f"Invalid batch FA output (missing 'items_per_sample'): {result}"
            )
        if len(items_per) != len(samples_payload):
            raise RuntimeError(
                f"Batch FA returned {len(items_per)} results but received "
                f"{len(samples_payload)} samples"
            )
        errors = result.get("sample_errors") or [None] * len(items_per)
        logger.info(
            "Batch FA done | model_load=%.1fs | infer=%.1fs | errors=%d/%d",
            float(result.get("model_load_sec", 0.0)),
            float(result.get("total_infer_sec", 0.0)),
            sum(1 for e in errors if e is not None),
            len(items_per),
        )
        return items_per, errors


def _align_qwen3_forced_batch(
    args: argparse.Namespace,
    samples: list[tuple[Any, str, str]],
    allow_fallback: bool = True,
) -> tuple[list[list[dict[str, Any]]], list[Optional[str]]]:
    """Public batch alignment. If the batch subprocess fails (e.g. a worker
    that doesn't understand the ``samples`` payload), fall back to calling
    the single-sample subprocess once per item — slower but guarantees
    forward progress."""
    try:
        return _run_qwen3_forced_aligner_batch_subprocess(args, samples)
    except Exception as e:
        if not allow_fallback:
            raise
        logger.warning(
            "Batch FA failed (%s); falling back to per-sample single calls "
            "(slow: model reloaded each time)",
            e,
        )
        items_per: list[list[dict[str, Any]]] = []
        errors: list[Optional[str]] = []
        for audio, text, language in samples:
            try:
                single = _run_qwen3_forced_aligner_subprocess(
                    args=args, audio=audio, text=text, language=language
                )
                items_per.append(list(single))
                errors.append(None)
            except Exception as ex:  # noqa: BLE001
                items_per.append([])
                errors.append(f"{type(ex).__name__}: {ex}")
        return items_per, errors


def _align_items_to_dicts(align_items) -> list[dict[str, Any]]:
    return [
        {
            "text": _align_item_text(item),
            "start_time": _align_item_start(item),
            "end_time": _align_item_end(item),
        }
        for item in align_items
    ]


def _align_item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", ""))


def _align_item_start(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("start_time", 0.0))
    return float(getattr(item, "start_time", 0.0))


def _align_item_end(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("end_time", 0.0))
    return float(getattr(item, "end_time", 0.0))


def _normalized_chars_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    prev_space = False
    for idx, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            offsets.append((idx, idx + 1))
            prev_space = False
        elif ch.isspace() and chars and not prev_space:
            chars.append(" ")
            offsets.append((idx, idx + 1))
            prev_space = True
    while chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def _align_items_to_source_chars(
    source_text: str,
    align_items,
) -> list[tuple[Any, tuple[int, int]]]:
    norm_source, offsets = _normalized_chars_with_offsets(source_text)
    out: list[tuple[Any, tuple[int, int]]] = []
    cursor = 0
    fallback_cursor = 0

    for item in align_items:
        item_text = _align_item_text(item)
        norm_item, _ = _normalized_chars_with_offsets(item_text)
        if len(norm_item) == 0:
            out.append((item, (fallback_cursor, fallback_cursor)))
            continue

        pos = norm_source.find(norm_item, cursor)
        if pos < 0:
            pos = norm_source.find(norm_item)
        if pos < 0 or not offsets:
            span = (fallback_cursor, min(len(source_text), fallback_cursor + len(item_text)))
        else:
            end_pos = min(len(offsets) - 1, pos + len(norm_item) - 1)
            span = (offsets[pos][0], offsets[end_pos][1])
            cursor = end_pos + 1
            fallback_cursor = span[1]
        out.append((item, span))
    return out


def _time_to_token_span(
    start_sec: float,
    end_sec: float,
    token_len: int,
    frame_rate: float,
) -> tuple[int, int]:
    if token_len <= 0:
        return 0, 0
    s = int(math.floor(start_sec * frame_rate))
    e = int(math.ceil(end_sec * frame_rate))
    s = max(0, min(token_len - 1, s))
    e = max(s + 1, min(token_len, e))
    return s, e


def build_forced_aligner_edit_spans(
    aligned_items,
    audio: Any,
    source_text: str,
    edited_text: str,
    language: str,
    source_token_len: int,
    frame_rate: float,
    margin_sec: float,
) -> tuple[dict[int, tuple[int, int]], list[ForcedAlignEditSpan]]:
    del audio, language
    aligned = _align_items_to_source_chars(source_text, list(aligned_items))
    opcodes = _forced_aligner_opcodes(source_text, edited_text)
    overrides: dict[int, tuple[int, int]] = {}
    spans: list[ForcedAlignEditSpan] = []
    audio_duration = source_token_len / max(1e-8, frame_rate)

    for op_idx, (tag, i1, i2, _j1, _j2) in enumerate(opcodes):
        if tag == "equal":
            continue

        overlap = [
            (item, char_span)
            for item, char_span in aligned
            if char_span[1] > i1 and char_span[0] < i2
        ]
        prev_items = [(item, span) for item, span in aligned if span[1] <= i1]
        next_items = [(item, span) for item, span in aligned if span[0] >= i2]
        prev_t = _align_item_end(prev_items[-1][0]) if prev_items else 0.0
        next_t = (
            _align_item_start(next_items[0][0])
            if next_items
            else audio_duration
        )

        if overlap:
            start_t = min(_align_item_start(item) for item, _ in overlap)
            end_t = max(_align_item_end(item) for item, _ in overlap)
        else:
            center = (prev_t + next_t) * 0.5
            start_t = center
            end_t = center

        is_insert = i1 == i2
        if is_insert:
            start_t = max(0.0, prev_t - max(0.0, margin_sec))
            end_t = min(audio_duration, next_t + max(0.0, margin_sec))
            if end_t <= start_t:
                center_t = max(0.0, min(audio_duration, (prev_t + next_t) * 0.5))
                center_tok = int(round(center_t * frame_rate))
                center_tok = max(0, min(source_token_len, center_tok))
                start_t = center_t
                end_t = center_t
                token_span = (center_tok, center_tok)
            else:
                token_span = _time_to_token_span(start_t, end_t, source_token_len, frame_rate)
        else:
            start_t = max(0.0, start_t - max(0.0, margin_sec))
            end_t = min(audio_duration, end_t + max(0.0, margin_sec))
            token_span = _time_to_token_span(start_t, end_t, source_token_len, frame_rate)
        overrides[op_idx] = token_span
        spans.append(
            ForcedAlignEditSpan(
                opcode_index=op_idx,
                source_char_span=(i1, i2),
                source_token_span=token_span,
                time_span_sec=(start_t, end_t),
                aligned_items=[
                    {
                        "text": _align_item_text(item),
                        "start_time": _align_item_start(item),
                        "end_time": _align_item_end(item),
                        "char_span": [int(char_span[0]), int(char_span[1])],
                    }
                    for item, char_span in overlap
                ],
            )
        )

    if not overrides:
        raise ValueError("No textual change detected between source text and edited text.")
    return overrides, spans


def build_forced_aligner_mask_from_char_spans(
    aligned_items,
    source_text: str,
    char_spans: list[tuple[int, int]],
    source_token_len: int,
    output_token_len: int,
    num_codebook: int,
    frame_rate: float,
    margin_sec: float,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[ForcedAlignEditSpan]]:
    if source_token_len <= 0 or output_token_len <= 0:
        raise ValueError("token lengths must be > 0")
    if not char_spans:
        raise ValueError("char_spans is empty")

    aligned = _align_items_to_source_chars(source_text, list(aligned_items))
    source_duration = source_token_len / max(1e-8, frame_rate)
    token_spans: list[tuple[int, int]] = []
    details: list[ForcedAlignEditSpan] = []

    for span_idx, (i1, i2) in enumerate(char_spans):
        overlap = [
            (item, char_span)
            for item, char_span in aligned
            if char_span[1] > i1 and char_span[0] < i2
        ]
        if overlap:
            start_t = min(_align_item_start(item) for item, _ in overlap)
            end_t = max(_align_item_end(item) for item, _ in overlap)
        else:
            prev_items = [(item, span) for item, span in aligned if span[1] <= i1]
            next_items = [(item, span) for item, span in aligned if span[0] >= i2]
            prev_t = _align_item_end(prev_items[-1][0]) if prev_items else 0.0
            next_t = (
                _align_item_start(next_items[0][0])
                if next_items
                else source_duration
            )
            start_t = prev_t
            end_t = next_t

        start_t = max(0.0, start_t - max(0.0, margin_sec))
        end_t = min(source_duration, end_t + max(0.0, margin_sec))
        s_src, e_src = _time_to_token_span(start_t, end_t, source_token_len, frame_rate)
        s_out = int(math.floor(s_src / source_token_len * output_token_len))
        e_out = int(math.ceil(e_src / source_token_len * output_token_len))
        s_out = max(0, min(output_token_len - 1, s_out))
        e_out = max(s_out + 1, min(output_token_len, e_out))
        token_spans.append((s_out, e_out))
        details.append(
            ForcedAlignEditSpan(
                opcode_index=span_idx,
                source_char_span=(i1, i2),
                source_token_span=(s_out, e_out),
                time_span_sec=(start_t, end_t),
                aligned_items=[
                    {
                        "text": _align_item_text(item),
                        "start_time": _align_item_start(item),
                        "end_time": _align_item_end(item),
                        "char_span": [int(char_span[0]), int(char_span[1])],
                    }
                    for item, char_span in overlap
                ],
            )
        )

    token_spans = merge_spans(token_spans)
    region_mask = torch.zeros((num_codebook, output_token_len), dtype=torch.bool)
    for s, e in token_spans:
        region_mask[:, s:e] = True
    return region_mask, token_spans, details


def build_local_edit_mask_from_char_spans(
    text: str,
    token_len: int,
    num_codebook: int,
    char_spans: list[tuple[int, int]],
    estimator,
    margin_ratio: float,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    if token_len <= 0:
        raise ValueError("token_len must be > 0")
    if not char_spans:
        raise ValueError("char_spans is empty")

    prefix = _prefix_char_weights(text, estimator)
    token_spans = [
        _weighted_char_span_to_token_span(prefix, token_len, span) for span in char_spans
    ]
    token_spans = merge_spans(token_spans)
    margin = max(0, int(round(margin_ratio * token_len)))
    expanded = []
    for s, e in token_spans:
        expanded.append((max(0, s - margin), min(token_len, e + margin)))
    expanded = merge_spans(expanded)

    region_mask = torch.zeros((num_codebook, token_len), dtype=torch.bool)
    for s, e in expanded:
        region_mask[:, s:e] = True
    return region_mask, token_spans, expanded


def build_edited_canvas_from_diff(
    source_tokens: torch.Tensor,
    source_text: str,
    edited_text: str,
    estimator,
    mask_id: int,
    edit_margin_ratio: float = 0.0,
    source_token_span_overrides: Optional[dict[int, tuple[int, int]]] = None,
    opcodes_override: Optional[list[tuple[str, int, int, int, int]]] = None,
    edit_extra_tokens: int = 0,
    frame_rate: float = 25.0,
    nonverbal_tag_duration_sec: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if source_tokens.ndim != 2:
        raise ValueError("source_tokens should have shape (C, T)")
    if source_tokens.size(1) <= 0:
        raise ValueError("source_tokens is empty")

    src_prefix = _prefix_char_weights(source_text, estimator)
    tgt_prefix = _prefix_char_weights(edited_text, estimator)
    src_total_w = max(1e-8, src_prefix[-1])
    token_per_weight = source_tokens.size(1) / src_total_w

    if opcodes_override is None:
        matcher = difflib.SequenceMatcher(a=source_text, b=edited_text)
        opcodes = matcher.get_opcodes()
    else:
        opcodes = opcodes_override
    if all(tag == "equal" for tag, *_ in opcodes):
        raise ValueError("No textual change detected between source text and edited text.")

    num_codebook = source_tokens.size(0)
    parts_tokens: list[torch.Tensor] = []
    parts_immutable: list[torch.Tensor] = []
    edited_regions: list[tuple[int, int]] = []
    out_cursor = 0
    stats = {
        "num_equal_ops": 0,
        "num_replace_ops": 0,
        "num_insert_ops": 0,
        "num_delete_ops": 0,
        "source_canvas_len": int(source_tokens.size(1)),
        "source_changed_tokens": 0,
        "edited_changed_tokens": 0,
    }

    source_token_span_overrides = source_token_span_overrides or {}
    opcode_token_spans: list[tuple[int, int]] = []
    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if op_idx in source_token_span_overrides:
            s_tok, e_tok = source_token_span_overrides[op_idx]
        else:
            s_tok = _char_index_to_token_index(src_prefix, i1, source_tokens.size(1))
            e_tok = _char_index_to_token_index(src_prefix, i2, source_tokens.size(1))
        s_tok = max(0, min(source_tokens.size(1), s_tok))
        e_tok = max(s_tok, min(source_tokens.size(1), e_tok))
        opcode_token_spans.append((s_tok, e_tok))

    def append_immutable_source(s_tok: int, e_tok: int):
        nonlocal out_cursor
        if e_tok <= s_tok:
            return
        src_len_local = e_tok - s_tok
        parts_tokens.append(source_tokens[:, s_tok:e_tok].clone())
        parts_immutable.append(
            torch.ones((num_codebook, src_len_local), dtype=torch.bool)
        )
        out_cursor += src_len_local

    cursor = 0
    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        raw_s_tok, raw_e_tok = opcode_token_spans[op_idx]

        if tag == "equal":
            stats["num_equal_ops"] += 1
            next_change_start = source_tokens.size(1)
            for next_idx in range(op_idx + 1, len(opcodes)):
                if opcodes[next_idx][0] != "equal":
                    next_change_start = max(cursor, opcode_token_spans[next_idx][0])
                    break
            append_immutable_source(cursor, next_change_start)
            cursor = next_change_start
            continue

        s_tok, e_tok = raw_s_tok, raw_e_tok
        if s_tok > cursor:
            append_immutable_source(cursor, s_tok)
            cursor = s_tok
        if s_tok < cursor:
            s_tok = cursor
        if e_tok < s_tok:
            e_tok = s_tok

        src_len = e_tok - s_tok

        tgt_segment = edited_text[j1:j2]
        if NONVERBAL_TAG_PATTERN.search(tgt_segment):
            tgt_len = _estimate_edit_segment_tokens(
                text=tgt_segment,
                estimator=estimator,
                token_per_weight=token_per_weight,
                frame_rate=frame_rate,
                nonverbal_tag_duration_sec=nonverbal_tag_duration_sec,
            )
        else:
            tgt_weight = tgt_prefix[j2] - tgt_prefix[j1]
            tgt_len = int(round(tgt_weight * token_per_weight))
        if tag in {"insert", "replace"} and j2 > j1 and tgt_len <= 0:
            tgt_len = 1
        if tag in {"insert", "replace"} and op_idx in source_token_span_overrides:
            tgt_len += max(0, int(edit_extra_tokens))

        if tag == "replace":
            stats["num_replace_ops"] += 1
        elif tag == "insert":
            stats["num_insert_ops"] += 1
        elif tag == "delete":
            stats["num_delete_ops"] += 1
        stats["source_changed_tokens"] += int(src_len)

        if tgt_len > 0:
            mask_seg = torch.full(
                (num_codebook, tgt_len),
                fill_value=mask_id,
                dtype=source_tokens.dtype,
            )
            parts_tokens.append(mask_seg)
            parts_immutable.append(torch.zeros((num_codebook, tgt_len), dtype=torch.bool))
            edited_regions.append((out_cursor, out_cursor + tgt_len))
            out_cursor += tgt_len
            stats["edited_changed_tokens"] += int(tgt_len)
        cursor = e_tok

    if cursor < source_tokens.size(1):
        append_immutable_source(cursor, source_tokens.size(1))

    if not parts_tokens:
        # Fallback for corner cases where every changed segment maps to zero length.
        parts_tokens.append(
            torch.full((num_codebook, 1), fill_value=mask_id, dtype=source_tokens.dtype)
        )
        parts_immutable.append(torch.zeros((num_codebook, 1), dtype=torch.bool))
        edited_regions.append((0, 1))
        out_cursor = 1
        stats["edited_changed_tokens"] = 1

    init_tokens = torch.cat(parts_tokens, dim=1)
    immutable = torch.cat(parts_immutable, dim=1)

    if edit_margin_ratio > 0.0 and edited_regions:
        margin = max(1, int(round(edit_margin_ratio * init_tokens.size(1))))
        for s, e in edited_regions:
            ms = max(0, s - margin)
            me = min(init_tokens.size(1), e + margin)
            immutable[:, ms:me] = False
            init_tokens[:, ms:me] = mask_id

    edited_region_mask = ~immutable
    stats["edited_canvas_len"] = int(init_tokens.size(1))
    stats["edited_region_token_count"] = int(edited_region_mask.sum().item())

    return init_tokens, immutable, edited_region_mask, stats


def token_accuracy(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    if a.shape != b.shape:
        raise ValueError("Token tensors must have the same shape")
    if mask is None:
        mask = torch.ones_like(a, dtype=torch.bool)
    valid = int(mask.sum().item())
    if valid == 0:
        return float("nan")
    correct = int(((a == b) & mask).sum().item())
    return correct / valid


def token_change_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    if a.shape != b.shape:
        raise ValueError("Token tensors must have the same shape")
    if mask is None:
        mask = torch.ones_like(a, dtype=torch.bool)
    valid = int(mask.sum().item())
    if valid == 0:
        return float("nan")
    changed = int(((a != b) & mask).sum().item())
    return changed / valid


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError("Mask tensors must have the same shape")
    inter = int((a & b).sum().item())
    union = int((a | b).sum().item())
    if union == 0:
        return float("nan")
    return inter / union


def mask_recall(pred: torch.Tensor, ref: torch.Tensor) -> float:
    if pred.shape != ref.shape:
        raise ValueError("Mask tensors must have the same shape")
    ref_total = int(ref.sum().item())
    if ref_total == 0:
        return float("nan")
    inter = int((pred & ref).sum().item())
    return inter / ref_total


def metric_by_layer(
    a: torch.Tensor,
    b: torch.Tensor,
    metric_fn,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError("Token tensors must have the same shape")
    if a.ndim != 2:
        raise ValueError("Expected token tensors with shape (C, T)")
    if mask is not None and mask.shape != a.shape:
        raise ValueError("mask shape mismatch")
    out: dict[str, float] = {}
    for layer in range(a.size(0)):
        layer_mask = mask[layer : layer + 1] if mask is not None else None
        out[str(layer)] = float(
            metric_fn(
                a[layer : layer + 1],
                b[layer : layer + 1],
                mask=layer_mask,
            )
        )
    return out


def token_accuracy_by_layer(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    return metric_by_layer(a=a, b=b, metric_fn=token_accuracy, mask=mask)


def token_change_ratio_by_layer(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    return metric_by_layer(a=a, b=b, metric_fn=token_change_ratio, mask=mask)



def _save_audio(path: Path, audio: np.ndarray, sr: int):
    sf.write(path, audio, sr)



# ---------------------------------------------------------------------------
# cs-attention-plc{,-langdir}: dynamic phrase mask from runtime attention
# ---------------------------------------------------------------------------


@torch.inference_mode()
def iterative_decode_with_attention_plc(
    model: OmniVoice,
    src_item: TaskItem,
    phr_item: TaskItem,
    gen_config: OmniVoiceGenerationConfig,
    *,
    mode: str,
    attention_setup: str,
    phrase_text_mask: np.ndarray,
    text_token_offset: int,
    text_token_end: int,
    lang_strength: float,
    gt_audio_mask_1d: Optional[np.ndarray] = None,
    phrase_mask_soft: bool = False,
    phrase_mask_topk: int = 1,
    phrase_mask_margin: int = 0,
    phrase_mask_xlang_union: bool = False,
    phrase_mask_full: bool = False,
    lang_direction_base: str = "src",
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """3-branch (c_src, c_phr, u) iterative decoding where the per-step
    phrase audio mask comes from runtime attention (no FA, no char-weight).

    Returns ``(final_tokens, phrase_audio_masks_per_step)``.

    Logit blend:
        everywhere uses ``c_src + g(c_src - u)``, and inside the phrase mask an
        additive ``λ·(c_phr - c_src)`` term is overlaid. The mask is hard by
        default (and in the paper); soft mask is exposed for ablation only.

        final = c_src + g·(c_src - u) + w · λ · (c_phr - c_src)

    Phrase-mask toggles (only used in attention mode; ignored when
    ``gt_audio_mask_1d is not None``):
      - ``phrase_mask_soft``         : if True, return continuous probability
                                       mass mask (float32 in [0,1]); overrides
                                       ``phrase_mask_topk``.
      - ``phrase_mask_topk``         : top-k union over text tokens. k=1
                                       reproduces argmax behavior.
      - ``phrase_mask_margin``       : symmetric dilation of the final per-step
                                       mask (logical_or for bool, max-filter
                                       for float). Applied AFTER xlang union.
      - ``phrase_mask_xlang_union``  : if True, also compute the mask from the
                                       c_phr branch attention and combine
                                       (logical_or for bool, np.maximum for
                                       float). Combination happens BEFORE
                                       ``phrase_mask_margin``.

    Defaults (soft=False, topk=1, margin=0, xlang_union=False) reproduce the
    original byte-identical bool-mask behavior.
    """
    if mode != "langdir":
        raise ValueError(f"Unsupported mode={mode!r}; expected 'langdir'.")
    if attention_setup not in ATTENTION_SETUPS:
        raise ValueError(
            f"Unknown attention_setup={attention_setup}; choose from {list(ATTENTION_SETUPS)}"
        )

    device = model.device
    mask_id = model.config.audio_mask_id
    num_codebook = model.config.num_audio_codebook
    t_len = src_item.target_len
    if int(phr_item.target_len) != t_len:
        phr_item = _task_with_target_len(phr_item, target_len=t_len)

    sample_tokens = torch.full(
        (1, num_codebook, t_len), mask_id, dtype=torch.long, device=device
    )

    c_src_branch = _prepare_decode_branch(model, src_item, gen_config, branch_type="cond")
    c_phr_branch = _prepare_decode_branch(model, phr_item, gen_config, branch_type="cond")
    u_branch = _prepare_decode_branch(model, src_item, gen_config, branch_type="uncond")
    branches = [c_src_branch, c_phr_branch, u_branch]

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

    masks_per_step: list[np.ndarray] = []

    for step in range(gen_config.num_step):
        cur_mask_count = int((sample_tokens == mask_id).sum().item())
        if cur_mask_count == 0:
            break

        audio_logits, attentions = _omnivoice_forward_with_attentions(
            model,
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        )
        audio_logits = audio_logits.to(torch.float32)

        # Extract per-branch logits over the target span.
        c_src_logits = audio_logits[
            0:1, :, c_src_branch.target_start : c_src_branch.target_start + t_len, :
        ]
        c_phr_logits = audio_logits[
            1:2, :, c_phr_branch.target_start : c_phr_branch.target_start + t_len, :
        ]
        u_logits = audio_logits[
            2:3, :, u_branch.target_start : u_branch.target_start + t_len, :
        ]

        # Build phrase audio mask. Either from runtime attention (default) or
        # from a precomputed forced-aligner GT mask (constant across steps).
        if gt_audio_mask_1d is not None:
            # GT mode: the 4 new flags (soft/topk/margin/xlang_union) are NO-OPs.
            phrase_audio_mask = gt_audio_mask_1d.astype(bool, copy=False)
        else:
            stacked = torch.stack(attentions, dim=0)  # (L, B, H, S, S)
            c_src_attn = stacked[:, 0]  # (L, H, S, S)
            phrase_audio_mask = _phrase_audio_mask_from_attention(
                c_src_attn,
                setup_name=attention_setup,
                text_token_offset=text_token_offset,
                text_token_end=text_token_end,
                target_start=c_src_branch.target_start,
                t_len=t_len,
                phrase_text_mask=phrase_text_mask,
                topk=phrase_mask_topk,
                soft=phrase_mask_soft,
            )  # (T_audio,) bool or float32
            if phrase_mask_xlang_union:
                c_phr_attn = stacked[:, 1]  # (L, H, S, S)
                phrase_audio_mask_phr = _phrase_audio_mask_from_attention(
                    c_phr_attn,
                    setup_name=attention_setup,
                    text_token_offset=text_token_offset,
                    text_token_end=text_token_end,
                    target_start=c_phr_branch.target_start,
                    t_len=t_len,
                    phrase_text_mask=phrase_text_mask,
                    topk=phrase_mask_topk,
                    soft=phrase_mask_soft,
                )
                if phrase_audio_mask.dtype == np.bool_:
                    phrase_audio_mask = np.logical_or(
                        phrase_audio_mask, phrase_audio_mask_phr
                    )
                else:
                    phrase_audio_mask = np.maximum(
                        phrase_audio_mask, phrase_audio_mask_phr
                    )
            if phrase_mask_margin > 0:
                phrase_audio_mask = _dilate_mask_1d(
                    phrase_audio_mask, margin=phrase_mask_margin
                )
        # Optional override: force the mask to cover the entire utterance.
        # Useful as an ablation ("what if recall is universal?"). When True,
        # the attention-derived mask + margin + xlang_union are all bypassed.
        if phrase_mask_full:
            phrase_audio_mask = np.ones(t_len, dtype=bool)
        masks_per_step.append(phrase_audio_mask.copy())

        # Broadcast mask to (C, T_audio) so it matches the codebook layout.
        mask_ct = torch.from_numpy(phrase_audio_mask).to(device=device, dtype=torch.float32)
        mask_ct = mask_ct.view(1, t_len).expand(num_codebook, t_len).contiguous()

        pred_tokens, confidence, _ = _predict_with_log_probs_langdir(
            model,
            c_src_logits,
            c_phr_logits,
            u_logits,
            gen_config,
            mask_ct,
            lang_strength,
            lang_direction_base=lang_direction_base,
        )

        # ---- Top-K token freeze (same scheduler as iterative_decode) ----
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

    return sample_tokens.squeeze(0).detach().cpu(), masks_per_step


def _detect_phrase_language_auto(
    phrases: list[str], source_language: Optional[str]
) -> str:
    """Pick a single phrase-language code by scoring per-char Unicode script
    against the source language. Falls back to ``source_language`` when no
    non-source-script char is seen.
    """
    candidate_scores: dict[str, int] = {}
    for ph in phrases:
        for ch in ph:
            if not ch or ch.isspace():
                continue
            cp = ord(ch)
            if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
                candidate_scores["ko"] = candidate_scores.get("ko", 0) + 1
            elif 0x3040 <= cp <= 0x30FF:
                candidate_scores["ja"] = candidate_scores.get("ja", 0) + 1
            elif 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
                candidate_scores["zh"] = candidate_scores.get("zh", 0) + 1
            elif 0x0400 <= cp <= 0x04FF:
                candidate_scores["ru"] = candidate_scores.get("ru", 0) + 1
            elif 0x0600 <= cp <= 0x06FF:
                candidate_scores["ar"] = candidate_scores.get("ar", 0) + 1
            elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                candidate_scores["en"] = candidate_scores.get("en", 0) + 1
    if not candidate_scores:
        return source_language or "en"
    # Prefer a non-source script if available.
    non_src = {k: v for k, v in candidate_scores.items() if k != source_language}
    pool = non_src if non_src else candidate_scores
    return max(pool.items(), key=lambda x: x[1])[0]


def run_cs_attention_plc(
    model: OmniVoice,
    args: argparse.Namespace,
    gen_config: OmniVoiceGenerationConfig,
    voice_prompt,
    out_dir: Path,
):
    """Code-switching synthesis with a runtime-attention-driven phrase mask
    and an additive ``λ·(c_phr - c_src)`` LCG term over the masked region.
    """
    if voice_prompt is None:
        raise ValueError("cs-attention-plc requires --ref-audio")
    if args.text is None or len(args.text.strip()) == 0:
        raise ValueError("--text is required")
    if args.language is None or len(str(args.language).strip()) == 0:
        raise ValueError("--language is required (source language)")

    if args.command not in ("cs-attention-plc-langdir", "cs-attention-plc"):
        raise ValueError(f"Unsupported command={args.command!r}.")
    mode = "langdir"
    setup_name = args.attention_setup
    if setup_name not in ATTENTION_SETUPS:
        raise ValueError(
            f"--attention-setup must be one of {list(ATTENTION_SETUPS)}; got {setup_name!r}"
        )

    insert_phrases = parse_delimited_list(args.insert_phrases)
    if not insert_phrases:
        raise ValueError("--insert-phrases is required (e.g. 'ABC||DEF')")

    # Phrase language: explicit override > auto-detect from phrase script.
    phrase_lang = args.phrase_lang
    if not phrase_lang:
        phrase_lang = _detect_phrase_language_auto(insert_phrases, args.language)
    logger.info(
        "cs-attention-plc | setup=%s | src=%s | phrase-lang=%s",
        setup_name,
        args.language,
        phrase_lang,
    )
    logger.info(
        "cs-attention-plc | flags: soft=%s topk=%d margin=%d xlang_union=%s "
        "full=%s lang_dir_base=%s",
        bool(getattr(args, "phrase_mask_soft", False)),
        int(getattr(args, "phrase_mask_topk", 1)),
        int(getattr(args, "phrase_mask_margin", 0)),
        bool(getattr(args, "phrase_mask_xlang_union", False)),
        bool(getattr(args, "phrase_mask_full", False)),
        str(getattr(args, "lang_direction_base", "src")),
    )

    src_item = prepare_single_item(model, args.text, args, voice_prompt, language=args.language)
    phr_text = args.text_phr if getattr(args, 'text_phr', None) else args.text
    phr_item_base = prepare_single_item(model, phr_text, args, voice_prompt, language=phrase_lang)
    t_len = int(src_item.target_len)
    phr_item = _task_with_target_len(phr_item_base, target_len=t_len)

    # Phrase char spans (in full_text_for_tokens = ref_text + " " + text).
    full_text_for_tokens = _combine_text(text=args.text, ref_text=src_item.ref_text)
    phrase_char_spans, merged_spans, missing_phrases = collect_phrase_char_spans(
        full_text_for_tokens,
        insert_phrases,
        ignore_case=bool(args.phrase_ignore_case),
    )
    if not merged_spans:
        raise ValueError(
            f"No phrases found in combined text. Phrases={insert_phrases!r} text={full_text_for_tokens!r}"
        )
    if missing_phrases:
        logger.warning("Phrases not located in combined ref+text: %s", missing_phrases)

    # Compute cond-branch token layout to align attention slice + phrase mask.
    style_text = ""
    if gen_config.denoise and src_item.ref_audio_tokens is not None:
        style_text += "<|denoise|>"
    lang_str = src_item.lang if src_item.lang else "None"
    instruct_str = src_item.instruct if src_item.instruct else "None"
    style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
    style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"
    style_token_len = int(
        model.text_tokenizer(style_text, return_tensors="pt").input_ids.size(-1)
    )
    wrapped_full_text = f"<|text_start|>{full_text_for_tokens}<|text_end|>"
    wrapped_text_token_len = _length_of_tokens_for_text(wrapped_full_text, model.text_tokenizer)
    text_token_offset = style_token_len
    text_token_end = text_token_offset + wrapped_text_token_len

    # Build the phrase text-token mask (union over all insert phrases).
    phrase_token_ranges: list[tuple[int, int]] = []
    for ph, char_spans in phrase_char_spans.items():
        for cs in char_spans:
            p_start, p_end = _map_phrase_char_span_to_token_range(
                full_text_for_tokens, cs, model.text_tokenizer
            )
            p_start = max(0, min(wrapped_text_token_len, p_start))
            p_end = max(p_start + 1, min(wrapped_text_token_len, p_end))
            phrase_token_ranges.append((p_start, p_end))
    phrase_text_mask = _build_phrase_text_mask(phrase_token_ranges, wrapped_text_token_len)

    # ---- (Optional) Load forced-aligner GT phrase mask ----
    # When --phrase-mask-source=gt, the per-step phrase audio mask is replaced
    # by a precomputed bool mask derived from an external forced-aligner pass.
    # The mask is the SAME at every decoding step; --attention-setup is unused.
    gt_audio_mask_1d: Optional[np.ndarray] = None
    if getattr(args, "phrase_mask_source", "attention") == "gt":
        gt_path_str = getattr(args, "gt_alignment_file", None)
        entry_id = getattr(args, "entry_id", None)
        if not gt_path_str or not entry_id:
            raise ValueError(
                "--phrase-mask-source=gt requires both --gt-alignment-file and --entry-id"
            )
        gt_path = Path(gt_path_str)
        if not gt_path.is_file():
            raise FileNotFoundError(f"--gt-alignment-file not found: {gt_path}")
        target_id = str(entry_id)
        gt_record = None
        with gt_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if str(rec.get("id", "")) == target_id:
                    gt_record = rec
                    break
        if gt_record is None:
            raise ValueError(f"entry_id={target_id!r} not found in {gt_path}")
        aligned_items_gt = gt_record.get("items") or []
        if not aligned_items_gt:
            raise ValueError(f"GT entry {target_id!r} has empty 'items'.")
        phrase_char_spans_fa_local, _, _gt_missing = collect_phrase_char_spans(
            args.text, insert_phrases, ignore_case=bool(args.phrase_ignore_case),
        )
        gt_masks_dict = _build_phrase_audio_gt_from_fa(
            aligned_items=aligned_items_gt,
            phrases=insert_phrases,
            phrase_char_spans_fa=phrase_char_spans_fa_local,
            fa_text=args.text,
            audio_token_len=t_len,
            frame_rate=model.audio_tokenizer.config.frame_rate,
        )
        gt_audio_mask_1d = gt_masks_dict["__all__"].astype(bool, copy=False)
        logger.info(
            "GT mask loaded from %s (entry_id=%s) | phrase-frames=%d/%d",
            gt_path, target_id, int(gt_audio_mask_1d.sum()), t_len,
        )

    # ---- Variant 1: src-only baseline (no phrase CFG, original sampling) ----
    set_seed(args.seed)
    src_only = iterative_decode(model, src_item, gen_config)
    src_audio = decode_tokens(model, src_only.tokens, src_item.ref_rms, gen_config)
    _save_audio(out_dir / "baseline_src_only.wav", src_audio, model.sampling_rate)

    # ---- Variant 2: attention-driven cs-attention-plc-langdir
    # OR GT-driven if gt_audio_mask_1d is not None ----
    set_seed(args.seed)
    final_tokens, masks_per_step = iterative_decode_with_attention_plc(
        model,
        src_item,
        phr_item,
        gen_config,
        mode=mode,
        attention_setup=setup_name,
        phrase_text_mask=phrase_text_mask,
        text_token_offset=text_token_offset,
        text_token_end=text_token_end,
        lang_strength=float(args.lang_cfg_strength),
        gt_audio_mask_1d=gt_audio_mask_1d,
        phrase_mask_soft=bool(getattr(args, "phrase_mask_soft", False)),
        phrase_mask_topk=int(getattr(args, "phrase_mask_topk", 1)),
        phrase_mask_margin=int(getattr(args, "phrase_mask_margin", 0)),
        phrase_mask_xlang_union=bool(getattr(args, "phrase_mask_xlang_union", False)),
        phrase_mask_full=bool(getattr(args, "phrase_mask_full", False)),
        lang_direction_base=str(getattr(args, "lang_direction_base", "src")),
    )
    out_audio = decode_tokens(model, final_tokens, src_item.ref_rms, gen_config)
    paper_lambda = float(args.lang_cfg_strength)
    out_name = f"cs_attention_plc__{setup_name}__lambda{paper_lambda:g}.wav"
    _save_audio(out_dir / out_name, out_audio, model.sampling_rate)

    # ---- Metrics & raw tensors ----
    # Mask dtype follows masks_per_step[0]: bool for hard, float32 for soft.
    def _mask_frame_count(m: np.ndarray) -> int:
        """Number of audio tokens treated as 'in phrase'. For bool masks this
        is `m.sum()`. For float masks we threshold at 0.5 (the same threshold
        used by `_phrase_mask_set_metrics` below) so the count semantics match
        a hard binarization rather than reporting fractional weight totals.
        """
        if m.dtype == np.bool_:
            return int(m.sum())
        return int((m >= 0.5).astype(np.bool_).sum())

    masks_stack_dtype = (
        masks_per_step[0].dtype if masks_per_step else np.bool_
    )
    masks_stack = (
        np.stack(masks_per_step, axis=0)
        if masks_per_step
        else np.zeros((0, t_len), dtype=masks_stack_dtype)
    )
    metrics = {
        "experiment": f"cs_attention_plc_{mode}",
        "text": args.text,
        "text_phr": args.text_phr if getattr(args, 'text_phr', None) else None,
        "phr_text_used": phr_text,
        "source_language": args.language,
        "phrase_language": phrase_lang,
        "phrase_lang_explicit": bool(args.phrase_lang),
        "insert_phrases": insert_phrases,
        "missing_phrases": missing_phrases,
        "phrase_char_spans": {p: [list(x) for x in s] for p, s in phrase_char_spans.items()},
        "phrase_token_ranges": [list(x) for x in phrase_token_ranges],
        "attention_setup": setup_name,
        "lang_cfg_strength_lambda": float(args.lang_cfg_strength),
        "target_len": t_len,
        "duration_sec": float(t_len / max(1e-8, model.audio_tokenizer.config.frame_rate)),
        "captured_steps": int(masks_stack.shape[0]),
        "phrase_audio_token_count_per_step": [_mask_frame_count(m) for m in masks_per_step],
        "mean_phrase_audio_token_count": float(
            np.mean([_mask_frame_count(m) for m in masks_per_step])
        ) if masks_per_step else 0.0,
        "out_wav": out_name,
    }

    # ---- FA alignment evaluation (per-step IoU/Recall/Precision/F1 vs FA GT) ----
    fa_gt_audio_mask: Optional[np.ndarray] = None
    if bool(getattr(args, "eval_with_fa", False)) and masks_per_step:
        try:
            aligner_language = _resolve_qwen_aligner_language(
                args.edit_aligner_language or args.language
            )
            raw_aligned_items = _align_qwen3_forced(
                args=args,
                audio=(out_audio.astype(np.float32, copy=False), model.sampling_rate),
                text=args.text,
                language=aligner_language,
            )
            phrase_char_spans_fa, _fa_merged, fa_missing = collect_phrase_char_spans(
                args.text, insert_phrases, ignore_case=bool(args.phrase_ignore_case),
            )
            gt_masks = _build_phrase_audio_gt_from_fa(
                aligned_items=raw_aligned_items,
                phrases=insert_phrases,
                phrase_char_spans_fa=phrase_char_spans_fa,
                fa_text=args.text,
                audio_token_len=t_len,
                frame_rate=model.audio_tokenizer.config.frame_rate,
            )
            fa_gt_audio_mask = gt_masks["__all__"]
            per_step = [
                _phrase_mask_set_metrics(
                    (m if m.dtype == np.bool_ else (m >= 0.5)).astype(bool),
                    fa_gt_audio_mask,
                )
                for m in masks_per_step
            ]

            def _series(key):
                return [d[key] for d in per_step]

            def _safe_stat(xs, fn):
                arr = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
                return float(fn(arr)) if arr else float("nan")

            ious = _series("iou")
            recalls = _series("recall")
            precisions = _series("precision")
            f1s = _series("f1")
            metrics["fa_alignment_eval"] = {
                "fa_phrase_audio_token_count": {
                    n: int(mm.sum()) for n, mm in gt_masks.items()
                },
                "fa_missing_phrases_in_text": fa_missing,
                "per_step_iou": ious,
                "per_step_recall": recalls,
                "per_step_precision": precisions,
                "per_step_f1": f1s,
                "iou_first": ious[0] if ious else float("nan"),
                "iou_last": ious[-1] if ious else float("nan"),
                "iou_mean": _safe_stat(ious, lambda a: sum(a) / len(a)),
                "iou_max": _safe_stat(ious, max),
                "iou_min": _safe_stat(ious, min),
                "recall_first": recalls[0] if recalls else float("nan"),
                "recall_last": recalls[-1] if recalls else float("nan"),
                "recall_mean": _safe_stat(recalls, lambda a: sum(a) / len(a)),
                "precision_last": precisions[-1] if precisions else float("nan"),
                "f1_last": f1s[-1] if f1s else float("nan"),
                "f1_mean": _safe_stat(f1s, lambda a: sum(a) / len(a)),
            }
            logger.info(
                "FA eval | iou first/last/mean = %.3f / %.3f / %.3f | f1 last = %.3f",
                metrics["fa_alignment_eval"]["iou_first"],
                metrics["fa_alignment_eval"]["iou_last"],
                metrics["fa_alignment_eval"]["iou_mean"],
                metrics["fa_alignment_eval"]["f1_last"],
            )
        except Exception as e:
            logger.warning("FA alignment evaluation failed: %s", e)
            metrics["fa_alignment_eval"] = {"error": str(e)}

    write_json(out_dir / "metrics.json", metrics)
    # Preserve mask dtype: bool for hard masks, float32 for soft masks.
    if masks_stack.dtype == np.bool_:
        masks_stack_to_save = masks_stack.astype(np.bool_)
    else:
        masks_stack_to_save = masks_stack.astype(np.float32)
    save_payload = {
        "tokens_final": final_tokens,
        "phrase_text_mask": torch.from_numpy(phrase_text_mask.astype(np.bool_)),
        "phrase_audio_masks_per_step": torch.from_numpy(masks_stack_to_save),
    }
    if fa_gt_audio_mask is not None:
        save_payload["fa_phrase_audio_gt_mask"] = torch.from_numpy(
            fa_gt_audio_mask.astype(np.bool_)
        )
    torch.save(save_payload, out_dir / "tokens.pt")
    logger.info(
        "Saved %s | captured %d steps | mean phrase audio tokens / step = %.1f",
        out_name,
        metrics["captured_steps"],
        metrics["mean_phrase_audio_token_count"],
    )


@torch.inference_mode()
def compute_final_token_confidence(
    model: OmniVoice,
    item: TaskItem,
    tokens: torch.Tensor,
    gen_config: OmniVoiceGenerationConfig,
) -> float:
    device = model.device
    t_len = item.target_len
    num_codebook = model.config.num_audio_codebook

    prepared = model._prepare_inference_inputs(
        text=item.text,
        num_target_tokens=item.target_len,
        ref_text=item.ref_text,
        ref_audio_tokens=item.ref_audio_tokens,
        lang=item.lang,
        instruct=item.instruct,
        denoise=gen_config.denoise,
    )
    cond_input_ids = prepared["input_ids"]
    cond_audio_mask = prepared["audio_mask"]
    c_len = cond_input_ids.size(2)

    batch_input_ids = torch.full(
        (2, num_codebook, c_len),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=device,
    )
    batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool, device=device)
    batch_attention_mask = torch.zeros(
        (2, 1, c_len, c_len), dtype=torch.bool, device=device
    )
    batch_input_ids[0, :, :c_len] = cond_input_ids[0]
    batch_audio_mask[0, :c_len] = cond_audio_mask[0]
    batch_attention_mask[0, :, :c_len, :c_len] = True
    batch_input_ids[1, :, :t_len] = cond_input_ids[0, :, -t_len:]
    batch_audio_mask[1, :t_len] = cond_audio_mask[0, -t_len:]
    batch_attention_mask[1, :, :t_len, :t_len] = True
    if c_len > t_len:
        diag = torch.arange(t_len, c_len, device=device)
        batch_attention_mask[1, :, diag, diag] = True

    final_tokens = tokens.to(device=device, dtype=torch.long)
    batch_input_ids[0, :, c_len - t_len : c_len] = final_tokens
    batch_input_ids[1, :, :t_len] = final_tokens

    logits = model(
        input_ids=batch_input_ids,
        audio_mask=batch_audio_mask,
        attention_mask=batch_attention_mask,
    ).logits.to(torch.float32)
    c_logits = logits[0:1, :, c_len - t_len : c_len, :]
    u_logits = logits[1:2, :, :t_len, :]
    _, _, log_probs = _predict_with_log_probs(model, c_logits, u_logits, gen_config)
    conf = torch.gather(log_probs, -1, final_tokens.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    return float(conf.mean().item())



def add_shared_model_and_sampling_args(p: argparse.ArgumentParser):
    p.add_argument("--model", type=str, default="k2-fsa/OmniVoice")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "float32", "bfloat16"],
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--language", type=str, default=None)
    p.add_argument("--ref-audio", type=str, default=None)
    p.add_argument("--ref-text", type=str, default=None)
    p.add_argument("--instruct", type=str, default=None)

    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)

    p.add_argument("--num-step", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--t-shift", type=float, default=0.1)
    p.add_argument("--denoise", type=str2bool, default=True)
    p.add_argument("--preprocess-prompt", type=str2bool, default=True)
    p.add_argument("--postprocess-output", type=str2bool, default=True)
    p.add_argument("--layer-penalty-factor", type=float, default=5.0)
    p.add_argument("--position-temperature", type=float, default=5.0)
    p.add_argument("--class-temperature", type=float, default=0.0)
    p.add_argument("--audio-chunk-duration", type=float, default=15.0)
    p.add_argument("--audio-chunk-threshold", type=float, default=30.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sampling-level diffusion LM experiments for OmniVoice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_cs_attention_plc_args(p):
        add_shared_model_and_sampling_args(p)
        p.add_argument("--text", type=str, required=True)
        p.add_argument(
            "--text-phr",
            type=str,
            default=None,
            help="Optional alternate text for the c_phr branch (e.g., loanwords swapped to originalForm). If omitted, --text is used for both branches (current behavior).",
        )
        p.add_argument(
            "--insert-phrases",
            type=str,
            required=True,
            help="Phrases to apply phrase-language CFG to, delimited by '||'.",
        )
        p.add_argument(
            "--phrase-lang",
            type=str,
            default=None,
            help="Override the auto-detected phrase language. If omitted, the "
            "language is detected from the Unicode script of each insert phrase "
            "(Hangul -> ko, CJK -> zh, Hiragana/Katakana -> ja, ...).",
        )
        p.add_argument("--phrase-ignore-case", type=str2bool, default=True)
        p.add_argument(
            "--attention-setup",
            type=str,
            choices=list(ATTENTION_SETUPS.keys()),
            default="last_1__mean__a2t",
            help="Which attention slice/reduction defines the per-step phrase "
            "audio mask. Run separately for each setup to compare. "
            "Ignored when --phrase-mask-source=gt.",
        )
        p.add_argument(
            "--phrase-mask-topk",
            type=int,
            default=1,
            help="When >1, mark an audio token as 'in phrase' if ANY of its top-k "
            "attended text tokens lies inside the phrase span (union over top-k). "
            "k=1 reproduces the default argmax behavior. Ignored when "
            "--phrase-mask-soft=True or --phrase-mask-source=gt.",
        )
        p.add_argument(
            "--phrase-mask-soft",
            type=str2bool,
            default=False,
            help="If True, replace the binary phrase mask with the probability mass "
            "over phrase text tokens (sum_{t in phrase} attn[a,t]) -- a continuous "
            "value in [0,1]. All PLC modes already accept float weights. "
            "Overrides --phrase-mask-topk. Ignored when --phrase-mask-source=gt.",
        )
        p.add_argument(
            "--phrase-mask-margin",
            type=int,
            default=0,
            help="Number of graded symmetric dilation rounds k applied to the "
            "per-step phrase audio mask: round s widens the mask by +-s frames, "
            "giving an effective radius of k(k+1)/2 audio tokens (k=4 -> +-10, "
            "the paper-final setting). Logical OR for binary masks, max-filter "
            "for soft masks. 0 = no dilation (default). Ignored when "
            "--phrase-mask-source=gt.",
        )
        p.add_argument(
            "--phrase-mask-xlang-union",
            type=str2bool,
            default=False,
            help="If True, compute the per-step phrase audio mask from BOTH the "
            "c_src (source-lang tag) AND c_phr (phrase-lang tag) attention branches "
            "and combine them (logical OR for hard, elementwise max for soft). "
            "c_phr attention is already computed every step in the 3-branch CFG "
            "forward pass, so this is effectively free. Ignored when "
            "--phrase-mask-source=gt.",
        )
        p.add_argument(
            "--phrase-mask-full",
            type=str2bool,
            default=False,
            help="If True, override the per-step phrase audio mask with ALL "
            "ones (= apply the language-guiding term everywhere, not just at "
            "the attention-detected phrase positions). Ablation: tests whether "
            "mask localization is necessary. Supersedes --phrase-mask-margin, "
            "--phrase-mask-xlang-union, --phrase-mask-topk, --phrase-mask-soft "
            "(all become moot when the mask is uniformly 1). Ignored when "
            "--phrase-mask-source=gt.",
        )
        p.add_argument(
            "--lang-direction-base",
            type=str,
            choices=["src", "uncond"],
            default="src",
            help="Base for the language-guiding direction. 'src' (default, "
            "paper formulation): direction = c_phr - c_src — pure language "
            "identity push. 'uncond': direction = c_phr - u — standard "
            "CFG-style push from unconditional toward the phrase-lang "
            "conditional (equivalent to 'src' plus an extra "
            "w*lang_strength*(c_src - u) term).",
        )
        p.add_argument(
            "--phrase-mask-source",
            type=str,
            choices=["attention", "gt"],
            default="attention",
            help="Source of the per-step phrase audio mask. "
            "'attention' (default) extracts mask from runtime cross-attention per --attention-setup. "
            "'gt' loads a precomputed mask from --gt-alignment-file (forced-aligner derived) "
            "and uses it unchanged at every decoding step.",
        )
        p.add_argument(
            "--gt-alignment-file",
            type=str,
            default=None,
            help="Path to a JSONL with one entry per utterance: "
            '{"id": "<entry_id>", "items": [{"text", "start_time", "end_time"}, ...]}. '
            "Required when --phrase-mask-source=gt.",
        )
        p.add_argument(
            "--entry-id",
            type=str,
            default=None,
            help="Entry id used to look up the GT alignment item from "
            "--gt-alignment-file. Required when --phrase-mask-source=gt.",
        )
        p.add_argument(
            "--lang-cfg-strength",
            type=float,
            default=2.0,
            help="Lambda for the langdir variant: extra +lambda*(c_phr - c_src) "
            "term on phrase audio tokens. Ignored by the swap variant.",
        )
        p.add_argument(
            "--attn-implementation",
            type=str,
            default="eager",
            help="Underlying LLM attn_implementation. Must be 'eager' for "
            "attention extraction (flash/sdpa return None).",
        )
        # ---- FA alignment evaluation (post-decoding) ----
        p.add_argument(
            "--eval-with-fa",
            type=str2bool,
            default=True,
            help="After decoding, run Qwen3 forced aligner on the output audio "
            "to derive a phrase-audio GT mask and record per-step IoU / Recall / "
            "Precision / F1 between the runtime attention mask and that GT.",
        )
        p.add_argument(
            "--edit-aligner-model",
            type=str,
            default="Qwen/Qwen3-ForcedAligner-0.6B",
        )
        p.add_argument("--edit-aligner-conda-env", type=str, default="qwen3-asr")
        p.add_argument("--edit-aligner-conda-executable", type=str, default="conda")
        p.add_argument("--edit-aligner-python", type=str, default=None)
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
        p.add_argument(
            "--edit-aligner-attn-implementation", type=str, default=None
        )
        p.add_argument("--edit-aligner-margin-sec", type=float, default=0.0)

    p_cs_attn_plc = subparsers.add_parser(
        "cs-attention-plc",
        help="Alias of cs-attention-plc-langdir (additive lang-direction "
        "term). The position-wise cond-swap variant from earlier "
        "experiments is not included in this release.",
    )
    _add_cs_attention_plc_args(p_cs_attn_plc)

    p_cs_attn_plc_lang = subparsers.add_parser(
        "cs-attention-plc-langdir",
        help="Code-switching with runtime-attention-driven phrase mask + "
        "additive lang-direction term (baseline + m*lambda*(c_phr - c_src); option a).",
    )
    _add_cs_attention_plc_args(p_cs_attn_plc_lang)


    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    args = build_parser().parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("examples") / "outputs" / args.command
    )
    ensure_dir(out_dir)
    set_seed(args.seed)

    device = args.device or get_best_device()
    if (
        args.command
        in {
            "text-edit",
            "accent-local-edit",
            "cs-lang-probe",
            "cs-oracle-phrase-cfg",
            "cs-sequential-insert",
            "cs-attention-plc",
            "cs-attention-plc-langdir",
        }
        and getattr(args, "edit_aligner_device", None) is None
    ):
        args.edit_aligner_device = device
    dtype = resolve_dtype(args.dtype, device)
    attn_impl = getattr(args, "attn_implementation", None)
    if args.command in (
        "cs-attention-plc",
        "cs-attention-plc-langdir",
    ) and attn_impl != "eager":
        logger.warning(
            "Forcing attn_implementation='eager' for %s (attention extraction "
            "requires eager; flash/sdpa return None).", args.command
        )
        attn_impl = "eager"
    logger.info(
        "Loading model=%s device=%s dtype=%s attn_impl=%s",
        args.model, device, dtype, attn_impl,
    )
    if attn_impl is not None:
        model = OmniVoice.from_pretrained(
            args.model, device_map=device, dtype=dtype, attn_implementation=attn_impl,
        )
    else:
        model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)

    gen_config = build_generation_config(args)
    voice_prompt = make_voice_prompt(model, args)

    logger.info("Running experiment: %s", args.command)
    if args.command in (
        "cs-attention-plc",
        "cs-attention-plc-langdir",
    ):
        run_cs_attention_plc(model, args, gen_config, voice_prompt, out_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026  Xiaomi Corp.
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

"""Attention-based audio<->text alignment probe POC for OmniVoice.

This probe measures, at every decoding step, how accurately the model's
self-attention encodes a full audio_token -> text_token alignment (relative
to a forced-aligner-derived ground truth). The goal is to identify
(layer-range, head-reduce) setups whose alignment accuracy *improves
iteratively* as decoding progresses -- the target use-case is a Code-Switching
CFG that consumes attention scores produced by the same forward pass that
samples each step.

What it does
============
1. Builds the same conditional input the model uses for inference
   (style tokens + text tokens + ref-audio tokens + masked target-audio tokens).
2. Runs an iterative_decode-style sampling loop (re-using helpers from
   ``examples/dlm_sampling_experiments.py``) and, at every step, captures the
   per-layer / per-head attention map from the conditional branch.
3. For each captured step, takes the attention argmax over the text axis for
   every audio token and turns it into a per-phrase predicted audio-token
   mask (audio_i is "in phrase p" iff its argmax lands inside phrase p's
   text-token range). Compares that mask against a Forced-Aligner-derived
   phrase audio mask, producing IoU / Recall / Precision / F1 per phrase
   (and a union ``__all__``).
4. Reports per-step trajectories for every (layer-range, head-reduce) setup
   and the top-K single heads, plus a trajectory summary capturing first /
   last / min / mean / delta / Spearman trend, with the primary ranking on
   ``recall`` of the ``__all__`` union (false negatives are the costly
   failure mode for downstream phrase-conditioned CFG).

Attention extraction approach
=============================
``OmniVoice.forward`` does NOT forward ``output_attentions`` to its underlying
HuggingFace LLM. Rather than patch the model class, we replicate the small
forward pre-amble (audio-shifted embeddings) inline and call
``model.llm(..., output_attentions=True)`` directly. That returns
``outputs.attentions`` as a tuple of length ``num_layers`` with each tensor of
shape ``(batch, heads, seq, seq)``.

OmniVoice declares ``_supports_flash_attn_2 = True``, which means the default
``attn_implementation`` may dispatch to flash-attn / SDPA - and those
backends usually return ``None`` for attentions even when
``output_attentions=True``. To force eager attention (which fills the tensor
correctly) pass ``--attn-implementation eager`` (the default in this script);
it is plumbed through ``OmniVoice.from_pretrained``.

Value-norm weighting (Kobayashi et al., EMNLP 2020)
===================================================
``--probe-vnorm True`` adds, for every layer-agg setup, a twin
``<layer_range>__<head_reduce>__vnorm`` that multiplies the audio->text
attention block by the key-side value-vector norms ``||v_j||`` before the
layer-slice / head-reduction / argmax. Since attention weights are
non-negative, ``||alpha_ij * v_j|| = alpha_ij * ||v_j||``, so per-key-position
norms suffice. ``--probe-vnormfx True`` additionally adds the full
``||alpha_ij * W_O^h v_j||`` variant (``...__vnormfx``) using per-head column
slices of ``o_proj.weight``. Norms are captured with forward hooks on every
decoder layer's ``self_attn.v_proj`` (see ``_ValueNormCapture``); GQA KV heads
are repeated to match the query-head axis (mirroring ``repeat_kv``).

Phrase token-range mapping
==========================
The text portion of input_ids is built by ``_prepare_inference_inputs`` as

    [style_tokens][text_tokens][ref_audio_tokens][target_audio_tokens]

where ``text_tokens`` wraps ``<|text_start|>{ref_text + " " + text}<|text_end|>``.
We tokenize ``<|text_start|>{prefix_up_to_phrase}`` and
``<|text_start|>{prefix_up_to_phrase}{phrase}`` and take the resulting token
length difference; the phrase token range starts at the style+prefix-token end
and is offset into the full input_ids tensor.

Caveats
=======
- char-weight oracle is only an approximation.
- FA oracle may itself have +-1-2 token offset.

Example
=======
    python analysis/attention_probe.py \
        --text "I really love bibimbap and 반반 치킨 too." \
        --language en \
        --insert-phrases "반반 치킨" \
        --ref-audio examples/en_ref.wav \
        --ref-text "Hello, this is a reference voice." \
        --output-dir analysis/outputs/attention_probe \
        --attn-implementation eager
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
from omnivoice.models.omnivoice import (  # noqa: E402
    _combine_text,
    _get_time_steps,
    _gumbel_sample,
    _tokenize_with_nonverbal_tags,
)

from dlm_sampling_experiments import (  # noqa: E402
    _alloc_decode_batch,
    _align_qwen3_forced,
    _align_items_to_dicts,
    _align_items_to_source_chars,
    _align_item_start,
    _align_item_end,
    _prepare_decode_branch,
    _resolve_qwen_aligner_language,
    add_shared_model_and_sampling_args,
    build_generation_config,
    collect_phrase_char_spans,
    decode_tokens,
    ensure_dir,
    get_best_device,
    iterative_decode,
    make_voice_prompt,
    parse_delimited_list,
    prepare_single_item,
    resolve_dtype,
    set_seed,
    str2bool,
    write_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Attention-based audio<->text alignment probe for OmniVoice's "
            "diffusion-LM TTS decoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_model_and_sampling_args(p)
    p.add_argument("--text", type=str, required=True)
    p.add_argument(
        "--insert-phrases",
        type=str,
        required=True,
        help="Phrases to align against, delimited by '||'.",
    )
    p.add_argument("--phrase-ignore-case", type=str2bool, default=True)

    p.add_argument(
        "--save-attn",
        type=str2bool,
        default=False,
        help="If True, save full per-step attention maps to disk. WARNING: "
        "this can produce many GB of data.",
    )
    p.add_argument(
        "--top-k-heads",
        type=int,
        default=5,
        help="How many top (layer, head) combos to report per ranking "
        "(by recall_last / recall_mean / recall_min / recall_delta).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If >0, only capture attention metrics for the first N decoding "
        "steps (sampling still continues to completion).",
    )
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="HuggingFace attn_implementation for the underlying LLM. Use "
        "'eager' for output_attentions to be populated (sdpa/flash_attention_2 "
        "typically return None for attentions).",
    )

    # FA oracle args (mirror cs-lang-probe's subprocess settings).
    p.add_argument(
        "--edit-aligner-model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    p.add_argument("--edit-aligner-conda-env", type=str, default=None)
    p.add_argument("--edit-aligner-conda-executable", type=str, default="conda")
    p.add_argument(
        "--edit-aligner-python", type=str, default=None,
        help="Path to a python interpreter with Qwen3-ForcedAligner installed. "
             "Falls back to the current interpreter when omitted.",
    )
    p.add_argument(
        "--edit-aligner-cuda-visible-devices", type=str, default=None
    )
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
        "--edit-aligner-attn-implementation",
        type=str,
        default=None,
    )
    p.add_argument(
        "--edit-aligner-margin-sec",
        type=float,
        default=0.0,
        help="Seconds added before/after forced-aligned phrase spans.",
    )
    p.add_argument(
        "--probe-layers",
        type=str,
        default=None,
        help="Optional layer specs delimited by '||', each either a single layer "
        "index ('13') or an inclusive range ('11-13'). For each spec, two extra "
        "head-aggregation setups (sum, mean) are added to the layer_aggregated "
        "ablation. Example: --probe-layers '13||11-13||12'.",
    )
    p.add_argument(
        "--probe-vnorm",
        type=str2bool,
        default=False,
        help="If True, add a value-norm-weighted twin (Kobayashi et al., EMNLP "
        "2020: alpha_ij * ||v_j||) for every layer-agg setup, named "
        "'<layer_range>__<head_reduce>__vnorm'. Value norms are captured via "
        "forward hooks on each decoder layer's self_attn.v_proj.",
    )
    p.add_argument(
        "--probe-vnormfx",
        type=str2bool,
        default=False,
        help="If True, add the full Kobayashi weighting ||alpha_ij * W_O^h v_j|| "
        "twin for every layer-agg setup, named "
        "'<layer_range>__<head_reduce>__vnormfx' (uses per-head o_proj slices).",
    )
    return p


# ---------------------------------------------------------------------------
# OmniVoice forward with attentions
# ---------------------------------------------------------------------------


def _omnivoice_forward_with_attentions(
    model: OmniVoice,
    input_ids: torch.Tensor,
    audio_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
    """Replicate ``OmniVoice.forward`` and request attention weights.

    Returns ``(audio_logits, attentions)`` where ``audio_logits`` has shape
    ``[B, C, S, V]`` (same as ``OmniVoice.forward``) and ``attentions`` is a
    tuple of length ``num_layers`` with each tensor of shape
    ``(batch, heads, seq, seq)``. ``attentions`` is ``None`` if the underlying
    LLM did not return attention weights (e.g. SDPA / flash-attn backends).
    """

    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    llm_outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=True,
    )
    hidden_states = llm_outputs[0]
    attentions = getattr(llm_outputs, "attentions", None)

    batch_size, seq_len, _ = hidden_states.shape
    logits_flat = model.audio_heads(hidden_states)
    audio_logits = logits_flat.view(
        batch_size,
        seq_len,
        model.config.num_audio_codebook,
        model.config.audio_vocab_size,
    ).permute(0, 2, 1, 3)
    return audio_logits, attentions


# ---------------------------------------------------------------------------
# Value-norm capture (Kobayashi et al., EMNLP 2020)
# ---------------------------------------------------------------------------


def _resolve_llm_decoder_layers(model: OmniVoice):
    """Locate the decoder-layer ``ModuleList`` of the underlying LLM.

    ``model.llm`` is normally a bare HF base model (e.g. ``Qwen3Model``) whose
    layers live at ``model.llm.layers``; CausalLM-style wrappers nest them
    under ``model.llm.model.layers``. Try both.
    """
    llm = model.llm
    layers = getattr(llm, "layers", None)
    if layers is None:
        inner = getattr(llm, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise AttributeError(
            "Could not locate decoder layers on model.llm "
            "(tried .layers and .model.layers)."
        )
    return layers


class _ValueNormCapture:
    """Capture per-key-position value-vector norms via ``v_proj`` forward hooks.

    Implements the measurement needed for the value-norm-weighted attention of
    Kobayashi et al. (EMNLP 2020, "Attention is Not Only a Weight"): because
    attention weights are non-negative,
    ``||alpha_ij * v_j|| = alpha_ij * ||v_j||``, so re-weighting an attention
    map only needs the per-position value norms ``||v_j||`` on the key side.

    A forward hook is registered on EVERY decoder layer's ``self_attn.v_proj``
    (all layers are needed for layer-sweep setups). Inside the hook the
    projection output ``(B, S, num_kv_heads * head_dim)`` is reshaped to
    ``(B, S, num_kv_heads, head_dim)``, L2-normed over ``head_dim``,
    transposed to ``(B, num_kv_heads, S)``, and each KV head is repeated
    ``num_attention_heads // num_key_value_heads`` times (mirroring
    ``repeat_kv`` under GQA) so the result aligns with the
    ``(B, num_heads, S, S)`` attention tensors. Norms are stored as float32 on
    CPU; the raw activation is discarded immediately.

    With ``capture_fx=True`` the full Kobayashi weighting
    ``||f(x_j)|| = ||W_O^h v_j||`` is additionally captured (``"vnormfx"``),
    using the per-head column slice ``o_proj.weight[:, h*D:(h+1)*D]`` of each
    layer's output projection.

    Hooks are inert unless ``enable()`` was called;
    ``_omnivoice_forward_with_attentions_and_vnorms`` toggles this around a
    single forward pass. Call ``remove()`` when done.
    """

    def __init__(self, model: OmniVoice, capture_fx: bool = False):
        llm_config = getattr(model.llm, "config", None)
        if llm_config is None:
            llm_config = getattr(model.config, "llm_config", None)
        if llm_config is None:
            raise AttributeError(
                "Could not resolve the LLM config for value-norm capture."
            )
        num_heads = int(llm_config.num_attention_heads)
        num_kv_heads = int(
            getattr(llm_config, "num_key_value_heads", None) or num_heads
        )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads={num_heads} is not divisible by "
                f"num_key_value_heads={num_kv_heads}"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.capture_fx = bool(capture_fx)

        layers = _resolve_llm_decoder_layers(model)
        self.num_layers = len(layers)
        self._enabled = False
        self._norms: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self._norms_fx: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self._o_proj_weights: list[Optional[torch.Tensor]] = (
            [None] * self.num_layers
        )
        self._handles = []
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if self.capture_fx:
                self._o_proj_weights[layer_idx] = attn.o_proj.weight
            self._handles.append(
                attn.v_proj.register_forward_hook(self._make_hook(layer_idx))
            )
        logger.info(
            "Registered value-norm hooks on %d layers (num_heads=%d, "
            "num_kv_heads=%d, kv_groups=%d, capture_fx=%s).",
            self.num_layers,
            num_heads,
            num_kv_heads,
            self.num_key_value_groups,
            self.capture_fx,
        )

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            bsz, seq_len, _ = output.shape
            v = output.detach().to(torch.float32).view(
                bsz, seq_len, self.num_kv_heads, -1
            )  # (B, S, KV, D)
            norms = v.norm(dim=-1).transpose(1, 2)  # (B, KV, S)
            norms = norms.repeat_interleave(self.num_key_value_groups, dim=1)
            self._norms[layer_idx] = norms.cpu()  # (B, H, S)
            if self.capture_fx:
                w = self._o_proj_weights[layer_idx]
                hidden_size, _ = w.shape
                head_dim = v.size(-1)
                # (hidden, H*D) -> (hidden, H, D); w_heads[:, h] == W_O^h.
                w_heads = w.detach().to(torch.float32).view(
                    hidden_size, self.num_heads, head_dim
                )
                # Expand KV heads to query heads (mirrors repeat_kv), then
                # take ||W_O^h v_j|| over the output (hidden) dim.
                v_q = v.repeat_interleave(
                    self.num_key_value_groups, dim=2
                )  # (B, S, H, D)
                fx = torch.einsum("bshd,ohd->bsho", v_q, w_heads)
                self._norms_fx[layer_idx] = (
                    fx.norm(dim=-1).transpose(1, 2).cpu()
                )  # (B, H, S)

        return hook

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset(self):
        self._norms = [None] * self.num_layers
        self._norms_fx = [None] * self.num_layers

    def collect(self) -> dict[str, tuple[torch.Tensor, ...]]:
        """Return captured norms as ``{weighting_name: per-layer tuple}``.

        Each per-layer tensor has shape ``(B, num_heads, S)`` (float32, CPU)
        and is aligned with the ``attentions`` tuple of the same forward pass.
        Keys: ``"vnorm"`` always; ``"vnormfx"`` when ``capture_fx=True``.
        """
        missing = [i for i, n in enumerate(self._norms) if n is None]
        if missing:
            raise RuntimeError(
                f"Value norms missing for layers {missing}; was the capture "
                "enabled during the forward pass?"
            )
        out: dict[str, tuple[torch.Tensor, ...]] = {"vnorm": tuple(self._norms)}
        if self.capture_fx:
            out["vnormfx"] = tuple(self._norms_fx)
        return out

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _omnivoice_forward_with_attentions_and_vnorms(
    model: OmniVoice,
    input_ids: torch.Tensor,
    audio_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    vnorm_capture: Optional[_ValueNormCapture] = None,
) -> tuple[
    torch.Tensor,
    Optional[tuple[torch.Tensor, ...]],
    Optional[dict[str, tuple[torch.Tensor, ...]]],
]:
    """Sibling of ``_omnivoice_forward_with_attentions`` that also captures
    per-layer key-side value norms (see ``_ValueNormCapture``).

    Returns ``(audio_logits, attentions, value_norms)`` where ``value_norms``
    maps each captured weighting name (``"vnorm"`` and, when the capture was
    built with ``capture_fx=True``, ``"vnormfx"``) to a tuple of per-layer
    ``(B, num_heads, S)`` float32 CPU tensors aligned with ``attentions``.
    ``value_norms`` is ``None`` when ``vnorm_capture`` is ``None``.
    """
    if vnorm_capture is None:
        audio_logits, attentions = _omnivoice_forward_with_attentions(
            model, input_ids, audio_mask, attention_mask
        )
        return audio_logits, attentions, None
    vnorm_capture.reset()
    vnorm_capture.enable()
    try:
        audio_logits, attentions = _omnivoice_forward_with_attentions(
            model, input_ids, audio_mask, attention_mask
        )
    finally:
        vnorm_capture.disable()
    return audio_logits, attentions, vnorm_capture.collect()


# ---------------------------------------------------------------------------
# Phrase token-range mapping
# ---------------------------------------------------------------------------


def _length_of_tokens(text: str, tokenizer) -> int:
    """Length of ``_tokenize_with_nonverbal_tags(text)`` (without batch dim)."""
    if len(text) == 0:
        # Match the behavior of _tokenize_with_nonverbal_tags on empty input.
        ids = tokenizer("", return_tensors="pt").input_ids
        return int(ids.size(-1))
    return int(_tokenize_with_nonverbal_tags(text, tokenizer).size(-1))


def map_phrase_to_text_token_range(
    full_text_for_tokens: str,
    phrase_char_span: tuple[int, int],
    tokenizer,
) -> tuple[int, int]:
    """Map a phrase's char span inside ``full_text_for_tokens`` (= ref_text + ' ' + text,
    after ``_combine_text`` cleanups) to a token range inside the wrapped
    sequence ``<|text_start|>{full_text_for_tokens}<|text_end|>``.

    Returns ``(p_start_in_text_tokens, p_end_in_text_tokens)`` measured from
    the start of the wrapped sequence's tokens (i.e. relative to the
    start of ``text_tokens`` in ``_prepare_inference_inputs``).
    """
    s, e = phrase_char_span
    s = max(0, min(len(full_text_for_tokens), s))
    e = max(s, min(len(full_text_for_tokens), e))

    pre = full_text_for_tokens[:s]
    incl = full_text_for_tokens[:e]

    pre_wrapped = f"<|text_start|>{pre}"
    incl_wrapped = f"<|text_start|>{incl}"

    # Length of tokens for the prefix (incl. <|text_start|>).
    p_start = _length_of_tokens(pre_wrapped, tokenizer)
    p_end = _length_of_tokens(incl_wrapped, tokenizer)
    if p_end <= p_start:
        p_end = p_start + 1
    return p_start, p_end


# ---------------------------------------------------------------------------
# Schedule helper (mirrors iterative_decode's behavior for the default path)
# ---------------------------------------------------------------------------


def _build_schedule(total_mask: int, num_step: int, t_shift: float) -> list[int]:
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
            num = min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                rem,
            )
        sched.append(int(num))
        rem -= int(num)
    return sched


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((rx_c * ry_c).sum() / denom)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


# ---------------------------------------------------------------------------
# Full TTS-alignment GT (audio_token -> text_token) from FA
# ---------------------------------------------------------------------------


def build_phrase_audio_gt_from_fa(
    aligned_items,
    phrases: list[str],
    phrase_char_spans_fa: dict[str, list[tuple[int, int]]],
    fa_text: str,
    audio_token_len: int,
    frame_rate: float,
) -> dict[str, np.ndarray]:
    """Build per-phrase audio-token bool masks from forced-aligner output.

    For each phrase, mark every audio token whose time falls inside a FA item
    whose char span overlaps the phrase's char span (in ``fa_text``).

    Returns a dict keyed by phrase string, plus the special key ``__all__``
    holding the union of every phrase mask. Phrases that the FA could not
    align (no overlapping items) map to all-False masks.
    """
    aligned = _align_items_to_source_chars(fa_text.strip(), list(aligned_items))
    # Pre-resolve each FA item's audio-token range once.
    item_ranges: list[tuple[int, int, int, int]] = []  # (a_start, a_end, c_start, c_end)
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
                # char-range overlap check
                if c_end <= pc_start or c_start >= pc_end:
                    continue
                m[a_start:a_end] = True
        masks[phrase] = m
        union |= m
    masks["__all__"] = union
    return masks


LAYER_AGG_SETUPS: list[tuple[str, str]] = [
    ("last_1", "sum"),
    ("last_1", "mean"),
    ("last_1", "max"),
    ("last_4", "sum"),
    ("last_4", "mean"),
    ("last_4", "max"),
    ("all", "sum"),
    ("all", "mean"),
    ("all", "max"),
]
HEAD_REDUCERS: tuple[str, ...] = ("sum", "mean", "max")
METRIC_KEYS: tuple[str, ...] = ("iou", "recall", "precision", "f1")


def _layer_slice(audio_to_text: torch.Tensor, layer_range: str) -> torch.Tensor:
    """Select layers from a (L, H, T_audio, T_text) tensor.

    Supported `layer_range` formats:
      - "last_1", "last_4", "all"           : built-in
      - "layer_13"                          : single layer (custom)
      - "layers_11-13"                      : inclusive range (custom)
    """
    L = audio_to_text.size(0)
    if layer_range == "all":
        return audio_to_text
    # ``last_N`` for any integer N (last_1, last_4, last_8, ...).
    if layer_range.startswith("last_"):
        n = max(1, int(layer_range.split("_", 1)[1]))
        return audio_to_text[max(0, L - n) : L]
    # ``layers_set_<i>-<j>-<k>-...`` — non-contiguous index set, fancy-indexed.
    if layer_range.startswith("layers_set_"):
        rest = layer_range[len("layers_set_"):]
        idxs = [int(x) for x in rest.split("-") if x != ""]
        idxs = sorted({max(0, min(L - 1, i)) for i in idxs})
        index = torch.tensor(idxs, dtype=torch.long, device=audio_to_text.device)
        return audio_to_text.index_select(0, index)
    if layer_range.startswith("layer_"):
        idx = int(layer_range.split("_", 1)[1])
        idx = max(0, min(L - 1, idx))
        return audio_to_text[idx : idx + 1]
    if layer_range.startswith("layers_"):
        rng = layer_range.split("_", 1)[1]
        a_str, b_str = rng.split("-", 1)
        a = max(0, min(L - 1, int(a_str)))
        b = max(a, min(L - 1, int(b_str)))
        return audio_to_text[a : b + 1]
    raise ValueError(f"Unknown layer_range: {layer_range}")


def _probe_layer_spec_to_range_name(spec: str) -> str:
    """Convert a CLI spec ("13", "11-13") to a layer_range name ("layer_13", "layers_11-13")."""
    s = spec.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return f"layers_{int(a.strip())}-{int(b.strip())}"
    return f"layer_{int(s)}"


def _phrase_text_mask(
    phrase_token_ranges: list[tuple[int, int]], text_token_len: int
) -> np.ndarray:
    """Build a 1-D bool mask over the text-token axis covering the phrase's
    token ranges. Shape: (text_token_len,)."""
    m = np.zeros(text_token_len, dtype=bool)
    for r_start, r_end in phrase_token_ranges:
        a = max(0, min(text_token_len, int(r_start)))
        b = max(a, min(text_token_len, int(r_end)))
        m[a:b] = True
    return m


def _binary_set_metrics(
    pred: np.ndarray, gt: np.ndarray, axis: int = -1
) -> dict[str, np.ndarray]:
    """Compute IoU / Recall / Precision / F1 between two bool arrays along
    ``axis``. Shapes must broadcast. Returns dict of float32 arrays with the
    reduced dimension removed.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum(axis=axis).astype(np.float64)
    union = np.logical_or(pred_b, gt_b).sum(axis=axis).astype(np.float64)
    pred_count = pred_b.sum(axis=axis).astype(np.float64)
    gt_count = gt_b.sum(axis=axis).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
        recall = np.where(gt_count > 0, inter / gt_count, np.nan)
        precision = np.where(pred_count > 0, inter / pred_count, np.nan)
        denom = recall + precision
        f1 = np.where(
            (denom > 0) & ~np.isnan(denom),
            2 * recall * precision / np.where(denom == 0, 1.0, denom),
            np.nan,
        )
    return {
        "iou": iou.astype(np.float32),
        "recall": recall.astype(np.float32),
        "precision": precision.astype(np.float32),
        "f1": f1.astype(np.float32),
    }


def compute_phrase_metrics_per_head_for_step(
    audio_to_text: torch.Tensor,
    phrase_text_masks: dict[str, np.ndarray],
    phrase_audio_gt_masks: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Vectorized per-(L, H) phrase-audio IoU/Recall/Precision/F1.

    audio_to_text: (L, H, T_audio, T_text)
    phrase_text_masks[name]: (T_text,) bool
    phrase_audio_gt_masks[name]: (T_audio,) bool

    Returns: dict[phrase_name] -> dict[metric] -> (L, H) float32 array.
    """
    L, H, T_audio, T_text = audio_to_text.shape
    text_argmax = (
        audio_to_text.argmax(dim=-1).detach().to(torch.int64).cpu().numpy()
    )  # (L, H, T_audio)
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, text_mask in phrase_text_masks.items():
        gt_mask = phrase_audio_gt_masks[name]
        # pred[L, H, t] = text_mask[text_argmax[L, H, t]]
        pred = text_mask[text_argmax]  # (L, H, T_audio) bool
        gt_bcast = gt_mask[None, None, :]  # (1, 1, T_audio)
        results[name] = _binary_set_metrics(pred, gt_bcast, axis=-1)
    return results


def compute_phrase_metrics_aggregated_for_step(
    audio_to_text: torch.Tensor,
    phrase_text_masks: dict[str, np.ndarray],
    phrase_audio_gt_masks: dict[str, np.ndarray],
    setups: list[tuple[str, ...]],
    key_value_norms: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-setup phrase-audio metrics. Aggregates the attention map per setup
    (layer-range, head-reduce), takes argmax over the text axis, then computes
    IoU/Recall/Precision/F1 for each phrase (and ``__all__``).

    Setups are ``(layer_range, head_reduce)`` or
    ``(layer_range, head_reduce, weighting)`` tuples. The 3-element form
    multiplies the attention block by the matching key-side value norms from
    ``key_value_norms`` (a ``(L, H, T_text)`` tensor per weighting name, e.g.
    ``"vnorm"`` / ``"vnormfx"``; Kobayashi et al., EMNLP 2020) before the
    layer-slice / head-reduction / argmax. No renormalization is needed for
    the argmax.

    Returns: dict[setup_name] -> dict[phrase_name] -> dict[metric] -> float.
    """
    weighted_cache: dict[str, torch.Tensor] = {}
    results: dict[str, dict[str, dict[str, float]]] = {}
    for setup in setups:
        layer_range, head_reduce = setup[0], setup[1]
        weighting = setup[2] if len(setup) > 2 else None
        if weighting is None:
            src = audio_to_text
        else:
            if weighting not in weighted_cache:
                if not key_value_norms or weighting not in key_value_norms:
                    raise ValueError(
                        f"Setup requests weighting {weighting!r} but no "
                        "matching key_value_norms entry was provided."
                    )
                vnorm = key_value_norms[weighting]  # (L, H, T_text)
                expected = (
                    audio_to_text.size(0),
                    audio_to_text.size(1),
                    audio_to_text.size(3),
                )
                if tuple(vnorm.shape) != expected:
                    raise ValueError(
                        f"key_value_norms[{weighting!r}] shape "
                        f"{tuple(vnorm.shape)} != expected {expected}"
                    )
                # weighted[l, h, q, k] = attn[l, h, q, k] * ||v_k|| (broadcast
                # over the query/audio axis).
                weighted_cache[weighting] = audio_to_text.to(
                    torch.float32
                ) * vnorm.to(
                    device=audio_to_text.device, dtype=torch.float32
                )[:, :, None, :]
            src = weighted_cache[weighting]
        sub = _layer_slice(src, layer_range)
        if head_reduce == "sum":
            agg = sub.sum(dim=1).sum(dim=0)
        elif head_reduce == "mean":
            agg = sub.mean(dim=1).mean(dim=0)
        elif head_reduce == "max":
            agg = sub.amax(dim=1).amax(dim=0)
        else:
            raise ValueError(f"Unknown head_reduce: {head_reduce}")
        text_argmax = agg.argmax(dim=-1).detach().to(torch.int64).cpu().numpy()  # (T_audio,)
        setup_name = "__".join(setup)
        per_phrase: dict[str, dict[str, float]] = {}
        for name, text_mask in phrase_text_masks.items():
            pred = text_mask[text_argmax]  # (T_audio,) bool
            gt_mask = phrase_audio_gt_masks[name]
            m = _binary_set_metrics(pred, gt_mask, axis=-1)
            per_phrase[name] = {k: float(v) for k, v in m.items()}
        results[setup_name] = per_phrase
    return results


# ---------------------------------------------------------------------------
# Custom decoding loop with attention capture
# ---------------------------------------------------------------------------


def _predict_with_log_probs_simple(
    model: OmniVoice,
    c_logits: torch.Tensor,
    u_logits: torch.Tensor,
    gen_config: OmniVoiceGenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gen_config.guidance_scale != 0:
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        u_log_probs = F.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(
            c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)

    log_probs[..., model.config.audio_mask_id] = -float("inf")

    if gen_config.class_temperature > 0.0:
        # Mirrors examples/dlm_sampling_experiments._filter_top_k.
        k = max(1, math.ceil(0.1 * log_probs.shape[-1]))
        val, ind = log_probs.topk(k, dim=-1)
        filtered = torch.full_like(log_probs, float("-inf"))
        filtered.scatter_(-1, ind, val)
        sampled = _gumbel_sample(filtered, gen_config.class_temperature)
        pred_tokens = sampled.argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)
    confidence = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence


@torch.inference_mode()
def iterative_decode_with_attention_capture(
    model: OmniVoice,
    item,
    gen_config: OmniVoiceGenerationConfig,
    text_token_offset: int,  # cond-branch start of textual region (inclusive)
    text_token_end: int,  # cond-branch end of textual region (exclusive)
    target_start: int,  # cond-branch start of audio target region
    phrase_text_masks: dict[str, np.ndarray],
    phrase_audio_gt_masks: dict[str, np.ndarray],
    layer_agg_setups: list[tuple[str, ...]] = LAYER_AGG_SETUPS,
    max_steps: int = -1,
    save_full_attn: bool = False,
    vnorm_capture: Optional[_ValueNormCapture] = None,
) -> dict[str, Any]:
    """Run iterative decoding while capturing per-step phrase-audio alignment
    metrics (IoU / Recall / Precision / F1) against FA-derived GT masks.

    For each captured step, two histories accumulate:
        - "phrase_per_head_history": list[dict[phrase] -> dict[metric] -> (L, H)]
        - "phrase_agg_history": list[dict[setup_name] -> dict[phrase] -> dict[metric] -> float]
    plus ``captured_step_indices``, ``tokens``, ``remaining_masks_per_step``,
    and optional ``full_attn_audio_text``.

    ``vnorm_capture`` must be provided when any setup in ``layer_agg_setups``
    carries a value-norm weighting suffix (3-element tuple); the captured
    key-side norms are sliced to the text-token window and fed to
    ``compute_phrase_metrics_aggregated_for_step``.
    """

    device = model.device
    mask_id = model.config.audio_mask_id
    num_codebook = model.config.num_audio_codebook
    t_len = item.target_len

    sample_tokens = torch.full(
        (1, num_codebook, t_len),
        mask_id,
        dtype=torch.long,
        device=device,
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

    remaining_masks: list[int] = []
    full_attn_audio_text: list[np.ndarray] = []
    captured_step_indices: list[int] = []
    phrase_per_head_history: list[dict[str, dict[str, np.ndarray]]] = []
    phrase_agg_history: list[dict[str, dict[str, dict[str, float]]]] = []

    capture_limit = gen_config.num_step if max_steps <= 0 else min(max_steps, gen_config.num_step)

    for step in range(gen_config.num_step):
        cur_mask_count = int((sample_tokens == mask_id).sum().item())
        if cur_mask_count == 0:
            break

        capture_this_step = step < capture_limit
        if capture_this_step:
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
            0:1,
            :,
            cond_branch.target_start : cond_branch.target_start + t_len,
            :,
        ]
        u_logits = audio_logits[
            1:2,
            :,
            uncond_branch.target_start : uncond_branch.target_start + t_len,
            :,
        ]
        pred_tokens, confidence = _predict_with_log_probs_simple(
            model, c_logits, u_logits, gen_config
        )

        if attentions is not None and capture_this_step:
            stacked = torch.stack(attentions, dim=0)  # (L, B, H, S, S)
            cond_attn = stacked[:, 0]  # (L, H, S, S)
            audio_to_text = cond_attn[
                :,
                :,
                target_start : target_start + t_len,
                text_token_offset:text_token_end,
            ].contiguous()
            # Key-side value norms for the same text-token window (cond
            # branch = batch index 0), one (L, H, T_text) tensor per weighting.
            key_value_norms: Optional[dict[str, torch.Tensor]] = None
            if value_norms is not None:
                key_value_norms = {
                    wname: torch.stack([n[0] for n in per_layer], dim=0)[
                        :, :, text_token_offset:text_token_end
                    ].to(device=audio_to_text.device)
                    for wname, per_layer in value_norms.items()
                }
            phrase_per_head_history.append(
                compute_phrase_metrics_per_head_for_step(
                    audio_to_text, phrase_text_masks, phrase_audio_gt_masks
                )
            )
            phrase_agg_history.append(
                compute_phrase_metrics_aggregated_for_step(
                    audio_to_text,
                    phrase_text_masks,
                    phrase_audio_gt_masks,
                    layer_agg_setups,
                    key_value_norms=key_value_norms,
                )
            )
            captured_step_indices.append(int(step))
            if save_full_attn:
                full_attn_audio_text.append(
                    audio_to_text.detach().to(torch.float32).cpu().numpy()
                )

        # ---- Token sampling step (mirrors OmniVoice._generate_iterative) ----
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

        remaining_masks.append(int((sample_tokens == mask_id).sum().item()))

    return {
        "captured_step_indices": captured_step_indices,
        "phrase_per_head_history": phrase_per_head_history,
        "phrase_agg_history": phrase_agg_history,
        "tokens": sample_tokens.squeeze(0).detach().cpu(),
        "remaining_masks_per_step": remaining_masks,
        "full_attn_audio_text": full_attn_audio_text if save_full_attn else None,
    }


# ---------------------------------------------------------------------------
# Trajectory analysis & plotting
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj):
    """Recursively replace NaN / Inf floats with None so the resulting blob is
    valid strict-JSON (parsers like jq, browsers, and some Python tools reject
    the bare ``NaN`` literal that ``json.dump`` emits by default).
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def _summarize_trajectory(
    values: list[float], steps: list[int]
) -> dict[str, float]:
    """Summarize a per-step trajectory of a single metric.

    Returns first/last/max/min, step_of_max, delta_last_minus_first, and the
    Spearman rank correlation between step index and value (positive = the
    metric improves monotonically with decoding step).
    """
    arr = np.array(values, dtype=np.float64)
    nan_summary = {
        "first": float("nan"),
        "last": float("nan"),
        "max": float("nan"),
        "min": float("nan"),
        "step_of_max": -1,
        "delta_last_minus_first": float("nan"),
        "trend_spearman_step_vs_value": float("nan"),
    }
    if arr.size == 0:
        return nan_summary
    not_nan = ~np.isnan(arr)
    if not not_nan.any():
        return nan_summary
    valid_steps = np.array(steps, dtype=np.float64)[not_nan]
    valid_vals = arr[not_nan]
    max_local = int(np.argmax(valid_vals))
    return {
        "first": float(valid_vals[0]),
        "last": float(valid_vals[-1]),
        "max": float(np.max(valid_vals)),
        "min": float(np.min(valid_vals)),
        "step_of_max": int(valid_steps[max_local]),
        "delta_last_minus_first": float(valid_vals[-1] - valid_vals[0]),
        "trend_spearman_step_vs_value": _spearman_corr(valid_steps, valid_vals),
    }


def _plot_setup_trajectories(
    setup_trajectories: dict[str, dict[str, list[float]]],
    captured_steps: list[int],
    metric_key: str,
    out_path: Path,
    title: str,
    ranked_names: Optional[list[str]] = None,
    top_k: int = 12,
):
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib not available, skipping plot %s: %s", out_path, e)
        return
    if not setup_trajectories or not captured_steps:
        logger.warning("Empty trajectories, skipping plot %s", out_path)
        return
    names = ranked_names if ranked_names is not None else list(setup_trajectories.keys())
    names = names[:top_k]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        traj = setup_trajectories.get(name, {}).get(metric_key, [])
        if not traj:
            continue
        ax.plot(captured_steps, traj, marker="o", markersize=3, label=name, linewidth=1.2)
    ax.set_xlabel("decoding step")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    if metric_key.startswith("acc@"):
        ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_head_trajectories(
    captured_steps: list[int],
    head_cube: np.ndarray,  # (L, H, S)
    top_heads: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ylabel: str = "recall",
    trajectory_field: str = "trajectory_recall",
):
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib not available, skipping plot %s: %s", out_path, e)
        return
    if head_cube.size == 0 or not top_heads:
        logger.warning("Empty head trajectories, skipping plot %s", out_path)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for entry in top_heads:
        L_idx = int(entry["layer"])
        H_idx = int(entry["head"])
        traj = entry.get(trajectory_field) or head_cube[L_idx, H_idx].tolist()
        ax.plot(
            captured_steps,
            traj,
            marker="o",
            markersize=3,
            label=f"L{L_idx}H{H_idx}",
            linewidth=1.2,
        )
    ax.set_xlabel("decoding step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    args = build_parser().parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("analysis") / "outputs" / "attention_probe"
    )
    ensure_dir(out_dir)
    set_seed(args.seed)

    device = args.device or get_best_device()
    if args.edit_aligner_device is None:
        args.edit_aligner_device = device

    dtype = resolve_dtype(args.dtype, device)
    logger.info(
        "Loading model=%s device=%s dtype=%s attn_impl=%s",
        args.model,
        device,
        dtype,
        args.attn_implementation,
    )
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    gen_config = build_generation_config(args)
    voice_prompt = make_voice_prompt(model, args)
    if voice_prompt is None:
        raise ValueError("cs-attention-probe requires --ref-audio")

    # ----- Phrase parsing -----
    phrases = parse_delimited_list(args.insert_phrases)
    if not phrases:
        raise ValueError("--insert-phrases must list at least one phrase")

    # Prepare the inference task (the same one the model would consume).
    item = prepare_single_item(
        model,
        args.text,
        args,
        voice_prompt,
    )
    full_text_for_tokens = _combine_text(text=args.text, ref_text=item.ref_text)

    # We need TWO coordinate systems:
    #   (a) attention text-token range -> phrase position inside
    #       <|text_start|>{ref_text + " " + args.text}<|text_end|>
    #       i.e. char spans inside ``full_text_for_tokens`` (= _combine_text(...)).
    #   (b) FA oracle -> the synthesized audio only utters ``args.text`` (ref_audio
    #       is a *condition*, not part of the utterance), so the forced aligner
    #       must receive ``args.text`` and char spans inside ``args.text``.
    # Compute both.
    phrase_char_spans, merged_spans, missing_phrases = collect_phrase_char_spans(
        full_text_for_tokens,
        phrases,
        ignore_case=bool(args.phrase_ignore_case),
    )
    if not merged_spans:
        raise ValueError(
            "None of the requested phrases were located in the prepared text "
            f"(combined ref+text). Phrases: {phrases!r}; combined: {full_text_for_tokens!r}"
        )
    if missing_phrases:
        logger.warning("Phrases not found in combined ref+text: %s", missing_phrases)
    # FA-coordinate: phrases inside args.text only (the actually synthesized utterance).
    fa_text = args.text
    phrase_char_spans_fa, _fa_merged, fa_missing = collect_phrase_char_spans(
        fa_text,
        phrases,
        ignore_case=bool(args.phrase_ignore_case),
    )
    if fa_missing:
        logger.warning(
            "Phrases not found in args.text for FA oracle: %s. "
            "FA oracle will be unavailable for those phrases.",
            fa_missing,
        )

    # ----- Region offsets in the cond-branch input_ids -----
    cond_branch = _prepare_decode_branch(model, item, gen_config, branch_type="cond")
    target_start = cond_branch.target_start  # = c_len - t_len
    t_len = item.target_len
    total_len = cond_branch.total_len
    # Compute layout: style_tokens + text_tokens + ref_audio_tokens + target_audio_tokens.
    # We need the index range for text_tokens.
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
    wrapped_text_token_len = _length_of_tokens(wrapped_full_text, model.text_tokenizer)
    text_token_offset = style_token_len  # cond-branch start of text tokens
    text_token_end = text_token_offset + wrapped_text_token_len  # exclusive

    # Sanity-check: the textual region must fit before the target audio span.
    if text_token_end > target_start:
        raise RuntimeError(
            "Computed textual region overruns the start of the audio target span. "
            f"text_token_end={text_token_end} target_start={target_start}"
        )

    # ----- Phrase text-token ranges (kept in metrics.json for downstream tooling) -----
    phrase_token_ranges: dict[str, list[tuple[int, int]]] = {}
    for phrase, char_spans in phrase_char_spans.items():
        ranges: list[tuple[int, int]] = []
        for cs in char_spans:
            p_start, p_end = map_phrase_to_text_token_range(
                full_text_for_tokens, cs, model.text_tokenizer
            )
            p_start = max(0, min(wrapped_text_token_len, p_start))
            p_end = max(p_start + 1, min(wrapped_text_token_len, p_end))
            ranges.append((p_start, p_end))
        phrase_token_ranges[phrase] = ranges

    # ----- Build FA-derived audio->text GT (required for trajectory probe) -----
    frame_rate = model.audio_tokenizer.config.frame_rate
    fa_aligned_dicts: Optional[list[dict[str, Any]]] = None

    logger.info("Synthesizing reference audio for forced aligner ...")
    set_seed(args.seed)
    first_out = iterative_decode(model, item, gen_config)
    first_audio = decode_tokens(model, first_out.tokens, item.ref_rms, gen_config)
    sf.write(out_dir / "fa_synth.wav", first_audio, model.sampling_rate)

    aligner_language = _resolve_qwen_aligner_language(
        args.edit_aligner_language or args.language
    )
    raw_aligned_items = _align_qwen3_forced(
        args=args,
        audio=(first_audio.astype(np.float32, copy=False), model.sampling_rate),
        text=fa_text,
        language=aligner_language,
    )
    fa_aligned_dicts = _align_items_to_dicts(raw_aligned_items)

    # ----- Build per-phrase audio-token GT masks from FA + the union "__all__" -----
    try:
        phrase_audio_gt_masks = build_phrase_audio_gt_from_fa(
            aligned_items=raw_aligned_items,
            phrases=phrases,
            phrase_char_spans_fa=phrase_char_spans_fa,
            fa_text=fa_text,
            audio_token_len=t_len,
            frame_rate=frame_rate,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to build phrase audio GT from FA: {e}. "
            "The probe requires a working forced aligner."
        ) from e

    gt_meta: dict[str, Any] = {
        "audio_token_len": int(t_len),
        "per_phrase_audio_token_count": {
            name: int(mask.sum()) for name, mask in phrase_audio_gt_masks.items()
        },
        "__all__audio_token_coverage": float(
            phrase_audio_gt_masks["__all__"].mean()
        ),
    }
    logger.info(
        "Built FA-derived phrase audio GT: phrases=%s, __all__ coverage=%.3f",
        {name: int(m.sum()) for name, m in phrase_audio_gt_masks.items()},
        gt_meta["__all__audio_token_coverage"],
    )

    # ----- Build phrase text-token masks (one bool array per phrase + __all__) -----
    phrase_text_masks: dict[str, np.ndarray] = {}
    union_text = np.zeros(wrapped_text_token_len, dtype=bool)
    for phrase, ranges in phrase_token_ranges.items():
        m = _phrase_text_mask(ranges, wrapped_text_token_len)
        phrase_text_masks[phrase] = m
        union_text |= m
    phrase_text_masks["__all__"] = union_text

    # ----- Build layer-agg setups (built-in 9 + optional user-specified probe layers) -----
    layer_agg_setups = list(LAYER_AGG_SETUPS)
    user_probe_specs = parse_delimited_list(args.probe_layers) if args.probe_layers else []
    for spec in user_probe_specs:
        try:
            range_name = _probe_layer_spec_to_range_name(spec)
        except Exception as e:
            logger.warning("Skipping invalid --probe-layers entry %r: %s", spec, e)
            continue
        for reducer in HEAD_REDUCERS:
            layer_agg_setups.append((range_name, reducer))
    if user_probe_specs:
        logger.info(
            "Layer-agg setups extended with --probe-layers: %s",
            [f"{lr}__{hr}" for lr, hr in layer_agg_setups[len(LAYER_AGG_SETUPS):]],
        )

    # ----- Optional value-norm-weighted twins (Kobayashi et al., 2020) -----
    weighted_variants: list[str] = []
    if bool(args.probe_vnorm):
        weighted_variants.append("vnorm")
    if bool(args.probe_vnormfx):
        weighted_variants.append("vnormfx")
    vnorm_capture: Optional[_ValueNormCapture] = None
    if weighted_variants:
        base_setups = list(layer_agg_setups)
        for weighting in weighted_variants:
            layer_agg_setups.extend(
                (lr, hr, weighting) for lr, hr in base_setups
            )
        logger.info(
            "Added %s twins for %d base setups (total setups: %d).",
            "+".join(weighted_variants),
            len(base_setups),
            len(layer_agg_setups),
        )
        vnorm_capture = _ValueNormCapture(
            model, capture_fx=("vnormfx" in weighted_variants)
        )

    # ----- Run instrumented decoding -----
    logger.info(
        "Running instrumented iterative decode (num_step=%d, capture<=%d) ...",
        gen_config.num_step,
        gen_config.num_step if args.max_steps <= 0 else args.max_steps,
    )
    set_seed(args.seed)
    capture = iterative_decode_with_attention_capture(
        model=model,
        item=item,
        gen_config=gen_config,
        text_token_offset=text_token_offset,
        text_token_end=text_token_end,
        target_start=target_start,
        phrase_text_masks=phrase_text_masks,
        phrase_audio_gt_masks=phrase_audio_gt_masks,
        layer_agg_setups=layer_agg_setups,
        max_steps=args.max_steps,
        save_full_attn=args.save_attn,
        vnorm_capture=vnorm_capture,
    )
    if vnorm_capture is not None:
        vnorm_capture.remove()

    captured_steps: list[int] = capture["captured_step_indices"]
    phrase_per_head_history = capture["phrase_per_head_history"]
    phrase_agg_history = capture["phrase_agg_history"]
    if not captured_steps or not phrase_per_head_history:
        raise RuntimeError(
            "No attention captured. Check --attn-implementation; eager is required."
        )
    phrase_names = list(phrase_audio_gt_masks.keys())  # includes "__all__"
    L, H = phrase_per_head_history[0]["__all__"]["recall"].shape
    S = len(captured_steps)
    logger.info(
        "Captured phrase metrics for %d layers x %d heads over %d steps "
        "(phrases: %s).",
        L, H, S, phrase_names,
    )

    setup_names = ["__".join(s) for s in layer_agg_setups]
    PRIMARY = "recall"   # main ranking metric -- false negatives are the costly failure mode for CFG
    SECONDARY = "f1"

    # ----- Per-head metric cubes (per phrase): dict[phrase][metric] -> (L, H, S) -----
    head_cubes: dict[str, dict[str, np.ndarray]] = {}
    for pname in phrase_names:
        head_cubes[pname] = {
            k: np.stack(
                [h[pname][k] for h in phrase_per_head_history], axis=-1
            )
            for k in METRIC_KEYS
        }

    # ----- Per-setup trajectories: dict[setup][phrase][metric] -> list[float] -----
    setup_trajectories: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {pname: {k: [] for k in METRIC_KEYS} for pname in phrase_names}
        for name in setup_names
    }
    for agg in phrase_agg_history:
        for name in setup_names:
            per_phrase = agg.get(name, {})
            for pname in phrase_names:
                m = per_phrase.get(pname, {})
                for k in METRIC_KEYS:
                    setup_trajectories[name][pname][k].append(
                        float(m.get(k, float("nan")))
                    )

    # ----- Per-setup summary (only the __all__ phrase block is summarized for
    # ranking; per-phrase trajectories are still preserved in the dump) -----
    setup_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name in setup_names:
        setup_summary[name] = {
            pname: {
                k: _summarize_trajectory(
                    setup_trajectories[name][pname][k], captured_steps
                )
                for k in METRIC_KEYS
            }
            for pname in phrase_names
        }

    def _setup_sort_key(name: str, metric: str, field: str) -> float:
        v = setup_summary[name]["__all__"][metric][field]
        return v if not math.isnan(v) else -float("inf")

    ranked_setups_by_last = sorted(
        setup_names,
        key=lambda n: _setup_sort_key(n, PRIMARY, "last"),
        reverse=True,
    )
    ranked_setups_by_mean = sorted(
        setup_names,
        key=lambda n: (
            float(np.nanmean(setup_trajectories[n]["__all__"][PRIMARY]))
            if any(
                not math.isnan(x) for x in setup_trajectories[n]["__all__"][PRIMARY]
            )
            else -float("inf")
        ),
        reverse=True,
    )
    ranked_setups_by_min = sorted(
        setup_names,
        key=lambda n: _setup_sort_key(n, PRIMARY, "min"),
        reverse=True,
    )
    ranked_setups_by_delta = sorted(
        setup_names,
        key=lambda n: _setup_sort_key(n, PRIMARY, "delta_last_minus_first"),
        reverse=True,
    )
    ranked_setups_by_trend = sorted(
        setup_names,
        key=lambda n: _setup_sort_key(n, PRIMARY, "trend_spearman_step_vs_value"),
        reverse=True,
    )

    # ----- Per-head trajectory summary on the __all__ block (vectorized) -----
    all_cubes = head_cubes["__all__"]
    primary_cube = all_cubes[PRIMARY]  # (L, H, S)
    steps_arr = np.array(captured_steps, dtype=np.float64)
    nan_mask = np.isnan(primary_cube)
    any_valid = ~nan_mask.all(axis=-1)
    first_idx = np.argmax(~nan_mask, axis=-1)
    last_idx = (primary_cube.shape[-1] - 1) - np.argmax(~nan_mask[..., ::-1], axis=-1)
    p_first = np.take_along_axis(primary_cube, first_idx[..., None], axis=-1).squeeze(-1)
    p_last = np.take_along_axis(primary_cube, last_idx[..., None], axis=-1).squeeze(-1)
    p_no_nan = np.where(nan_mask, -np.inf, primary_cube)
    p_argmax_local = np.argmax(p_no_nan, axis=-1)
    p_step_of_max = steps_arr[p_argmax_local].astype(np.int64)
    p_max = np.where(any_valid, np.nanmax(primary_cube, axis=-1), np.nan)
    p_min = np.where(any_valid, np.nanmin(primary_cube, axis=-1), np.nan)
    p_mean = np.where(any_valid, np.nanmean(primary_cube, axis=-1), np.nan)
    p_delta = np.where(any_valid, p_last - p_first, np.nan)
    p_first = np.where(any_valid, p_first, np.nan)
    p_last = np.where(any_valid, p_last, np.nan)
    p_step_of_max = np.where(any_valid, p_step_of_max, -1)

    secondary_cube = all_cubes[SECONDARY]
    s_first = np.take_along_axis(secondary_cube, first_idx[..., None], axis=-1).squeeze(-1)
    s_last = np.take_along_axis(secondary_cube, last_idx[..., None], axis=-1).squeeze(-1)
    s_first = np.where(any_valid, s_first, np.nan)
    s_last = np.where(any_valid, s_last, np.nan)

    def _select_top_heads(score_matrix: np.ndarray, K: int) -> list[tuple[int, int]]:
        flat = score_matrix.flatten()
        valid = ~np.isnan(flat)
        if not valid.any():
            return []
        Keff = max(1, min(K, int(valid.sum())))
        idx = np.argpartition(-np.where(valid, flat, -np.inf), Keff - 1)[:Keff]
        idx = idx[np.argsort(-flat[idx])]
        return [(int(fi // H), int(fi % H)) for fi in idx]

    def _build_head_entry(L_idx: int, H_idx: int) -> dict[str, Any]:
        return {
            "layer": L_idx,
            "head": H_idx,
            f"{PRIMARY}_first": float(p_first[L_idx, H_idx]),
            f"{PRIMARY}_last": float(p_last[L_idx, H_idx]),
            f"{PRIMARY}_min": float(p_min[L_idx, H_idx]),
            f"{PRIMARY}_max": float(p_max[L_idx, H_idx]),
            f"{PRIMARY}_mean": float(p_mean[L_idx, H_idx]),
            f"{PRIMARY}_step_of_max": int(p_step_of_max[L_idx, H_idx]),
            f"{PRIMARY}_delta_last_minus_first": float(p_delta[L_idx, H_idx]),
            f"{SECONDARY}_first": float(s_first[L_idx, H_idx]),
            f"{SECONDARY}_last": float(s_last[L_idx, H_idx]),
            f"trajectory_{PRIMARY}": [float(x) for x in primary_cube[L_idx, H_idx].tolist()],
            f"trajectory_{SECONDARY}": [float(x) for x in secondary_cube[L_idx, H_idx].tolist()],
        }

    top_heads_by_last = [
        _build_head_entry(L_idx, H_idx)
        for L_idx, H_idx in _select_top_heads(p_last, args.top_k_heads)
    ]
    top_heads_by_mean = [
        _build_head_entry(L_idx, H_idx)
        for L_idx, H_idx in _select_top_heads(p_mean, args.top_k_heads)
    ]
    top_heads_by_min = [
        _build_head_entry(L_idx, H_idx)
        for L_idx, H_idx in _select_top_heads(p_min, args.top_k_heads)
    ]
    top_heads_by_delta = [
        _build_head_entry(L_idx, H_idx)
        for L_idx, H_idx in _select_top_heads(p_delta, args.top_k_heads)
    ]

    # ----- Plots -----
    title_suffix = (
        f"{args.model} | text='{args.text[:60] + ('...' if len(args.text) > 60 else '')}'"
    )
    # Per-setup trajectory plots use the __all__ phrase block.
    all_setup_traj = {
        name: setup_trajectories[name]["__all__"] for name in setup_names
    }
    _plot_setup_trajectories(
        setup_trajectories=all_setup_traj,
        captured_steps=captured_steps,
        metric_key=PRIMARY,
        out_path=out_dir / f"setup_{PRIMARY}_trajectory.png",
        title=f"Per-setup phrase-audio {PRIMARY} (__all__) - {title_suffix}",
        ranked_names=ranked_setups_by_mean,
    )
    _plot_setup_trajectories(
        setup_trajectories=all_setup_traj,
        captured_steps=captured_steps,
        metric_key=SECONDARY,
        out_path=out_dir / f"setup_{SECONDARY}_trajectory.png",
        title=f"Per-setup phrase-audio {SECONDARY} (__all__) - {title_suffix}",
        ranked_names=ranked_setups_by_mean,
    )
    _plot_head_trajectories(
        captured_steps=captured_steps,
        head_cube=primary_cube,
        top_heads=top_heads_by_mean,
        out_path=out_dir / f"head_{PRIMARY}_trajectory_by_mean.png",
        title=f"Top heads by {PRIMARY}_mean - {title_suffix}",
        ylabel=PRIMARY,
        trajectory_field=f"trajectory_{PRIMARY}",
    )
    _plot_head_trajectories(
        captured_steps=captured_steps,
        head_cube=primary_cube,
        top_heads=top_heads_by_min,
        out_path=out_dir / f"head_{PRIMARY}_trajectory_by_min.png",
        title=f"Top heads by {PRIMARY}_min (worst-step guarantee) - {title_suffix}",
        ylabel=PRIMARY,
        trajectory_field=f"trajectory_{PRIMARY}",
    )

    # ----- Optional: save full attention -----
    if args.save_attn:
        full_attn_list = capture["full_attn_audio_text"] or []
        if full_attn_list:
            arr = np.stack(full_attn_list, axis=0)  # (S, L, H, T_audio, T_text)
            logger.warning(
                "Saving attn_maps.npz: shape=%s dtype=%s (~%.2f MB).",
                arr.shape,
                arr.dtype,
                arr.nbytes / (1024 * 1024),
            )
            np.savez_compressed(out_dir / "attn_maps.npz", attn=arr)

    # ----- metrics.json -----
    config_payload = {k: v for k, v in vars(args).items()}
    for k, v in list(config_payload.items()):
        if isinstance(v, Path):
            config_payload[k] = str(v)

    metrics: dict[str, Any] = {
        "config": config_payload,
        "model": {
            "name": args.model,
            "num_layers": int(L),
            "num_heads": int(H),
        },
        "text_layout": {
            "full_text_for_tokens": full_text_for_tokens,
            "style_token_len": style_token_len,
            "wrapped_text_token_len": wrapped_text_token_len,
            "text_token_offset": text_token_offset,
            "text_token_end": text_token_end,
            "target_start": target_start,
            "target_len": t_len,
            "total_len": total_len,
        },
        "phrases": phrases,
        "missing_phrases": missing_phrases,
        "phrase_char_spans": {
            phrase: [list(x) for x in spans]
            for phrase, spans in phrase_char_spans.items()
        },
        "phrase_text_token_ranges": {
            phrase: [list(x) for x in ranges]
            for phrase, ranges in phrase_token_ranges.items()
        },
        "phrase_audio_gt_meta": gt_meta,
        "captured_step_indices": captured_steps,
        "remaining_masks_per_step": capture["remaining_masks_per_step"],
        "primary_metric": PRIMARY,
        "per_setup": {
            "setup_names": setup_names,
            "phrase_names": phrase_names,
            "trajectories": setup_trajectories,
            "summary": setup_summary,
            f"ranked_by_{PRIMARY}_last": ranked_setups_by_last,
            f"ranked_by_{PRIMARY}_mean": ranked_setups_by_mean,
            f"ranked_by_{PRIMARY}_min": ranked_setups_by_min,
            f"ranked_by_{PRIMARY}_delta_last_minus_first": ranked_setups_by_delta,
            f"ranked_by_{PRIMARY}_trend": ranked_setups_by_trend,
            f"best_setup_by_{PRIMARY}_mean": (
                ranked_setups_by_mean[0] if ranked_setups_by_mean else None
            ),
            f"best_setup_{PRIMARY}_mean": (
                float(np.nanmean(setup_trajectories[ranked_setups_by_mean[0]]["__all__"][PRIMARY]))
                if ranked_setups_by_mean else float("nan")
            ),
            f"best_setup_by_{PRIMARY}_min": (
                ranked_setups_by_min[0] if ranked_setups_by_min else None
            ),
            f"best_setup_{PRIMARY}_min": (
                setup_summary[ranked_setups_by_min[0]]["__all__"][PRIMARY]["min"]
                if ranked_setups_by_min else float("nan")
            ),
        },
        "top_heads": {
            f"by_{PRIMARY}_last": top_heads_by_last,
            f"by_{PRIMARY}_mean": top_heads_by_mean,
            f"by_{PRIMARY}_min": top_heads_by_min,
            f"by_{PRIMARY}_delta_last_minus_first": top_heads_by_delta,
        },
    }
    if fa_aligned_dicts is not None:
        metrics["fa_aligned_items"] = fa_aligned_dicts

    write_json(out_dir / "metrics.json", _sanitize_for_json(metrics))

    torch.save(
        {
            "tokens_final": capture["tokens"],
            "phrase_audio_gt_masks": {
                name: torch.from_numpy(mask.astype(np.bool_))
                for name, mask in phrase_audio_gt_masks.items()
            },
            "phrase_text_masks": {
                name: torch.from_numpy(mask.astype(np.bool_))
                for name, mask in phrase_text_masks.items()
            },
            "captured_step_indices": torch.tensor(captured_steps, dtype=torch.long),
            f"all_{PRIMARY}_cube": torch.from_numpy(primary_cube),
            f"all_{SECONDARY}_cube": torch.from_numpy(secondary_cube),
        },
        out_dir / "tokens.pt",
    )

    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026  Xiaomi Corp.
#
# See ../LICENSE for clarification regarding multiple authors

"""Smoke test for the value-norm-weighted attention probe (a2t_vnorm).

Two stages:

1. Pure-tensor unit tests (always run, no model / GPU needed):
   - ``_parse_setup`` accepts a2t / a2t_vnorm / a2t_vnormfx, rejects others.
   - The Kobayashi weighting math (broadcast over the query axis) matches a
     reference loop, and a constant vnorm leaves the argmax unchanged.
   - ``_ValueNormCapture`` hook output on a tiny fake GQA model matches
     hand-computed ``||v_j||`` (incl. the repeat_kv head expansion) and
     ``||W_O^h v_j||`` (capture_fx).
   - ``compute_phrase_metrics_aggregated_for_step`` raw setups are unchanged
     by the presence of vnorm twins in the same call.

2. Model-level smoke (skipped gracefully if OmniVoice cannot be loaded):
   loads k2-fsa/OmniVoice, decodes ONE short CS utterance with setups
   ``layers_set_8-12__max__a2t`` (with and without vnorm capture active) and
   ``layers_set_8-12__max__a2t_vnorm``, and checks:
     (a) raw a2t argmax indices are bit-identical with/without vnorm hooks;
     (b) the vnorm path runs end-to-end and produces a mask;
     (c) prints the fraction of frames where raw vs vnorm masks differ.

Usage (single GPU recommended; the model is small):
    CUDA_VISIBLE_DEVICES=0 python analysis/_smoke_vnorm.py
    python analysis/_smoke_vnorm.py --skip-model   # tensor tests only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in (_ROOT, _ROOT / "synth", _THIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: pure-tensor unit tests
# ---------------------------------------------------------------------------


def _check(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    logger.info("[unit] %-55s %s", name, status)
    if not cond:
        raise AssertionError(f"unit test failed: {name}")


def test_parse_setup():
    from attention_probe_dataset import _parse_setup, _weightings_needed

    _check(
        "parse a2t",
        _parse_setup("layers_set_8-12__max__a2t")
        == ("layers_set_8-12", "max", "a2t"),
    )
    _check(
        "parse a2t_vnorm",
        _parse_setup("layers_set_8-12__max__a2t_vnorm")
        == ("layers_set_8-12", "max", "a2t_vnorm"),
    )
    _check(
        "parse a2t_vnormfx",
        _parse_setup("layer_9__mean__a2t_vnormfx")
        == ("layer_9", "mean", "a2t_vnormfx"),
    )
    for bad in ("layer_9__mean__t2a", "layer_9__mean__a2t_bogus", "layer_9__a2t"):
        try:
            _parse_setup(bad)
            ok = False
        except ValueError:
            ok = True
        _check(f"reject {bad!r}", ok)
    _check(
        "weightings_needed",
        _weightings_needed(
            [
                ("all", "max", "a2t"),
                ("all", "max", "a2t_vnorm"),
                ("all", "mean", "a2t_vnormfx"),
            ]
        )
        == ["vnorm", "vnormfx"],
    )
    _check("weightings_needed empty", _weightings_needed([("all", "max", "a2t")]) == [])


def test_weighting_math():
    from attention_probe import _layer_slice

    g = torch.Generator().manual_seed(0)
    L, H, Ta, Tt = 4, 6, 10, 7
    attn = torch.rand((L, H, Ta, Tt), generator=g)
    attn = attn / attn.sum(dim=-1, keepdim=True)  # row-stochastic like softmax
    vnorm = torch.rand((L, H, Tt), generator=g) + 0.1

    weighted = attn * vnorm[:, :, None, :]
    ref = torch.empty_like(weighted)
    for l in range(L):
        for h in range(H):
            for q in range(Ta):
                for k in range(Tt):
                    ref[l, h, q, k] = attn[l, h, q, k] * vnorm[l, h, k]
    _check("weighted[b,h,q,k] = attn * vnorm[k]", torch.allclose(weighted, ref))

    # Constant key-side norm must not change any argmax (pure rescale).
    const = attn * torch.full((L, H, Tt), 3.7)[:, :, None, :]
    same = torch.equal(
        _layer_slice(const, "layers_set_0-2").amax(dim=1).amax(dim=0).argmax(-1),
        _layer_slice(attn, "layers_set_0-2").amax(dim=1).amax(dim=0).argmax(-1),
    )
    _check("constant vnorm leaves argmax unchanged", same)

    # A spiked norm on one key column must attract the argmax there.
    vnorm_spike = torch.ones((L, H, Tt))
    vnorm_spike[..., 3] = 1e6
    spiked = attn * vnorm_spike[:, :, None, :]
    agg = _layer_slice(spiked, "all").amax(dim=1).amax(dim=0)
    _check("spiked ||v_k|| attracts argmax", bool((agg.argmax(-1) == 3).all()))


class _FakeAttn(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, *a):
        super().__init__()
        self.self_attn = _FakeAttn(*a)


def _make_fake_model(num_layers=3, hidden=32, num_heads=4, num_kv_heads=2, head_dim=8):
    llm = nn.Module()
    llm.layers = nn.ModuleList(
        _FakeLayer(hidden, num_heads, num_kv_heads, head_dim)
        for _ in range(num_layers)
    )
    llm.config = SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
    )
    model = nn.Module()
    model.llm = llm
    return model, num_layers, hidden, num_heads, num_kv_heads, head_dim


def _repeat_kv_reference(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """transformers.modeling_*.repeat_kv reference: (B, KV, S, D) -> (B, H, S, D)."""
    b, kv, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


def test_vnorm_capture_hooks():
    from attention_probe import _ValueNormCapture

    torch.manual_seed(1)
    model, L, hidden, H, KV, D = _make_fake_model()
    groups = H // KV
    cap = _ValueNormCapture(model, capture_fx=True)

    B, S = 2, 5
    x = torch.randn(B, S, hidden)

    # Hooks must be inert while disabled.
    cap.reset()
    for layer in model.llm.layers:
        layer.self_attn.v_proj(x)
    _check("hooks inert when disabled", all(n is None for n in cap._norms))

    cap.reset()
    cap.enable()
    try:
        for layer in model.llm.layers:
            layer.self_attn.v_proj(x)
    finally:
        cap.disable()
    norms = cap.collect()
    _check("collect keys", set(norms.keys()) == {"vnorm", "vnormfx"})
    _check("per-layer count", len(norms["vnorm"]) == L)
    _check(
        "vnorm shape (B, H, S)",
        all(tuple(n.shape) == (B, H, S) for n in norms["vnorm"]),
    )

    # Reference: ||v|| after an explicit repeat_kv on (B, KV, S, D).
    for li, layer in enumerate(model.llm.layers):
        v = layer.self_attn.v_proj(x).view(B, S, KV, D).permute(0, 2, 1, 3)
        v_rep = _repeat_kv_reference(v.to(torch.float32), groups)  # (B, H, S, D)
        ref = v_rep.norm(dim=-1)  # (B, H, S)
        _check(
            f"layer {li} ||v_j|| matches repeat_kv reference",
            torch.allclose(norms["vnorm"][li], ref, atol=1e-5),
        )
        # Full Kobayashi: ||W_O^h v_j|| with per-head o_proj column slices.
        w = layer.self_attn.o_proj.weight.to(torch.float32)  # (hidden, H*D)
        ref_fx = torch.empty(B, H, S)
        for h in range(H):
            w_h = w[:, h * D : (h + 1) * D]  # (hidden, D)
            ref_fx[:, h] = (v_rep[:, h] @ w_h.T).norm(dim=-1)
        _check(
            f"layer {li} ||W_O^h v_j|| matches per-head slice reference",
            torch.allclose(norms["vnormfx"][li], ref_fx, atol=1e-5),
        )

    cap.remove()
    _check("remove() detaches hooks", not any(
        layer.self_attn.v_proj._forward_hooks for layer in model.llm.layers
    ))


def test_aggregated_metrics_raw_unchanged():
    from attention_probe import compute_phrase_metrics_aggregated_for_step

    torch.manual_seed(2)
    L, H, Ta, Tt = 4, 6, 12, 9
    attn = torch.rand(L, H, Ta, Tt)
    attn = attn / attn.sum(-1, keepdim=True)
    vnorm = torch.rand(L, H, Tt) + 0.1
    tmask = np.zeros(Tt, dtype=bool)
    tmask[3:6] = True
    gmask = np.zeros(Ta, dtype=bool)
    gmask[4:9] = True
    text_masks = {"phr": tmask, "__all__": tmask}
    gt_masks = {"phr": gmask, "__all__": gmask}

    raw_only = compute_phrase_metrics_aggregated_for_step(
        attn, text_masks, gt_masks, [("all", "max"), ("last_1", "mean")]
    )
    mixed = compute_phrase_metrics_aggregated_for_step(
        attn,
        text_masks,
        gt_masks,
        [("all", "max"), ("last_1", "mean"), ("all", "max", "vnorm")],
        key_value_norms={"vnorm": vnorm},
    )
    same = all(
        raw_only[name][ph][m] == mixed[name][ph][m]
        for name in ("all__max", "last_1__mean")
        for ph in ("phr", "__all__")
        for m in ("iou", "recall", "precision", "f1")
    )
    _check("raw setups unchanged when vnorm twins present", same)
    _check("vnorm twin present in results", "all__max__vnorm" in mixed)

    # Missing norms for a requested weighting must raise.
    try:
        compute_phrase_metrics_aggregated_for_step(
            attn, text_masks, gt_masks, [("all", "max", "vnorm")]
        )
        ok = False
    except ValueError:
        ok = True
    _check("missing key_value_norms raises", ok)


# ---------------------------------------------------------------------------
# Stage 2: model-level smoke
# ---------------------------------------------------------------------------


def run_model_smoke(args_cli) -> bool:
    """Returns True if the model smoke ran, False if it was skipped."""
    try:
        from omnivoice import OmniVoice
        from omnivoice.models.omnivoice import _combine_text

        from dlm_sampling_experiments import (
            _prepare_decode_branch,
            add_shared_model_and_sampling_args,
            build_generation_config,
            collect_phrase_char_spans,
            get_best_device,
            make_voice_prompt,
            prepare_single_item,
            resolve_dtype,
            set_seed,
        )
        from attention_probe import (
            _length_of_tokens,
            _phrase_text_mask,
            _ValueNormCapture,
            map_phrase_to_text_token_range,
        )
        from attention_probe_dataset import _decode_capture_text_idx, _parse_setup
    except Exception as e:  # noqa: BLE001
        logger.warning("[model] import failed, skipping model smoke: %s", e)
        return False

    p = argparse.ArgumentParser()
    add_shared_model_and_sampling_args(p)
    args = p.parse_args([])
    args.ref_audio = str(_ROOT / "examples" / "en_ref.wav")
    args.ref_text = (
        "with the particular purposes of the agency involved. The Commission "
        "recognizes that this is a controversial area"
    )
    args.language = "en"
    args.num_step = int(args_cli.num_step)

    device = args.device or get_best_device()
    dtype = resolve_dtype(args.dtype, device)
    logger.info("[model] Loading %s on %s (%s) ...", args.model, device, dtype)
    try:
        model = OmniVoice.from_pretrained(
            args.model,
            device_map=device,
            dtype=dtype,
            attn_implementation="eager",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[model] could not load %s (%s); skipping model smoke. "
            "Tensor-level unit tests above still validate the vnorm math.",
            args.model,
            e,
        )
        return False

    gen_config = build_generation_config(args)
    voice_prompt = make_voice_prompt(model, args)
    text = args_cli.text
    phrase = args_cli.phrase

    item = prepare_single_item(model, text, args, voice_prompt)
    full_text_for_tokens = _combine_text(text=text, ref_text=item.ref_text)
    cond_branch = _prepare_decode_branch(model, item, gen_config, branch_type="cond")
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
    wrapped_text_token_len = _length_of_tokens(wrapped_full_text, model.text_tokenizer)
    text_token_offset = style_token_len
    text_token_end = text_token_offset + wrapped_text_token_len

    phrase_char_spans, _, _ = collect_phrase_char_spans(
        full_text_for_tokens, [phrase], ignore_case=True
    )
    ranges = [
        map_phrase_to_text_token_range(full_text_for_tokens, cs, model.text_tokenizer)
        for cs in phrase_char_spans.get(phrase, [])
    ]
    phrase_mask = _phrase_text_mask(ranges, wrapped_text_token_len)
    logger.info(
        "[model] t_len=%d text_tokens=%d phrase_token_ranges=%s",
        t_len, wrapped_text_token_len, ranges,
    )

    raw_name = "layers_set_8-12__max__a2t"
    vnorm_name = "layers_set_8-12__max__a2t_vnorm"
    vnormfx_name = "layers_set_8-12__max__a2t_vnormfx"

    # Run A: raw a2t only, NO vnorm code active (pre-existing path).
    set_seed(args.seed)
    cap_a = _decode_capture_text_idx(
        model, item, gen_config,
        text_token_offset=text_token_offset,
        text_token_end=text_token_end,
        target_start=target_start,
        setups=[_parse_setup(raw_name)],
        max_steps=-1,
        vnorm_capture=None,
    )

    # Run B: raw + vnorm + vnormfx setups, hooks active.
    vnorm_capture = _ValueNormCapture(model, capture_fx=True)
    set_seed(args.seed)
    cap_b = _decode_capture_text_idx(
        model, item, gen_config,
        text_token_offset=text_token_offset,
        text_token_end=text_token_end,
        target_start=target_start,
        setups=[
            _parse_setup(raw_name),
            _parse_setup(vnorm_name),
            _parse_setup(vnormfx_name),
        ],
        max_steps=-1,
        vnorm_capture=vnorm_capture,
    )
    vnorm_capture.remove()

    idx_a = cap_a["per_step_text_idx"][raw_name]  # (S, T_audio)
    idx_b_raw = cap_b["per_step_text_idx"][raw_name]
    idx_b_vn = cap_b["per_step_text_idx"][vnorm_name]
    idx_b_fx = cap_b["per_step_text_idx"][vnormfx_name]

    same_raw = np.array_equal(idx_a, idx_b_raw)
    logger.info(
        "[model] (a) raw a2t argmax identical with/without vnorm capture: %s",
        "PASS" if same_raw else "FAIL",
    )
    logger.info(
        "[model] (b) vnorm path end-to-end: PASS (shape=%s, steps=%s)",
        idx_b_vn.shape, len(cap_b["captured_step_indices"]),
    )

    mask_raw = phrase_mask[idx_b_raw]  # (S, T_audio) bool
    mask_vn = phrase_mask[idx_b_vn]
    mask_fx = phrase_mask[idx_b_fx]
    frac_idx = float((idx_b_raw != idx_b_vn).mean())
    frac_mask = float((mask_raw != mask_vn).mean())
    frac_mask_last = float((mask_raw[-1] != mask_vn[-1]).mean())
    logger.info(
        "[model] (c) raw vs vnorm: argmax-index diff=%.4f | phrase-mask diff="
        "%.4f (all steps) / %.4f (last step) | mask coverage raw=%.4f vnorm=%.4f",
        frac_idx, frac_mask, frac_mask_last,
        float(mask_raw.mean()), float(mask_vn.mean()),
    )
    logger.info(
        "[model] (c') raw vs vnormfx: argmax-index diff=%.4f | phrase-mask "
        "diff=%.4f (all steps) | mask coverage vnormfx=%.4f",
        float((idx_b_raw != idx_b_fx).mean()),
        float((mask_raw != mask_fx).mean()),
        float(mask_fx.mean()),
    )
    if not same_raw:
        raise AssertionError("raw a2t path changed when vnorm capture is active")
    return True


def main():
    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--text", type=str, default="I really love 김치찌개 in winter.")
    p.add_argument("--phrase", type=str, default="김치찌개")
    args_cli = p.parse_args()

    test_parse_setup()
    test_weighting_math()
    test_vnorm_capture_hooks()
    test_aggregated_metrics_raw_unchanged()
    logger.info("[unit] ALL PASS")

    if args_cli.skip_model:
        logger.info("[model] skipped (--skip-model).")
        return
    ran = run_model_smoke(args_cli)
    logger.info("[model] %s", "DONE" if ran else "SKIPPED (see warnings above)")


if __name__ == "__main__":
    main()

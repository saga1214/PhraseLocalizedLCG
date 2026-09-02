"""Attention-backend overhead benchmark (paper appendix: Runtime Cost of Attention Extraction).

Quantifies the runtime cost of the LCG method's requirement for attention
weights (which forces ``attn_implementation='eager'``) against a production
SDPA setup, on a subset of the released 1,200-entry benchmark manifest.

Three conditions (all mirror ``synth/synth.py`` sampling defaults: seed 42,
bfloat16, num_step 32, guidance_scale 2.0, t_shift 0.1):

    A  baseline (no-CS synthesis, vanilla CFG ``iterative_decode``)     sdpa
    B  baseline (no-CS synthesis, vanilla CFG ``iterative_decode``)     eager
    C  LCG (cs-attention-plc-langdir, λ=7, margin=4, xlang-union on,
       attention setup layers_set_8-12__max__a2t) — eager is the only
       possible backend (sdpa/flash return ``attentions=None``)         eager

Per utterance we measure wall-clock synthesis time — excluding model load
and voice-prompt preparation, including text preprocessing, the full
iterative decode and the audio-token -> waveform decode (wav file I/O is
excluded) — plus the output audio duration read back from the written wav,
giving RTF = synth_sec / audio_sec.

Outputs (under --out-root, default ``_artifacts/bench/bench_backend_out/``):
    <condition>/<entry_id>.wav   generated audio (for later quality metrics)
    results.jsonl                one record per measured utterance
    summary.json                 per-condition stats + B/A, C/A, C/B ratios

The engine's baseline path is the exact code ``run_cs_attention_plc`` uses to
emit ``baseline_src_only.wav``; the LCG path mirrors the same function's
setup + ``iterative_decode_with_attention_plc`` call, skipping only the
baseline byproduct and the metrics/tokens dumps so the timed region covers
exactly one synthesis.

Usage (single GPU; respect CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0 python analysis/bench_attention_backend.py
    python analysis/bench_attention_backend.py --dry-run
    See analysis/run_bench_backend.sh for the launcher.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

# Resolve repo root (PhraseLocalizedLCG/). This file lives at <repo>/analysis/.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
SYNTH_DIR = REPO_ROOT / "synth"

# Both the omnivoice/ package and the LCG engine module (synth/dlm_sampling_experiments.py)
# need to be importable; synth/synth.py provides the canonical args-namespace builder.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SYNTH_DIR))

# Manifest references "examples/<lang>_ref.wav" with paths relative to the repo root.
os.chdir(REPO_ROOT)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    force=True,
)
log = logging.getLogger("bench_backend")

# synth/synth.py only imports stdlib at module level, so this is dry-run safe.
from synth import ATTENTION_SETUP, build_args_namespace, load_manifest  # noqa: E402

# Paper-final "Ours" cell: λ=7, margin k=4, cross-language union on
# (synth_l8-12_m4_xu in scripts/run_synth.sh).
PAPER_LAMBDA = 7
PHRASE_MASK_MARGIN = 4
PHRASE_MASK_XLANG_UNION = True

MODEL_NAME = "k2-fsa/OmniVoice"

# Condition key -> (output dir name, synthesis mode, required attention backend).
CONDITIONS: dict[str, dict[str, str]] = {
    "A": {"name": "A_baseline_sdpa", "mode": "baseline", "attn_implementation": "sdpa"},
    "B": {"name": "B_baseline_eager", "mode": "baseline", "attn_implementation": "eager"},
    "C": {"name": "C_lcg_eager", "mode": "lcg", "attn_implementation": "eager"},
}


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--directions", type=str, default="J_to_E,K_to_E",
                    help="Comma-separated manifest 'setup' values to sample from "
                    "(default 'J_to_E,K_to_E').")
    ap.add_argument("--num-per-direction", type=int, default=25,
                    help="Entries per direction; deterministic selection = first N "
                    "entries of each direction in manifest order (default 25).")
    ap.add_argument("--entry-offset", type=int, default=0,
                    help="Skip the first K manifest-order entries of each direction "
                    "before selecting (default 0). Use to benchmark a disjoint "
                    "subset, e.g. --entry-offset 25 --num-per-direction 25.")
    ap.add_argument("--conditions", type=str, default="A,B,C",
                    help="Comma-separated subset of conditions to run, in order "
                    f"(default 'A,B,C'; choices {sorted(CONDITIONS)}).")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Timed-excluded warmup utterances per condition (default 2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=str, default=None,
                    help="Path to manifest JSONL. Default: <repo-root>/benchmark/manifest.jsonl. "
                    "Relative paths are resolved against the repo root.")
    ap.add_argument("--out-root", type=str, default="_artifacts/bench/bench_backend_out",
                    help="Output root; wavs go to <out-root>/<condition>/<entry_id>.wav "
                    "(default 'bench_backend_out'). Relative paths are resolved "
                    "against the repo root.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse the manifest, select the subset, build all args "
                    "namespaces and print the plan without loading the model.")
    args = ap.parse_args()

    args.conditions_list = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in args.conditions_list if c not in CONDITIONS]
    if unknown:
        ap.error(f"--conditions contains unknown keys {unknown}; choose from {sorted(CONDITIONS)}")
    if not args.conditions_list:
        ap.error("--conditions must name at least one condition")
    args.directions_list = [d.strip() for d in args.directions.split(",") if d.strip()]
    if not args.directions_list:
        ap.error("--directions must name at least one direction")
    if args.num_per_direction < 1:
        ap.error("--num-per-direction must be >= 1")
    if args.entry_offset < 0:
        ap.error("--entry-offset must be >= 0")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    return args


def select_entries(entries: list[dict], directions: list[str], num_per_direction: int,
                   entry_offset: int = 0) -> list[dict]:
    """Deterministic subset: manifest-order entries [offset, offset+N) per direction."""
    selected: list[dict] = []
    for direction in directions:
        dir_entries = [e for e in entries if e.get("setup") == direction]
        if not dir_entries:
            raise ValueError(f"No manifest entries with setup={direction!r}")
        window = dir_entries[entry_offset:entry_offset + num_per_direction]
        if len(window) < num_per_direction:
            log.warning("Direction %s yields only %d entries for offset=%d (< %d requested)",
                        direction, len(window), entry_offset, num_per_direction)
        selected.extend(window)
    return selected


def build_condition_namespace(entry: dict, cond: str, out_dir: Path, seed: int) -> SimpleNamespace:
    """Mirror synth/synth.py's build_args_namespace for each condition.

    Conditions A/B reuse the same namespace (the baseline decode ignores the
    LCG-specific fields); only ``attn_implementation`` is overridden so the
    record reflects the backend the model was actually loaded with.
    """
    spec = CONDITIONS[cond]
    if spec["mode"] == "lcg":
        ns = build_args_namespace(
            entry, PAPER_LAMBDA, out_dir, seed,
            attention_setup=ATTENTION_SETUP,
            phrase_mask_margin=PHRASE_MASK_MARGIN,
            phrase_mask_xlang_union=PHRASE_MASK_XLANG_UNION,
        )
    else:
        ns = build_args_namespace(
            entry, PAPER_LAMBDA, out_dir, seed,
            attention_setup=ATTENTION_SETUP,
        )
    ns.attn_implementation = spec["attn_implementation"]
    return ns


# ---------------------------------------------------------------------------
# Per-utterance synthesis (the timed region). ``eng`` is the imported
# synth/dlm_sampling_experiments module (heavy import deferred for --dry-run).
# ---------------------------------------------------------------------------


def synth_baseline(model, eng, args_ns: SimpleNamespace, gen_config, voice_prompt):
    """No-CS synthesis: the exact path run_cs_attention_plc uses to emit
    baseline_src_only.wav (vanilla CFG iterative_decode over c_src + u)."""
    src_item = eng.prepare_single_item(
        model, args_ns.text, args_ns, voice_prompt, language=args_ns.language
    )
    eng.set_seed(args_ns.seed)
    src_only = eng.iterative_decode(model, src_item, gen_config)
    audio = eng.decode_tokens(model, src_only.tokens, src_item.ref_rms, gen_config)
    return audio, {"target_len": int(src_item.target_len)}


def synth_lcg(model, eng, args_ns: SimpleNamespace, gen_config, voice_prompt):
    """Phrase-Localized LCG synthesis: mirrors run_cs_attention_plc's setup +
    variant-2 (iterative_decode_with_attention_plc) path, skipping the
    baseline byproduct and the metrics/tokens.pt dumps."""
    insert_phrases = eng.parse_delimited_list(args_ns.insert_phrases)
    if not insert_phrases:
        raise ValueError("insert_phrases is required for the LCG condition")
    phrase_lang = args_ns.phrase_lang
    if not phrase_lang:
        phrase_lang = eng._detect_phrase_language_auto(insert_phrases, args_ns.language)

    src_item = eng.prepare_single_item(
        model, args_ns.text, args_ns, voice_prompt, language=args_ns.language
    )
    phr_text = args_ns.text_phr if getattr(args_ns, "text_phr", None) else args_ns.text
    phr_item_base = eng.prepare_single_item(
        model, phr_text, args_ns, voice_prompt, language=phrase_lang
    )
    t_len = int(src_item.target_len)
    phr_item = eng._task_with_target_len(phr_item_base, target_len=t_len)

    # Phrase char spans (in full_text_for_tokens = ref_text + " " + text).
    full_text_for_tokens = eng._combine_text(text=args_ns.text, ref_text=src_item.ref_text)
    phrase_char_spans, merged_spans, missing_phrases = eng.collect_phrase_char_spans(
        full_text_for_tokens,
        insert_phrases,
        ignore_case=bool(args_ns.phrase_ignore_case),
    )
    if not merged_spans:
        raise ValueError(
            f"No phrases found in combined text. Phrases={insert_phrases!r}"
        )
    if missing_phrases:
        log.warning("Phrases not located in combined ref+text: %s", missing_phrases)

    # Cond-branch token layout (aligns attention slice + phrase mask).
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
    wrapped_text_token_len = eng._length_of_tokens_for_text(wrapped_full_text, model.text_tokenizer)
    text_token_offset = style_token_len
    text_token_end = text_token_offset + wrapped_text_token_len

    phrase_token_ranges: list[tuple[int, int]] = []
    for _ph, char_spans in phrase_char_spans.items():
        for cs in char_spans:
            p_start, p_end = eng._map_phrase_char_span_to_token_range(
                full_text_for_tokens, cs, model.text_tokenizer
            )
            p_start = max(0, min(wrapped_text_token_len, p_start))
            p_end = max(p_start + 1, min(wrapped_text_token_len, p_end))
            phrase_token_ranges.append((p_start, p_end))
    phrase_text_mask = eng._build_phrase_text_mask(phrase_token_ranges, wrapped_text_token_len)

    eng.set_seed(args_ns.seed)
    final_tokens, _masks_per_step = eng.iterative_decode_with_attention_plc(
        model,
        src_item,
        phr_item,
        gen_config,
        mode="langdir",
        attention_setup=args_ns.attention_setup,
        phrase_text_mask=phrase_text_mask,
        text_token_offset=text_token_offset,
        text_token_end=text_token_end,
        lang_strength=float(args_ns.lang_cfg_strength),
        gt_audio_mask_1d=None,
        phrase_mask_soft=bool(args_ns.phrase_mask_soft),
        phrase_mask_topk=int(args_ns.phrase_mask_topk),
        phrase_mask_margin=int(args_ns.phrase_mask_margin),
        phrase_mask_xlang_union=bool(args_ns.phrase_mask_xlang_union),
        phrase_mask_full=bool(args_ns.phrase_mask_full),
        lang_direction_base=str(args_ns.lang_direction_base),
    )
    audio = eng.decode_tokens(model, final_tokens, src_item.ref_rms, gen_config)
    return audio, {"target_len": t_len}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def condition_stats(records: list[dict]) -> Optional[dict]:
    if not records:
        return None
    rtfs = [r["rtf"] for r in records]
    synth = [r["synth_sec"] for r in records]
    audio = [r["audio_sec"] for r in records]
    return {
        "n": len(records),
        "mean_rtf": statistics.fmean(rtfs),
        "median_rtf": statistics.median(rtfs),
        "mean_synth_sec": statistics.fmean(synth),
        "median_synth_sec": statistics.median(synth),
        "mean_audio_sec": statistics.fmean(audio),
        "total_synth_sec": sum(synth),
        "total_audio_sec": sum(audio),
        # Duration-weighted RTF: total synth time / total audio time.
        "pooled_rtf": sum(synth) / max(sum(audio), 1e-9),
    }


def stat_ratios(num: dict, den: dict) -> dict:
    keys = ("mean_rtf", "median_rtf", "pooled_rtf", "mean_synth_sec")
    return {k: num[k] / den[k] for k in keys if den.get(k)}


def main() -> int:
    cli = parse_cli()

    out_root = Path(cli.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    entries = load_manifest(cli.manifest)
    log.info("Loaded %d manifest entries", len(entries))
    selected = select_entries(entries, cli.directions_list, cli.num_per_direction,
                              cli.entry_offset)
    n_warmup = min(cli.warmup, len(selected))
    log.info(
        "Config: conditions=%s directions=%s num_per_direction=%d (selected=%d) "
        "warmup=%d seed=%d out_root=%s",
        cli.conditions_list, cli.directions_list, cli.num_per_direction,
        len(selected), n_warmup, cli.seed, out_root,
    )
    log.info(
        "Paper cell: attention_setup=%s lambda=%d margin=%d xlang_union=%s | "
        "sampling: dtype=bfloat16 num_step=32 guidance_scale=2.0 t_shift=0.1 "
        "(from synth.build_args_namespace)",
        ATTENTION_SETUP, PAPER_LAMBDA, PHRASE_MASK_MARGIN, PHRASE_MASK_XLANG_UNION,
    )

    # Build every (condition, entry) namespace up front; this validates the
    # manifest fields and is all the --dry-run plan needs.
    plan: dict[str, list[tuple[dict, SimpleNamespace]]] = {}
    for cond in cli.conditions_list:
        cond_dir = out_root / CONDITIONS[cond]["name"]
        plan[cond] = [
            (entry, build_condition_namespace(entry, cond, cond_dir, cli.seed))
            for entry in selected
        ]

    for cond in cli.conditions_list:
        spec = CONDITIONS[cond]
        log.info(
            "PLAN condition %s (%s): mode=%s attn_implementation=%s "
            "entries=%d (+%d warmup) -> %s/",
            cond, spec["name"], spec["mode"], spec["attn_implementation"],
            len(plan[cond]), n_warmup, out_root / spec["name"],
        )

    if cli.dry_run:
        for direction in cli.directions_list:
            ids = [e["id"] for e in selected if e.get("setup") == direction]
            log.info("DRY direction %s: %d entries: %s%s",
                     direction, len(ids), ", ".join(ids[:5]),
                     " ..." if len(ids) > 5 else "")
        for cond in cli.conditions_list:
            entry, ns = plan[cond][0]
            log.info(
                "DRY %s sample namespace: id=%s command=%s attn=%s lambda=%s "
                "margin=%s xlang_union=%s num_step=%s guidance=%s t_shift=%s "
                "dtype=%s lang=%s phrase_lang=%s ref=%s n_phrases=%d",
                cond, entry["id"], ns.command, ns.attn_implementation,
                ns.lang_cfg_strength, ns.phrase_mask_margin,
                ns.phrase_mask_xlang_union, ns.num_step, ns.guidance_scale,
                ns.t_shift, ns.dtype, ns.language, ns.phrase_lang,
                ns.ref_audio, len(entry.get("phrases", [])),
            )
        total = len(cli.conditions_list) * (len(selected) + n_warmup)
        log.info("DRY total synthesis calls = %d (%d measured + %d warmup)",
                 total, len(cli.conditions_list) * len(selected),
                 len(cli.conditions_list) * n_warmup)
        return 0

    # --- Heavy imports + benchmark loop (after dry-run early-exit) ---
    import torch  # noqa: E402
    from omnivoice import OmniVoice  # noqa: E402
    import dlm_sampling_experiments as eng  # noqa: E402
    import soundfile as sf  # noqa: E402

    device = eng.get_best_device()
    dtype = eng.resolve_dtype("bfloat16", device)
    use_cuda = device.startswith("cuda")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else device

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    eng.ensure_dir(out_root)
    results_path = out_root / "results.jsonl"
    summary_path = out_root / "summary.json"
    results_f = results_path.open("w", encoding="utf-8")

    synth_fns = {"baseline": synth_baseline, "lcg": synth_lcg}
    per_condition_records: dict[str, list[dict]] = {}
    backend_report: dict[str, dict] = {}
    errors: list[dict] = []

    model = None
    loaded_impl: Optional[str] = None
    for cond in cli.conditions_list:
        spec = CONDITIONS[cond]
        want_impl = spec["attn_implementation"]
        cond_dir = eng.ensure_dir(out_root / spec["name"])
        synth_fn = synth_fns[spec["mode"]]

        # (Re)load the model only when the required backend changes (B and C
        # share one eager load).
        #
        # The OmniVoice wrapper class only accepts attn_implementation='eager'
        # at load time (it declares _supports_flex_attn/_supports_flash_attn_2
        # but not _supports_sdpa, so transformers rejects a top-level 'sdpa').
        # All attention compute happens inside ``model.llm`` (a Qwen3 stack;
        # OmniVoice.forward routes through self.llm), so the production-SDPA
        # condition is realized by switching the LLM submodule after load via
        # ``model.llm.set_attn_implementation('sdpa')`` and verified below on
        # ``model.llm.config._attn_implementation``.
        if model is None or loaded_impl != want_impl:
            if model is not None:
                del model
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
            log.info("[%s] Loading %s on device=%s dtype=%s (llm attn_implementation=%s) ...",
                     cond, MODEL_NAME, device, dtype, want_impl)
            t_load = time.time()
            model = OmniVoice.from_pretrained(
                MODEL_NAME,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            if want_impl != "eager":
                model.llm.set_attn_implementation(want_impl)
            loaded_impl = want_impl
            log.info("[%s] Model loaded in %.1fs", cond, time.time() - t_load)

        # Verify the requested backend is actually active on the LLM.
        active_top = getattr(model.config, "_attn_implementation", None)
        active_llm = getattr(getattr(model.llm, "config", None), "_attn_implementation", None)
        log.info("[%s] attn_implementation: requested=%s model.config=%s model.llm.config=%s",
                 cond, want_impl, active_top, active_llm)
        if active_llm is not None and active_llm != want_impl:
            log.error("[%s] Backend mismatch: requested %s but LLM reports %s -- "
                      "timings for this condition are NOT valid.", cond, want_impl, active_llm)
        backend_report[cond] = {
            "requested": want_impl,
            "model_config": active_top,
            "llm_config": active_llm,
        }

        # Voice prompts are cached per (ref_audio, ref_text) and prepared
        # OUTSIDE the timed region (per model instance).
        voice_prompt_cache: dict[tuple[str, Optional[str]], object] = {}
        for entry, ns in plan[cond]:
            key = (ns.ref_audio, ns.ref_text)
            if key not in voice_prompt_cache:
                voice_prompt_cache[key] = eng.make_voice_prompt(model, ns)
                log.info("[%s] Cached voice prompt for %s", cond, ns.ref_audio)

        # Warmup (timed-excluded; wavs not saved).
        for w, (entry, ns) in enumerate(plan[cond][:n_warmup]):
            ns = SimpleNamespace(**vars(ns))  # do not mutate the plan copy
            ns.device = device
            gen_config = eng.build_generation_config(ns)
            voice_prompt = voice_prompt_cache[(ns.ref_audio, ns.ref_text)]
            t0 = time.time()
            try:
                synth_fn(model, eng, ns, gen_config, voice_prompt)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] warmup %d id=%s FAILED: %s", cond, w, entry["id"], e)
            log.info("[%s] warmup %d/%d id=%s done in %.2fs (excluded)",
                     cond, w + 1, n_warmup, entry["id"], time.time() - t0)

        # Measured loop.
        records: list[dict] = []
        for i, (entry, ns) in enumerate(plan[cond]):
            ns.device = device
            entry_id = entry["id"]
            wav_path = cond_dir / f"{entry_id}.wav"
            try:
                gen_config = eng.build_generation_config(ns)
                voice_prompt = voice_prompt_cache[(ns.ref_audio, ns.ref_text)]

                sync()
                t_start = time.perf_counter()
                audio, extras = synth_fn(model, eng, ns, gen_config, voice_prompt)
                sync()
                synth_sec = time.perf_counter() - t_start

                eng._save_audio(wav_path, audio, model.sampling_rate)
                info = sf.info(str(wav_path))
                audio_sec = info.frames / float(info.samplerate)
                rtf = synth_sec / max(audio_sec, 1e-9)
                try:
                    wav_rel = str(wav_path.relative_to(REPO_ROOT))
                except ValueError:  # --out-root outside the repo
                    wav_rel = str(wav_path)

                rec = {
                    "condition": cond,
                    "condition_name": spec["name"],
                    "mode": spec["mode"],
                    "attn_implementation": want_impl,
                    "entry_id": entry_id,
                    "direction": entry.get("setup"),
                    "language": entry.get("language"),
                    "phrase_lang": entry.get("phrase_lang"),
                    "n_phrases": len(entry.get("phrases", [])),
                    "target_len": extras.get("target_len"),
                    "synth_sec": synth_sec,
                    "audio_sec": audio_sec,
                    "rtf": rtf,
                    "wav": wav_rel,
                }
                records.append(rec)
                results_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results_f.flush()
                log.info("[%s] %d/%d id=%s synth=%.2fs audio=%.2fs rtf=%.3f",
                         cond, i + 1, len(plan[cond]), entry_id, synth_sec, audio_sec, rtf)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("[%s] id=%s FAILED: %s\n%s", cond, entry_id, e, tb)
                errors.append({"condition": cond, "entry_id": entry_id, "error": str(e)})
        per_condition_records[cond] = records

    results_f.close()

    # --- Summary ---
    stats = {c: condition_stats(per_condition_records.get(c, [])) for c in cli.conditions_list}
    ratios: dict[str, dict] = {}
    for num, den in (("B", "A"), ("C", "A"), ("C", "B")):
        if stats.get(num) and stats.get(den):
            ratios[f"{num}_over_{den}"] = stat_ratios(stats[num], stats[den])

    summary = {
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "device": device,
        "dtype": "bfloat16",
        "num_step": 32,
        "guidance_scale": 2.0,
        "t_shift": 0.1,
        "seed": cli.seed,
        "attention_setup": ATTENTION_SETUP,
        "paper_lambda": PAPER_LAMBDA,
        "phrase_mask_margin": PHRASE_MASK_MARGIN,
        "phrase_mask_xlang_union": PHRASE_MASK_XLANG_UNION,
        "directions": cli.directions_list,
        "num_per_direction": cli.num_per_direction,
        "entry_offset": cli.entry_offset,
        "warmup": n_warmup,
        "conditions": {
            c: {
                "name": CONDITIONS[c]["name"],
                "mode": CONDITIONS[c]["mode"],
                "attn_backend": backend_report.get(c),
                "stats": stats.get(c),
            }
            for c in cli.conditions_list
        },
        "ratios": ratios,
        "errors": errors,
    }
    eng.write_json(summary_path, summary)

    log.info("Results:  %s", results_path)
    log.info("Summary:  %s", summary_path)
    for c in cli.conditions_list:
        s = stats.get(c)
        if not s:
            log.warning("Condition %s produced no successful records", c)
            continue
        log.info(
            "%s (%s): N=%d mean_rtf=%.4f median_rtf=%.4f pooled_rtf=%.4f "
            "mean_synth=%.2fs mean_audio=%.2fs",
            c, CONDITIONS[c]["name"], s["n"], s["mean_rtf"], s["median_rtf"],
            s["pooled_rtf"], s["mean_synth_sec"], s["mean_audio_sec"],
        )
    for name, r in ratios.items():
        log.info("RATIO %s: mean_rtf=%.3f median_rtf=%.3f pooled_rtf=%.3f mean_synth_sec=%.3f",
                 name, r["mean_rtf"], r["median_rtf"], r["pooled_rtf"], r["mean_synth_sec"])
    if errors:
        log.error("%d utterances failed (see summary.json errors[])", len(errors))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

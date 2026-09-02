"""Synthesis runner for the Phrase-Localized LCG benchmark corpus.

Loads OmniVoice once and invokes ``run_cs_attention_plc`` from the bundled
LCG engine (``synth/dlm_sampling_experiments.py``) for each manifest entry,
sweeping the ``LAMBDA_KEYS`` list (default = paper-final 6 λ values: 3, 5,
7, 9, 11, 13). The paper's three reference systems map as:

    no_CS  ->  baseline_src_only.wav                                       (emitted as a byproduct of every λ call)
    Swap   ->  cs_attention_plc__layers_set_8-12__max__a2t__lambda3.wav
    Ours   ->  cs_attention_plc__layers_set_8-12__max__a2t__lambda7.wav

Each paper-λ is dispatched to the engine's LCG path; the wav filename is
keyed solely by paper-λ.

Sharding:  --gpu-shard k --num-shards N processes entries with idx % N == k.
Resume:    an already-written wav > MIN_WAV_BYTES is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

# Resolve repo root (PhraseLocalizedLCG/). This file lives at <repo>/synth/synth.py.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
SYNTH_DIR = THIS_FILE.parent

# Both the omnivoice/ package and the LCG engine module (synth/dlm_sampling_experiments.py
# providing run_cs_attention_plc) need to be importable.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SYNTH_DIR))

# Manifest references "examples/<lang>_ref.wav" with paths relative to the repo root.
os.chdir(REPO_ROOT)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    force=True,
)
log = logging.getLogger("synth")

# Default manifest + output paths (overridable via CLI flags below).
MANIFEST = REPO_ROOT / "benchmark" / "manifest.jsonl"
OUT_ROOT = REPO_ROOT / "synth_l8-12"

ATTENTION_SETUP = "layers_set_8-12__max__a2t"

BASELINE_FILENAME = "baseline_src_only.wav"
LAMBDA_TEMPLATE = f"cs_attention_plc__{ATTENTION_SETUP}__lambda{{lam}}.wav"

MIN_WAV_BYTES = 50 * 1024  # 50 KB resume threshold.


def _str2bool(value: str | bool) -> bool:
    """argparse helper -- accept the usual truthy/falsy strings."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("y", "yes", "true", "t", "1", "on"):
        return True
    if v in ("n", "no", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _auto_out_suffix(cli) -> str:
    """Encode the active phrase-mask toggles into a short suffix tag.

    - phrase_mask_source='gt' short-circuits to 'gt' (the 4 toggles are no-ops).
    - Otherwise the base is 'l8-12' (attention setup tag) with optional
      ``soft``, ``topkK`` (K>1), ``mM`` (M>0), ``xu`` joined by ``_``.
    - Defaults (no flag active) → 'l8-12'.
    """
    if cli.phrase_mask_source == "gt":
        return "gt"
    parts: list[str] = ["l8-12"]
    if bool(cli.phrase_mask_soft):
        parts.append("soft")
    if int(cli.phrase_mask_topk) > 1:
        parts.append(f"topk{int(cli.phrase_mask_topk)}")
    if int(cli.phrase_mask_margin) > 0:
        parts.append(f"m{int(cli.phrase_mask_margin)}")
    if bool(cli.phrase_mask_xlang_union):
        parts.append("xu")
    return "_".join(parts)


# Paper-λ → engine dispatch. With a hard phrase mask (paper-final setup), the
# engine's two algorithmic variants (langdir vs hybrid) produce bit-exact
# identical audio whenever λ_paper coincides — hybrid_K ≡ langdir_(K+3). The
# repo therefore exposes a single langdir-based path with K = λ_paper, and
# `LAMBDA_KEYS` simply enumerates supported λ values:
#   • λ=3   → paper "Swap"  (minimal language steering)
#   • λ=7   → paper "Ours"  (Phrase-Localized LCG headline result)
#   • others are intermediate / high-λ sweep points reported in the figures.
# The wav filename is always ``cs_attention_plc__<setup>__lambda{λ:g}.wav``;
# baseline_src_only.wav is auto-emitted as a byproduct of every call.
LAMBDA_KEYS: list[int] = [3, 5, 7, 9, 11, 13]


def expected_filenames_for_lambda(lam: int) -> list[str]:
    """Wav filenames the engine writes for paper-λ = ``lam`` (used for resume check).
    Engine always emits baseline_src_only.wav alongside the per-λ wav."""
    if lam < 1:
        raise ValueError(f"Unsupported λ={lam}; expected a positive integer.")
    return [LAMBDA_TEMPLATE.format(lam=lam)]


def load_manifest(manifest_path: Path | None = None) -> list[dict]:
    p = Path(manifest_path) if manifest_path else MANIFEST
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    entries: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def wav_present(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > MIN_WAV_BYTES
    except OSError:
        return False


def all_outputs_present(out_dir: Path, lam: int) -> bool:
    return all(wav_present(out_dir / n) for n in expected_filenames_for_lambda(lam))


def build_args_namespace(
    entry: dict,
    lam: int,
    out_dir: Path,
    seed: int,
    *,
    attention_setup: str,
    phrase_mask_source: str = "attention",
    gt_alignment_file: Optional[str] = None,
    edit_aligner_python: Optional[str] = None,
    phrase_mask_soft: bool = False,
    phrase_mask_topk: int = 1,
    phrase_mask_margin: int = 0,
    phrase_mask_xlang_union: bool = False,
    phrase_mask_full: bool = False,
    lang_direction_base: str = "src",
) -> SimpleNamespace:
    """Construct the argparse.Namespace expected by run_cs_attention_plc."""
    command = "cs-attention-plc-langdir"
    lang_cfg_strength = float(lam)
    # Phrases joined with the CLI's "||" delimiter.
    phrases = entry.get("phrases", [])
    insert_phrases = "||".join(phrases)
    # Mirror the defaults declared in build_parser() for cs-attention-plc{,-langdir,-hybrid}.
    ns = SimpleNamespace(
        command=command,
        # Shared model + sampling args.
        model="k2-fsa/OmniVoice",
        device=None,             # filled by caller -> resolved device string
        dtype="bfloat16",
        output_dir=str(out_dir),
        seed=seed,
        language=entry["language"],
        ref_audio=entry["ref_audio_path"],
        ref_text=entry.get("ref_text"),
        instruct=None,
        speed=None,
        duration=None,
        num_step=32,
        guidance_scale=2.0,
        t_shift=0.1,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
        layer_penalty_factor=5.0,
        position_temperature=5.0,
        class_temperature=0.0,
        audio_chunk_duration=15.0,
        audio_chunk_threshold=30.0,
        # cs-attention-plc args.
        # NOTE: text_phr=None on purpose. The original design uses the SAME `text`
        # for both c_src and c_phr branches; only the lang tag differs
        # (lang=language on c_src, lang=phrase_lang on c_phr). The manifest's
        # `text_phr` field (base-lang-translated prose) is retained for analysis
        # but not used in synthesis.
        text=entry["text"],
        text_phr=None,
        insert_phrases=insert_phrases,
        phrase_lang=entry.get("phrase_lang"),
        phrase_ignore_case=True,
        attention_setup=attention_setup,
        lang_cfg_strength=lang_cfg_strength,
        attn_implementation="eager",
        # Phrase-mask source: 'attention' (runtime) or 'gt' (precomputed JSONL).
        phrase_mask_source=phrase_mask_source,
        gt_alignment_file=gt_alignment_file,
        entry_id=entry["id"],
        # Phrase-mask toggle flags (no-ops when phrase_mask_source='gt').
        phrase_mask_soft=bool(phrase_mask_soft),
        phrase_mask_topk=int(phrase_mask_topk),
        phrase_mask_margin=int(phrase_mask_margin),
        phrase_mask_xlang_union=bool(phrase_mask_xlang_union),
        phrase_mask_full=bool(phrase_mask_full),
        lang_direction_base=str(lang_direction_base),
        # In-synthesis FA evaluation (mask-vs-forced-alignment recall/IoU stats).
        # Always off here: it is slow and the reported metrics all come from the
        # standalone `python -m eval ...` stage. The underlying engine still
        # exposes it as `--eval-with-fa` in dlm_sampling_experiments.py.
        eval_with_fa=False,
        edit_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
        edit_aligner_conda_env="qwen3-asr",
        edit_aligner_conda_executable="conda",
        edit_aligner_python=edit_aligner_python,
        edit_aligner_cuda_visible_devices=None,
        edit_aligner_timeout_sec=0.0,
        edit_aligner_language=None,
        edit_aligner_device=None,
        edit_aligner_dtype="bfloat16",
        edit_aligner_attn_implementation=None,
        edit_aligner_margin_sec=0.0,
    )
    return ns


def rename_metrics_and_tokens(out_dir: Path, lam: int) -> None:
    """Rename the metrics.json / tokens.pt written by run_cs_attention_plc so
    that subsequent λ calls do not overwrite them."""
    tag = f"lambda{lam}"
    for stem, ext in (("metrics", ".json"), ("tokens", ".pt")):
        src = out_dir / f"{stem}{ext}"
        if src.exists():
            dst = out_dir / f"{stem}__{tag}{ext}"
            try:
                src.replace(dst)
            except OSError as e:
                log.warning("Could not rename %s -> %s: %s", src, dst, e)


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=8,
                    help="Total number of shards across GPUs (default 8 for B200 x 8).")
    ap.add_argument("--gpu-shard", type=int, required=True,
                    help="This shard's index in [0, num_shards). Entries with idx %% num_shards == gpu_shard are processed.")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Path to manifest JSONL. Default: <repo-root>/benchmark/manifest.jsonl. "
                    "Relative paths are resolved against the repo root.")
    ap.add_argument("--out-root", type=str, default=None,
                    help="Override the per-cell output directory entirely. When set, "
                    "wavs go to <out-root>/<entry_id>/. Overrides --out-suffix. "
                    "Relative paths are resolved against the repo root.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only process first N entries of this shard (sanity check).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned (entry, mode) work list and exit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-name", type=str, default=None,
                    help="Override the [GPU<N>] log prefix; otherwise uses shard idx.")
    ap.add_argument("--attention-setup", type=str, default=ATTENTION_SETUP,
                    help="Attention setup string. Drives output filenames and the runtime "
                    "attention mask. Ignored as a mask source when --phrase-mask-source=gt.")
    ap.add_argument("--phrase-mask-source", type=str, choices=["attention", "gt"],
                    default="attention",
                    help="Source of the per-step phrase audio mask passed to "
                    "iterative_decode_with_attention_plc.")
    ap.add_argument("--gt-alignment-file", type=str, default=None,
                    help="Path to a JSONL with one record per manifest entry "
                    '{"id", "items":[{"text","start_time","end_time"}, ...]}. '
                    "Required when --phrase-mask-source=gt.")
    ap.add_argument("--out-suffix", type=str, default=None,
                    help="Suffix for OUT_ROOT (= REPO_ROOT/synth_<suffix>). "
                    "Default: 'gt' when --phrase-mask-source=gt; otherwise "
                    "'l8-12' plus encoded toggle tags (e.g. 'l8-12_soft_m4').")
    ap.add_argument("--phrase-mask-soft", type=_str2bool, default=False,
                    help="Use soft (probability-mass) phrase mask. Overrides "
                    "--phrase-mask-topk. Ignored when --phrase-mask-source=gt.")
    ap.add_argument("--phrase-mask-topk", type=int, default=1,
                    help="Top-k union over attended text tokens (k=1 = argmax). "
                    "Ignored when --phrase-mask-soft=True or --phrase-mask-source=gt.")
    ap.add_argument("--phrase-mask-margin", type=int, default=0,
                    help="Number of graded dilation rounds k applied to the "
                    "per-step phrase mask; round s widens it by +-s frames, for "
                    "an effective radius k(k+1)/2 (k=4 -> +-10 frames, the "
                    "paper-final setting). 0 = off. Ignored when "
                    "--phrase-mask-source=gt.")
    ap.add_argument("--phrase-mask-xlang-union", type=_str2bool, default=False,
                    help="Combine c_src AND c_phr attention masks (logical OR / max). "
                    "Ignored when --phrase-mask-source=gt.")
    ap.add_argument("--phrase-mask-full", type=_str2bool, default=False,
                    help="Override the per-step phrase mask with all-1s (= apply "
                    "language guiding term everywhere). Ablation: tests whether "
                    "mask localization is necessary. Supersedes margin / xu / "
                    "topk / soft flags. Ignored when --phrase-mask-source=gt.")
    ap.add_argument("--lang-direction-base", type=str, choices=["src", "uncond"],
                    default="src",
                    help="Base for the language guiding direction in "
                    "langdir/hybrid: 'src' (default) = c_phr - c_src; "
                    "'uncond' = c_phr - u (mixes language direction with extra "
                    "CFG strength at phrase positions).")
    ap.add_argument("--lambdas", type=str, default=None,
                    help="Comma-separated subset of paper-λ values to synthesize "
                    "(e.g. '7' for Ours only, or '3,5,7,9,11,13' for the paper sweep). "
                    f"Default: {LAMBDA_KEYS} (paper-final).")
    ap.add_argument(
        "--edit-aligner-python", type=str,
        default=os.environ.get("QWEN_PY", "python"),
        help="Path to the python interpreter that has Qwen3-ForcedAligner "
        "installed (a separate env from OmniVoice). Defaults to $QWEN_PY, "
        "falling back to 'python'. Unused by this runner (in-synthesis FA "
        "evaluation is off; see `eval_with_fa` below) and kept only so the "
        "value is recorded in the run config.",
    )
    args = ap.parse_args()
    if not 0 <= args.gpu_shard < args.num_shards:
        ap.error(f"--gpu-shard must be in [0, {args.num_shards}); got {args.gpu_shard}")
    if args.phrase_mask_source == "gt" and not args.gt_alignment_file:
        ap.error("--phrase-mask-source=gt requires --gt-alignment-file")
    if args.lambdas:
        try:
            requested = [int(x.strip()) for x in args.lambdas.split(",") if x.strip()]
        except ValueError as e:
            ap.error(f"--lambdas expects comma-separated integers: {e}")
        if any(lam < 1 for lam in requested):
            ap.error(f"--lambdas values must be positive integers; got {requested}")
        args.lambdas_list = requested
    else:
        args.lambdas_list = list(LAMBDA_KEYS)
    return args


def main() -> int:
    cli = parse_cli()
    shard = cli.gpu_shard
    num_shards = cli.num_shards
    shard_name = cli.shard_name or f"GPU-shard{shard}of{num_shards}"

    # CLI-driven globals: attention setup, output root, filename template.
    global ATTENTION_SETUP, OUT_ROOT, LAMBDA_TEMPLATE
    ATTENTION_SETUP = cli.attention_setup
    LAMBDA_TEMPLATE = f"cs_attention_plc__{ATTENTION_SETUP}__lambda{{lam}}.wav"
    out_suffix = cli.out_suffix or _auto_out_suffix(cli)
    if getattr(cli, "out_root", None):
        _or = Path(cli.out_root)
        OUT_ROOT = _or if _or.is_absolute() else (REPO_ROOT / cli.out_root)
    else:
        OUT_ROOT = REPO_ROOT / f"synth_{out_suffix}"
    log.info(
        "Config: num_shards=%d shard=%d attention_setup=%s phrase_mask_source=%s out_root=%s gt_file=%s "
        "phrase_mask: soft=%s topk=%d margin=%d xlang_union=%s",
        num_shards, shard, ATTENTION_SETUP, cli.phrase_mask_source, OUT_ROOT,
        cli.gt_alignment_file,
        cli.phrase_mask_soft, cli.phrase_mask_topk,
        cli.phrase_mask_margin, cli.phrase_mask_xlang_union,
    )

    entries = load_manifest(getattr(cli, "manifest", None))
    log.info("Loaded %d manifest entries", len(entries))

    # Shard split: idx % num_shards == shard.
    shard_entries = [(i, e) for i, e in enumerate(entries) if i % num_shards == shard]
    if cli.limit > 0:
        shard_entries = shard_entries[: cli.limit]
    log.info("Shard %d: %d entries", shard, len(shard_entries))

    # Pre-compute (and report) the action plan.
    plan: list[tuple[int, dict, list[int]]] = []
    for idx, entry in shard_entries:
        missing = [lam for lam in cli.lambdas_list
                   if not all_outputs_present(OUT_ROOT / entry["id"], lam)]
        plan.append((idx, entry, missing))

    total_calls = sum(len(m) for _, _, m in plan)
    total_remaining_entries = sum(1 for _, _, m in plan if m)
    log.info(
        "Plan: %d entries (%d not yet complete) -> %d outstanding λ calls "
        "(max possible %d, λ=%s)",
        len(plan), total_remaining_entries, total_calls,
        len(plan) * len(cli.lambdas_list), cli.lambdas_list,
    )

    if cli.dry_run:
        # Print the first few to keep logs short.
        for idx, entry, missing in plan[:5]:
            log.info("DRY idx=%d id=%s setup=%s missing=%s", idx, entry["id"], entry["setup"], missing)
        log.info("DRY total outstanding λ calls = %d", total_calls)
        return 0

    # --- Heavy imports + model load (after dry-run early-exit) ---
    from omnivoice import OmniVoice  # noqa: E402
    from dlm_sampling_experiments import (  # noqa: E402
        build_generation_config,
        ensure_dir,
        get_best_device,
        make_voice_prompt,
        resolve_dtype,
        run_cs_attention_plc,
        set_seed,
    )

    # We use a stable device str regardless of CUDA_VISIBLE_DEVICES because, in
    # this script, CUDA_VISIBLE_DEVICES is set to a SINGLE GPU id at launch.
    device = get_best_device()
    dtype = resolve_dtype("bfloat16", device)
    log.info("[%s] Loading OmniVoice on device=%s dtype=%s ...", shard_name, device, dtype)
    t_load_start = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    log.info("[%s] Model loaded in %.1fs", shard_name, time.time() - t_load_start)

    # Cache voice prompts per (ref_audio, ref_text) tuple to avoid recomputing.
    voice_prompt_cache: dict[tuple[str, str | None], object] = {}

    def get_voice_prompt(ref_audio: str, ref_text: str | None, args: SimpleNamespace):
        key = (ref_audio, ref_text)
        if key not in voice_prompt_cache:
            voice_prompt_cache[key] = make_voice_prompt(model, args)
            log.info("[%s] Cached voice prompt for %s", shard_name, ref_audio)
        return voice_prompt_cache[key]

    # --- Main loop ---
    t_loop_start = time.time()
    entries_done = 0
    calls_done = 0
    errors: list[dict] = []
    total = len(plan)

    for plan_idx, (idx, entry, _) in enumerate(plan):
        entry_id = entry["id"]
        out_dir = ensure_dir(OUT_ROOT / entry_id)
        t_entry = time.time()
        lambdas_skipped = 0
        lambdas_run = 0
        for lam in cli.lambdas_list:
            if all_outputs_present(out_dir, lam):
                lambdas_skipped += 1
                continue
            args_ns = build_args_namespace(
                entry, lam, out_dir, cli.seed,
                attention_setup=ATTENTION_SETUP,
                phrase_mask_source=cli.phrase_mask_source,
                gt_alignment_file=cli.gt_alignment_file,
                edit_aligner_python=cli.edit_aligner_python,
                phrase_mask_soft=cli.phrase_mask_soft,
                phrase_mask_topk=cli.phrase_mask_topk,
                phrase_mask_margin=cli.phrase_mask_margin,
                phrase_mask_xlang_union=cli.phrase_mask_xlang_union,
                phrase_mask_full=cli.phrase_mask_full,
                lang_direction_base=cli.lang_direction_base,
            )
            args_ns.device = device
            try:
                set_seed(args_ns.seed)
                gen_config = build_generation_config(args_ns)
                voice_prompt = get_voice_prompt(args_ns.ref_audio, args_ns.ref_text, args_ns)
                run_cs_attention_plc(model, args_ns, gen_config, voice_prompt, Path(args_ns.output_dir))
                rename_metrics_and_tokens(Path(args_ns.output_dir), lam)
                lambdas_run += 1
                calls_done += 1
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("[%s] entry=%s λ=%s FAILED: %s\n%s", shard_name, entry_id, lam, e, tb)
                errors.append({"entry_id": entry_id, "lambda": lam, "error": str(e)})

        entries_done += 1
        elapsed_entry = time.time() - t_entry
        elapsed_total = time.time() - t_loop_start
        rate = entries_done / max(elapsed_total, 1e-6)
        remaining = total - entries_done
        eta_sec = remaining / max(rate, 1e-9)
        log.info(
            "[%s] entry %d/%d id=%s setup=%s lambdas_run=%d lambdas_skipped=%d elapsed_s=%.1f total_s=%.1f eta_h=%.2f",
            shard_name, entries_done, total, entry_id, entry.get("setup"),
            lambdas_run, lambdas_skipped, elapsed_entry, elapsed_total, eta_sec / 3600.0,
        )

        if entries_done % 10 == 0:
            log.info(
                "[%s] PROGRESS %d/%d entries | %d/%d calls done | %.1f min elapsed | ETA %.2f h",
                shard_name, entries_done, total, calls_done, total_calls,
                elapsed_total / 60.0, eta_sec / 3600.0,
            )

    log.info(
        "[%s] DONE entries=%d calls=%d errors=%d total_time_min=%.1f",
        shard_name, entries_done, calls_done, len(errors),
        (time.time() - t_loop_start) / 60.0,
    )
    if errors:
        err_path = REPO_ROOT / f"synth_errors_shard{shard}.json"
        err_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        log.error("[%s] %d errors written to %s", shard_name, len(errors), err_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""Standalone UTMOS scorer for a synth directory.

Walks every ``<synth_dir>/<entry_id>/*.wav`` (skipping ref / non-content files),
loads each into 16 kHz mono, runs torch.hub UTMOS22Strong, and writes one JSON
record per wav to ``<utmos_dir>/utmos.shard<N>.jsonl``. Resumable via
``--skip-existing``: any wav whose key already appears in the shard JSONL is
skipped.

Output schema (one JSON per line):
    {
      "entry_id":   "syncs_E_to_K_0",
      "mode":       "baseline_src_only",          # = wav stem
      "wav_path":   "synth_l8-12_m4/syncs_E_to_K_0/baseline_src_only.wav",
      "utmos":      3.4517,
      "audio_s":    5.32,
    }

Why standalone: UTMOS depends on whole-audio quality and has nothing to do with
phrase spans, FA, or any of the Whisper passes. Decoupling keeps the FA eval
pipeline untouched (re-eval not required) while letting UTMOS run in parallel
across 8 GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), len(audio) / SAMPLE_RATE


def discover_wavs(synth_dir: Path) -> list[tuple[str, str, Path]]:
    """Return list of (entry_id, mode_stem, wav_path) for every wav under
    synth_dir/<entry_id>/<stem>.wav. Sorted for deterministic sharding."""
    out: list[tuple[str, str, Path]] = []
    if not synth_dir.is_dir():
        return out
    for entry_dir in sorted(synth_dir.iterdir()):
        if not entry_dir.is_dir():
            continue
        for wav in sorted(entry_dir.glob("*.wav")):
            if wav.name.startswith("."):
                continue
            out.append((entry_dir.name, wav.stem, wav))
    return out


def load_existing_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file():
        return set()
    keys: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(f"{rec['entry_id']}::{rec['mode']}")
            except Exception:
                continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=str, required=True,
                    help="Synth dir (relative to repo root or absolute).")
    ap.add_argument("--utmos-dir", type=str, default=None,
                    help="Output dir for per-shard JSONLs. "
                         "Default: utmos_<suffix>/ where synth='synth_<suffix>'.")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--gpu-shard", type=int, required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--shard-name", type=str, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip wavs already scored in the shard JSONL (default on).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--max-audio-s", type=float, default=30.0,
                    help="Truncate audio to this many seconds before scoring.")
    ap.add_argument("--utmos-hub-id", type=str, default="tarepan/SpeechMOS:v1.2.0")
    ap.add_argument("--utmos-model", type=str, default="utmos22_strong")
    args = ap.parse_args()

    if not 0 <= args.gpu_shard < args.num_shards:
        ap.error(f"--gpu-shard must be in [0, {args.num_shards}); got {args.gpu_shard}")

    tag = f"[{args.shard_name}] " if args.shard_name else f"[shard{args.gpu_shard}] "

    # Resolve dirs
    synth_path = Path(args.synth_dir)
    synth_dir = synth_path if synth_path.is_absolute() else REPO_ROOT / args.synth_dir
    if not synth_dir.is_dir():
        print(f"{tag}ERROR: synth_dir not found: {synth_dir}", file=sys.stderr)
        return 2

    if args.utmos_dir:
        utmos_path = Path(args.utmos_dir)
        utmos_dir = utmos_path if utmos_path.is_absolute() else REPO_ROOT / args.utmos_dir
    else:
        # Derive 'utmos_<suffix>' from 'synth_<suffix>'.
        name = synth_dir.name
        if name.startswith("synth_"):
            utmos_dir = synth_dir.parent / ("utmos_" + name[len("synth_"):])
        elif name == "synth":
            utmos_dir = synth_dir.parent / "utmos"
        else:
            utmos_dir = synth_dir.parent / ("utmos_" + name)
    utmos_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = utmos_dir / f"utmos.shard{args.gpu_shard}.jsonl"

    print(f"{tag}synth_dir={synth_dir}", flush=True)
    print(f"{tag}utmos_dir={utmos_dir}  output={out_jsonl.name}", flush=True)

    # Discover + shard
    all_wavs = discover_wavs(synth_dir)
    print(f"{tag}discovered {len(all_wavs)} wavs", flush=True)
    my_wavs = [t for i, t in enumerate(all_wavs) if i % args.num_shards == args.gpu_shard]
    print(f"{tag}shard {args.gpu_shard}/{args.num_shards}: {len(my_wavs)} wavs", flush=True)

    # Resume
    existing = load_existing_keys(out_jsonl) if args.skip_existing else set()
    if existing:
        before = len(my_wavs)
        my_wavs = [(eid, mode, p) for (eid, mode, p) in my_wavs
                   if f"{eid}::{mode}" not in existing]
        print(f"{tag}skip-existing: {before - len(my_wavs)} already scored; {len(my_wavs)} to go",
              flush=True)

    if not my_wavs:
        print(f"{tag}nothing to do", flush=True)
        return 0

    # Load UTMOS
    device = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"
    print(f"{tag}[{time.strftime('%H:%M:%S')}] loading UTMOS on {device} ...", flush=True)
    t0 = time.time()
    predictor = torch.hub.load(args.utmos_hub_id, args.utmos_model,
                                trust_repo=True, source="github")
    predictor = predictor.to(device).eval()
    print(f"{tag}UTMOS loaded in {time.time()-t0:.1f}s", flush=True)

    max_samples = int(args.max_audio_s * SAMPLE_RATE)
    n_done = 0
    n_err = 0
    t_loop = time.time()
    with out_jsonl.open("a", encoding="utf-8") as fout:
        for eid, mode, wav_path in my_wavs:
            try:
                audio, dur = load_audio(wav_path)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
                with torch.inference_mode():
                    score = predictor(wav_t, SAMPLE_RATE)
                score_f = float(score.detach().cpu().squeeze().item()) if hasattr(score, "detach") else float(score)
                rec = {
                    "entry_id": eid,
                    "mode": mode,
                    "wav_path": str(wav_path.relative_to(REPO_ROOT)) if str(wav_path).startswith(str(REPO_ROOT)) else str(wav_path),
                    "utmos": score_f,
                    "audio_s": dur,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 100 == 0:
                    rate = n_done / max(time.time() - t_loop, 1e-6)
                    eta_s = (len(my_wavs) - n_done) / max(rate, 1e-6)
                    print(f"{tag}{n_done}/{len(my_wavs)}  {rate:.1f} wav/s  ETA {eta_s/60:.1f} min",
                          flush=True)
            except Exception as e:
                n_err += 1
                print(f"{tag}ERR {eid}/{mode}: {type(e).__name__}: {e}", flush=True)
    print(f"{tag}DONE in {time.time()-t_loop:.1f}s | {n_done} scored | {n_err} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

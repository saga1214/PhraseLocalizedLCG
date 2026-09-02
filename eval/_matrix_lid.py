"""Matrix-region LID scorer.

Mirror of eval_extended.py's Pass E (phrase-segment LID) but applied to the
*matrix* region (= full wav with phrase regions removed). For each wav, we
look up the phrase_time_spans from the existing FA-eval per-wav JSON, cut the
non-phrase audio (using the same cut_matrix_audio logic as eval_extended.py),
pad to Whisper's 30-second window, and run a single-token language posterior
forward.

Output schema (one JSON per line):
    {
      "entry_id": "...", "mode": "...",
      "base_lang": "en", "phr_lang": "ko",
      "matrix_LID_base":  <prob of base_lang>,
      "matrix_LID_phr":   <prob of phr_lang>,
      "matrix_LID_top":   "<2-letter code of the top language>",
      "matrix_LID_top_p": <prob of the top language>,
      "matrix_audio_s":   <duration of cut matrix audio in seconds>
    }

Why standalone: the existing eval_extended.py already runs Whisper LID on the
*phrase* segment. Re-running its full pipeline to add a matrix-LID pass would
be slow (FA + 4 other Whisper passes + WavLM all redundant). This script
reuses the existing FA spans cached in the per-wav JSONs and runs only the
LID forward on the cut matrix audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
WHISPER_AUDIO_SAMPLES = 30 * SAMPLE_RATE
MIN_SEG_S = 0.5
MODEL_ID = "openai/whisper-large-v3"
DTYPE = torch.float16
LANG_CODES = ["en", "ko", "ja", "zh", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "yue"]


def load_audio(path: Path):
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), len(audio) / SAMPLE_RATE


def pad30(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WHISPER_AUDIO_SAMPLES:
        pad = np.zeros(WHISPER_AUDIO_SAMPLES - len(audio), dtype=audio.dtype)
        return np.concatenate([audio, pad])
    return audio[:WHISPER_AUDIO_SAMPLES]


def cut_matrix_audio(audio: np.ndarray, phrase_spans) -> np.ndarray:
    """Concatenate non-phrase pieces. Mirror of eval_extended.py:cut_matrix_audio."""
    if not phrase_spans:
        return audio.astype(np.float32)
    sp = sorted(phrase_spans)
    pieces = []
    cur = 0.0
    dur = len(audio) / SAMPLE_RATE
    for s, e in sp:
        s = max(cur, s)
        if s > cur:
            i0 = int(cur * SAMPLE_RATE)
            i1 = int(s * SAMPLE_RATE)
            if i1 > i0:
                pieces.append(audio[i0:i1])
        cur = max(cur, e)
    if cur < dur:
        i0 = int(cur * SAMPLE_RATE)
        pieces.append(audio[i0:])
    if not pieces:
        return audio[: int(MIN_SEG_S * SAMPLE_RATE)]
    out = np.concatenate(pieces).astype(np.float32)
    if len(out) < int(MIN_SEG_S * SAMPLE_RATE):
        pad = np.zeros(int(MIN_SEG_S * SAMPLE_RATE) - len(out), dtype=np.float32)
        out = np.concatenate([out, pad])
    return out


class WhisperLIDEngine:
    def __init__(self, device: str = "cuda:0", dtype=DTYPE, model_id: str = MODEL_ID):
        self.device = device
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype
        ).to(device).eval()
        tok = self.processor.tokenizer
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.all_lang_ids: list[int] = []
        for tid in range(50000, 52000):
            t = tok.convert_ids_to_tokens(tid)
            if (
                isinstance(t, str) and len(t) == 6 and t.startswith("<|") and
                t.endswith("|>") and t[2:4].isalpha()
            ):
                self.all_lang_ids.append(tid)
        self.code_to_lang_id = {
            c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in LANG_CODES
        }

    @torch.inference_mode()
    def language_posteriors_batch(self, audios: List[np.ndarray]) -> List[Dict[str, float]]:
        padded = [pad30(a) for a in audios]
        inputs = self.processor(padded, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(self.device).to(self.dtype)
        B = feats.shape[0]
        dec_in = torch.full((B, 1), self.sot_id, dtype=torch.long, device=self.device)
        out = self.model(feats, decoder_input_ids=dec_in)
        logits = out.logits[:, -1, :].float()
        mask = torch.full_like(logits, float("-inf"))
        mask[:, self.all_lang_ids] = logits[:, self.all_lang_ids]
        probs = torch.softmax(mask, dim=-1)
        results = []
        for i in range(B):
            d = {c: float(probs[i, lid].item()) for c, lid in self.code_to_lang_id.items()}
            top_id = int(torch.argmax(probs[i]).item())
            top_tok = self.processor.tokenizer.convert_ids_to_tokens(top_id)
            d["_top"] = top_tok[2:4] if isinstance(top_tok, str) and len(top_tok) == 6 else "?"
            d["_top_p"] = float(probs[i, top_id].item())
            results.append(d)
        return results


def discover_tasks(synth_dir: Path, eval_dir: Path) -> list[dict]:
    """Walk synth_dir/<entry>/*.wav. For each wav, look up phrase_time_spans
    + base_lang + phr_lang from eval_dir/extended/<entry>__<stem>.json."""
    tasks = []
    if not synth_dir.is_dir() or not eval_dir.is_dir():
        return tasks
    for entry_dir in sorted(synth_dir.iterdir()):
        if not entry_dir.is_dir():
            continue
        entry_id = entry_dir.name
        for wav in sorted(entry_dir.glob("*.wav")):
            mode = wav.stem
            json_path = eval_dir / "extended" / f"{entry_id}__{mode}.json"
            if not json_path.is_file():
                continue
            try:
                d = json.load(open(json_path, encoding="utf-8"))
            except Exception:
                continue
            spans = d.get("phrase_time_spans") or []
            base_lang = d.get("base_lang", "en")
            phr_lang = d.get("phrase_lang", "en")
            tasks.append({
                "entry_id": entry_id, "mode": mode, "wav_path": str(wav),
                "phrase_time_spans": [tuple(s) for s in spans],
                "base_lang": base_lang, "phr_lang": phr_lang,
            })
    return tasks


def load_existing(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file():
        return set()
    keys: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add(f"{r['entry_id']}::{r['mode']}")
            except Exception:
                continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=str, required=True,
                    help="Synth dir (relative to repo root or absolute).")
    ap.add_argument("--eval-dir", type=str, default=None,
                    help="Source of per-wav JSON for phrase_time_spans. "
                         "Default: 'eval_<suffix>_fa' where synth='synth_<suffix>'.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Default: 'matrix_lid_<suffix>' where synth='synth_<suffix>'.")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--gpu-shard", type=int, required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--shard-name", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    if not 0 <= args.gpu_shard < args.num_shards:
        ap.error(f"--gpu-shard must be in [0, {args.num_shards}); got {args.gpu_shard}")

    tag = f"[{args.shard_name}] " if args.shard_name else f"[shard{args.gpu_shard}] "

    # Resolve paths
    synth_path = Path(args.synth_dir)
    synth_dir = synth_path if synth_path.is_absolute() else REPO_ROOT / args.synth_dir
    if args.eval_dir:
        eval_path = Path(args.eval_dir)
        eval_dir = eval_path if eval_path.is_absolute() else REPO_ROOT / args.eval_dir
    else:
        name = synth_dir.name
        if name == "synth":
            eval_name = "eval_fa"
        elif name.startswith("synth_"):
            eval_name = "eval_" + name[len("synth_"):] + "_fa"
        else:
            eval_name = "eval_" + name + "_fa"
        eval_dir = synth_dir.parent / eval_name
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_dir = out_path if out_path.is_absolute() else REPO_ROOT / args.output_dir
    else:
        name = synth_dir.name
        if name == "synth":
            out_name = "matrix_lid"
        elif name.startswith("synth_"):
            out_name = "matrix_lid_" + name[len("synth_"):]
        else:
            out_name = "matrix_lid_" + name
        out_dir = synth_dir.parent / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"matrix_lid.shard{args.gpu_shard}.jsonl"

    print(f"{tag}synth_dir={synth_dir}", flush=True)
    print(f"{tag}eval_dir={eval_dir}", flush=True)
    print(f"{tag}out_jsonl={out_jsonl}", flush=True)

    all_tasks = discover_tasks(synth_dir, eval_dir)
    print(f"{tag}discovered {len(all_tasks)} (wav, mode) pairs", flush=True)
    my_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_shards == args.gpu_shard]
    print(f"{tag}shard {args.gpu_shard}/{args.num_shards}: {len(my_tasks)} to do", flush=True)

    existing = load_existing(out_jsonl) if args.skip_existing else set()
    if existing:
        before = len(my_tasks)
        my_tasks = [t for t in my_tasks if f"{t['entry_id']}::{t['mode']}" not in existing]
        print(f"{tag}skip-existing: {before - len(my_tasks)} already done; {len(my_tasks)} remaining", flush=True)

    if not my_tasks:
        print(f"{tag}nothing to do", flush=True)
        return 0

    device = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"
    print(f"{tag}[{time.strftime('%H:%M:%S')}] loading Whisper LID on {device} ...", flush=True)
    t0 = time.time()
    engine = WhisperLIDEngine(device=device, dtype=DTYPE)
    print(f"{tag}loaded in {time.time() - t0:.1f}s", flush=True)

    t_loop = time.time()
    n_done = 0
    n_err = 0
    with out_jsonl.open("a", encoding="utf-8") as fout:
        for batch_start in range(0, len(my_tasks), args.batch_size):
            batch = my_tasks[batch_start:batch_start + args.batch_size]
            audios = []
            keep = []
            durations = []
            for task in batch:
                try:
                    audio, _ = load_audio(Path(task["wav_path"]))
                    matrix_audio = cut_matrix_audio(audio, task["phrase_time_spans"])
                    audios.append(matrix_audio)
                    keep.append(task)
                    durations.append(len(matrix_audio) / SAMPLE_RATE)
                except Exception as e:
                    n_err += 1
                    print(f"{tag}ERR load {task['entry_id']}/{task['mode']}: {type(e).__name__}: {e}", flush=True)
            if not audios:
                continue
            try:
                lids = engine.language_posteriors_batch(audios)
                for task, lid, dur in zip(keep, lids, durations):
                    rec = {
                        "entry_id": task["entry_id"],
                        "mode": task["mode"],
                        "base_lang": task["base_lang"],
                        "phr_lang": task["phr_lang"],
                        "matrix_LID_base":  float(lid.get(task["base_lang"], 0.0)),
                        "matrix_LID_phr":   float(lid.get(task["phr_lang"],  0.0)),
                        "matrix_LID_top":   lid.get("_top", "?"),
                        "matrix_LID_top_p": float(lid.get("_top_p", 0.0)),
                        "matrix_audio_s":   round(dur, 3),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_done += 1
                fout.flush()
                if n_done % 200 < args.batch_size:
                    rate = n_done / max(time.time() - t_loop, 1e-6)
                    eta = (len(my_tasks) - n_done) / max(rate, 1e-6)
                    print(f"{tag}{n_done}/{len(my_tasks)}  {rate:.1f} wav/s  ETA {eta/60:.1f} min", flush=True)
            except Exception as e:
                n_err += len(audios)
                print(f"{tag}ERR forward batch: {type(e).__name__}: {e}", flush=True)

    print(f"{tag}DONE in {time.time()-t_loop:.1f}s  | {n_done} ok  | {n_err} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

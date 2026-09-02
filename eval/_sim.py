"""Compute speaker similarity with both WavLM-base-plus-sv and ECAPA-TDNN.

For each wav, compute four SIM values:
  - SIM_w_wavlm  = cos(WavLM(voice_prompt),     WavLM(whole_synth))
  - SIM_w_ecapa  = cos(ECAPA(voice_prompt),     ECAPA(whole_synth))
  - SIM_me_wavlm = cos(WavLM(matrix_region),    WavLM(embedded_region))
  - SIM_me_ecapa = cos(ECAPA(matrix_region),    ECAPA(embedded_region))

`SIM_w` (whole utterance vs voice prompt) measures global speaker preservation.
`SIM_me` (intra-utterance: matrix region vs embedded region) measures whether
the speaker drifts when the language switches inside the same utterance.

Reuses FA phrase spans from the existing per-wav eval JSONs (no FA re-run).

Output: per-shard JSONL  {entry_id, mode, base_lang, phrase_lang,
                          SIM_w_wavlm, SIM_w_ecapa,
                          SIM_me_wavlm, SIM_me_ecapa}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
MIN_SEG_S = 0.5
MAX_AUDIO_S = 30.0  # safety clip to bound GPU memory
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"


# --- Audio cutting (mirror of eval_extended.py logic) ------------------------

def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def cut_matrix_audio(audio: np.ndarray, phrase_spans) -> np.ndarray:
    """Non-phrase pieces concatenated."""
    if not phrase_spans:
        return audio.astype(np.float32)
    sp = sorted(phrase_spans)
    pieces, cur = [], 0.0
    dur = len(audio) / SAMPLE_RATE
    for s, e in sp:
        s = max(cur, s)
        if s > cur:
            i0 = int(cur * SAMPLE_RATE); i1 = int(s * SAMPLE_RATE)
            if i1 > i0: pieces.append(audio[i0:i1])
        cur = max(cur, e)
    if cur < dur:
        pieces.append(audio[int(cur * SAMPLE_RATE):])
    if not pieces:
        return audio[: int(MIN_SEG_S * SAMPLE_RATE)]
    out = np.concatenate(pieces).astype(np.float32)
    if len(out) < int(MIN_SEG_S * SAMPLE_RATE):
        pad = np.zeros(int(MIN_SEG_S * SAMPLE_RATE) - len(out), dtype=np.float32)
        out = np.concatenate([out, pad])
    return out


def cut_embedded_concat(audio: np.ndarray, phrase_spans) -> np.ndarray:
    """All phrase regions concatenated into one audio."""
    if not phrase_spans:
        return audio[: int(MIN_SEG_S * SAMPLE_RATE)].astype(np.float32)
    sp = sorted(phrase_spans)
    pieces = []
    for s, e in sp:
        i0 = int(max(0.0, s) * SAMPLE_RATE)
        i1 = int(e * SAMPLE_RATE)
        if i1 > i0:
            pieces.append(audio[i0:i1])
    if not pieces:
        return audio[: int(MIN_SEG_S * SAMPLE_RATE)].astype(np.float32)
    out = np.concatenate(pieces).astype(np.float32)
    if len(out) < int(MIN_SEG_S * SAMPLE_RATE):
        pad = np.zeros(int(MIN_SEG_S * SAMPLE_RATE) - len(out), dtype=np.float32)
        out = np.concatenate([out, pad])
    return out


def clip(audio: np.ndarray, max_s: float) -> np.ndarray:
    n = int(max_s * SAMPLE_RATE)
    return audio[:n] if len(audio) > n else audio


# --- Engines -----------------------------------------------------------------

class WavLMEngine:
    def __init__(self, device: str):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
        self.model = WavLMForXVector.from_pretrained(WAVLM_ID).to(device).eval()
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embed(self, audios: List[np.ndarray]) -> torch.Tensor:
        audios = [clip(a, MAX_AUDIO_S) for a in audios]
        inputs = self.fe(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        x = inputs.input_values.to(self.device)
        attn = inputs.get("attention_mask")
        attn = attn.to(self.device) if attn is not None else None
        out = self.model(x, attention_mask=attn) if attn is not None else self.model(x)
        emb = F.normalize(out.embeddings, dim=-1)
        return emb.cpu()  # (B, D)

    def embed_ref(self, ref_path: Path) -> torch.Tensor:
        k = str(ref_path)
        if k in self.cache: return self.cache[k]
        audio = load_audio(ref_path)
        emb = self.embed([audio])[0]
        self.cache[k] = emb
        return emb


class ECAPAEngine:
    def __init__(self, device: str):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_ID,
            savedir=f"/tmp/spkrec-ecapa-voxceleb-{device.replace(':', '_')}",
            run_opts={"device": device},
        )
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embed(self, audios: List[np.ndarray]) -> torch.Tensor:
        # ECAPA expects same-length batch -> pad to max length
        audios = [clip(a, MAX_AUDIO_S) for a in audios]
        max_len = max(len(a) for a in audios)
        padded = np.zeros((len(audios), max_len), dtype=np.float32)
        lengths = torch.zeros(len(audios), dtype=torch.float32)
        for i, a in enumerate(audios):
            padded[i, :len(a)] = a
            lengths[i] = len(a) / max_len
        x = torch.from_numpy(padded).to(self.device)
        lengths = lengths.to(self.device)
        emb = self.model.encode_batch(x, wav_lens=lengths)  # (B, 1, 192)
        emb = emb.squeeze(1)  # (B, 192)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu()

    def embed_ref(self, ref_path: Path) -> torch.Tensor:
        k = str(ref_path)
        if k in self.cache: return self.cache[k]
        audio = load_audio(ref_path)
        emb = self.embed([audio])[0]
        self.cache[k] = emb
        return emb


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # a, b are L2-normalized -> dot product == cosine
    return float((a * b).sum().item())


# --- Task discovery ----------------------------------------------------------

def discover_tasks(synth_dir: Path, eval_dir: Path) -> List[dict]:
    tasks = []
    if not synth_dir.is_dir() or not eval_dir.is_dir():
        return tasks
    for entry_dir in sorted(synth_dir.iterdir()):
        if not entry_dir.is_dir(): continue
        for wav in sorted(entry_dir.glob("*.wav")):
            mode = wav.stem
            jp = eval_dir / "extended" / f"{entry_dir.name}__{mode}.json"
            if not jp.is_file(): continue
            try:
                d = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            spans = d.get("phrase_time_spans") or []
            tasks.append({
                "entry_id": entry_dir.name,
                "mode": mode,
                "wav_path": str(wav),
                "phrase_time_spans": [tuple(s) for s in spans],
                "base_lang": d.get("base_lang", ""),
                "phrase_lang": d.get("phrase_lang", ""),
            })
    return tasks


def load_existing(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file(): return set()
    keys: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add(f"{r['entry_id']}::{r['mode']}")
            except Exception:
                continue
    return keys


# --- Ref voice resolution ----------------------------------------------------
# We need the voice prompt path per entry. The manifest holds it.

def load_ref_index(manifest_path: Path) -> Dict[str, Path]:
    """entry_id -> resolved absolute ref_audio_path"""
    idx: Dict[str, Path] = {}
    if not manifest_path.is_file():
        return idx
    for line in manifest_path.open():
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        rp = d.get("ref_audio_path")
        if not rp: continue
        rp_path = Path(rp)
        if not rp_path.is_absolute():
            # Manifest holds paths like "examples/<lang>_ref.wav" relative to repo root.
            rp_path = REPO_ROOT / rp
        idx[d["id"]] = rp_path
    return idx


# --- Main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=str, required=True,
                    help="Synth dir (relative to repo root or absolute).")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Path to manifest JSONL. Default: <repo-root>/benchmark/manifest.jsonl.")
    ap.add_argument("--eval-dir", type=str, default=None,
                    help="Default: 'eval_<suffix>' where synth='synth_<suffix>'.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Default: 'sim_dual_<suffix>' where synth='synth_<suffix>'.")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--gpu-shard", type=int, required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--shard-name", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Audios per embed call. Each wav contributes 3 audios"
                         " (whole, matrix, embedded) so effective batch is 3x.")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    tag = f"[{args.shard_name}] " if args.shard_name else f"[shard{args.gpu_shard}] "

    # Resolve paths (relative inputs are taken w.r.t. the repo root).
    synth_dir = Path(args.synth_dir)
    if not synth_dir.is_absolute(): synth_dir = REPO_ROOT / args.synth_dir
    name = synth_dir.name
    suffix = name[len("synth_"):] if name.startswith("synth_") else name
    if args.eval_dir:
        eval_path = Path(args.eval_dir)
        eval_dir = eval_path if eval_path.is_absolute() else REPO_ROOT / args.eval_dir
    else:
        eval_dir = REPO_ROOT / f"eval_{suffix}"
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_dir = out_path if out_path.is_absolute() else REPO_ROOT / args.output_dir
    else:
        out_dir = REPO_ROOT / f"sim_dual_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"sim_dual.shard{args.gpu_shard}.jsonl"

    print(f"{tag}synth_dir={synth_dir}", flush=True)
    print(f"{tag}eval_dir={eval_dir}", flush=True)
    print(f"{tag}out={out_jsonl}", flush=True)

    if args.manifest:
        m_path = Path(args.manifest)
        manifest = m_path if m_path.is_absolute() else (REPO_ROOT / args.manifest)
    else:
        manifest = REPO_ROOT / "benchmark" / "manifest.jsonl"
    ref_idx = load_ref_index(manifest)
    print(f"{tag}manifest entries with ref: {len(ref_idx)}", flush=True)

    all_tasks = discover_tasks(synth_dir, eval_dir)
    my_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_shards == args.gpu_shard]
    print(f"{tag}shard {args.gpu_shard}/{args.num_shards}: {len(my_tasks)} tasks", flush=True)

    existing = load_existing(out_jsonl) if args.skip_existing else set()
    if existing:
        before = len(my_tasks)
        my_tasks = [t for t in my_tasks if f"{t['entry_id']}::{t['mode']}" not in existing]
        print(f"{tag}skip-existing: {before - len(my_tasks)} done; {len(my_tasks)} remaining", flush=True)

    if not my_tasks:
        print(f"{tag}nothing to do", flush=True)
        return 0

    device = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"
    print(f"{tag}[{time.strftime('%H:%M:%S')}] loading models on {device} ...", flush=True)
    t0 = time.time()
    wavlm = WavLMEngine(device)
    ecapa = ECAPAEngine(device)
    print(f"{tag}loaded in {time.time()-t0:.1f}s", flush=True)

    t_loop = time.time()
    n_done, n_err = 0, 0
    BS = args.batch_size

    with out_jsonl.open("a", encoding="utf-8") as fout:
        for batch_start in range(0, len(my_tasks), BS):
            batch = my_tasks[batch_start:batch_start + BS]
            # Collect 3 audios per wav: whole, matrix, embedded
            all_audios: List[np.ndarray] = []
            slots: List[Tuple[int, int]] = []  # (task_idx, audio_role 0/1/2)
            ref_paths: List[Optional[Path]] = []
            valid_tasks = []
            for task in batch:
                try:
                    audio = load_audio(Path(task["wav_path"]))
                    matrix = cut_matrix_audio(audio, task["phrase_time_spans"])
                    embedded = cut_embedded_concat(audio, task["phrase_time_spans"])
                    ti = len(valid_tasks)
                    valid_tasks.append(task)
                    slots.extend([(ti, 0), (ti, 1), (ti, 2)])
                    all_audios.extend([audio, matrix, embedded])
                    ref_paths.append(ref_idx.get(task["entry_id"]))
                except Exception as e:
                    n_err += 1
                    print(f"{tag}ERR load {task['entry_id']}/{task['mode']}: {e}", flush=True)
            if not all_audios:
                continue

            try:
                wavlm_embs = wavlm.embed(all_audios)  # (3B, 256)
                ecapa_embs = ecapa.embed(all_audios)  # (3B, 192)
            except Exception as e:
                n_err += len(valid_tasks)
                print(f"{tag}ERR forward batch: {type(e).__name__}: {e}", flush=True)
                continue

            for ti, task in enumerate(valid_tasks):
                w_whole, w_mat, w_emb = wavlm_embs[3*ti], wavlm_embs[3*ti+1], wavlm_embs[3*ti+2]
                e_whole, e_mat, e_emb = ecapa_embs[3*ti], ecapa_embs[3*ti+1], ecapa_embs[3*ti+2]
                ref_p = ref_paths[ti]
                if ref_p is None:
                    sim_w_wavlm = float("nan"); sim_w_ecapa = float("nan")
                else:
                    try:
                        w_ref = wavlm.embed_ref(ref_p)
                        e_ref = ecapa.embed_ref(ref_p)
                        sim_w_wavlm = cosine(w_ref, w_whole)
                        sim_w_ecapa = cosine(e_ref, e_whole)
                    except Exception as ex:
                        sim_w_wavlm = float("nan"); sim_w_ecapa = float("nan")
                        print(f"{tag}WARN ref embed failed for {task['entry_id']}: {ex}", flush=True)
                sim_me_wavlm = cosine(w_mat, w_emb)
                sim_me_ecapa = cosine(e_mat, e_emb)
                rec = {
                    "entry_id":      task["entry_id"],
                    "mode":          task["mode"],
                    "base_lang":     task["base_lang"],
                    "phrase_lang":   task["phrase_lang"],
                    "ref_audio":     str(ref_p) if ref_p else None,
                    "SIM_w_wavlm":   sim_w_wavlm,
                    "SIM_w_ecapa":   sim_w_ecapa,
                    "SIM_me_wavlm":  sim_me_wavlm,
                    "SIM_me_ecapa":  sim_me_ecapa,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
            fout.flush()

            if n_done % 200 < BS:
                rate = n_done / max(time.time() - t_loop, 1e-6)
                eta = (len(my_tasks) - n_done) / max(rate, 1e-6)
                print(f"{tag}{n_done}/{len(my_tasks)}  {rate:.2f} wav/s  ETA {eta/60:.1f}m", flush=True)

    print(f"{tag}DONE in {time.time()-t_loop:.1f}s | {n_done} ok | {n_err} err", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

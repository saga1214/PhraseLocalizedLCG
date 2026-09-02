#!/usr/bin/env python
"""Agent 5b: Extended Code-Switching TTS evaluation (BATCHED + SHARDED).

For each (entry, mode) wav, compute:
  1. mWER  -- matrix-language WER/CER (script-filtered, forced ASR)
  2. eWER  -- embedded-language WER/CER over phrase audio segments
  3. MER   -- mixed error rate (weighted average)
  4. LA (segment)    -- per-phrase language accuracy on the cut phrase span
  5. LID confidence  -- whisper P(phrase_lang | phrase_segment)
  6. SIM-o (WavLM-base-plus-sv) vs ref audio: whole, matrix, embedded(per-phrase)

PIPELINE:
  Pass F: Qwen3 ForcedAligner (one subprocess per shard, model loaded once)
          -> char-level alignment items per wav, used for phrase spans.
  (per chunk of N wavs, N = --batch-size):
  Pass A: Whisper auto-detect (no word timestamps; FA replaces that purpose)
                                                     (batch=N)
  Pass B: Whisper forced base_lang                   (group by lang, batch=N)
  Phrase spans derived from Pass F (FA) char-overlaps with phrase_char_spans.
  FA is mandatory: a wav whose spans cannot be resolved is recorded as a
  failure rather than silently falling back to proportional spans.
  Segment cuts collected ACROSS the entire chunk into M flat segments:
    Pass C: Whisper forced phrase_lang on segments   (group by lang, batch>=16)
    Pass D: Whisper auto on segments                 (batch>=16, for LA-segment)
    Pass E: 1-step Whisper LID posteriors            (batch>=32, cheap)
  WavLM-sv: whole-wavs (N) + matrix audio (N) + phrase segments (M)
            run in length-sorted mini-batches (padding-efficient).

Per-wav output: eval/extended/<id>__<mode>.json (resume-safe).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    WavLMForXVector,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

try:
    import jiwer  # type: ignore
    HAS_JIWER = True
except Exception:
    HAS_JIWER = False


# --- Paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)  # resolve manifest ref_audio_path entries from the repo root
MANIFEST = REPO_ROOT / "benchmark" / "manifest.jsonl"

# Default synth/eval dir pair; both are overridden in main() from CLI flags.
SYNTH_DIR = REPO_ROOT / "synth"
EVAL_DIR = REPO_ROOT / "eval"
PER_WAV_DIR = EVAL_DIR / "extended"
USAGE_LOG = EVAL_DIR / "extended_usage_log.jsonl"
SUMMARY_CSV = EVAL_DIR / "extended_summary.csv"
REPORT_MD = EVAL_DIR / "extended_report.md"


# --- Modes --------------------------------------------------------------------
# Populated in main() via discover_modes_from_synth_dir(SYNTH_DIR).
MODES: list = []


def discover_modes_from_synth_dir(synth_dir) -> list:
    """Scan ``synth_dir/<entry_id>/*.wav`` and return all distinct stems.

    Ordering: ``baseline_src_only`` first (if present), then the remaining stems
    sorted alphabetically so downstream report rows are deterministic across runs.
    """
    found = set()
    if not synth_dir.exists():
        return []
    for entry_dir in synth_dir.iterdir():
        if not entry_dir.is_dir():
            continue
        for wav in entry_dir.glob("*.wav"):
            if wav.name.startswith("."):
                continue
            found.add(wav.stem)
    if not found:
        return []
    out = []
    if "baseline_src_only" in found:
        out.append("baseline_src_only")
        found.discard("baseline_src_only")
    out.extend(sorted(found))
    return out


# --- Constants ----------------------------------------------------------------

MODEL_ID = "openai/whisper-large-v3"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
DTYPE = torch.float16
SAMPLE_RATE = 16000
WHISPER_AUDIO_SAMPLES = 30 * SAMPLE_RATE
MIN_SEG_S = 0.5
# Max wavlm audio length (sec) per batched item to bound memory.
WAVLM_MAX_S = 30.0

LANG_MAP = {
    "en": "english", "ko": "korean", "zh": "chinese", "ja": "japanese",
    "de": "german",  "fr": "french",
}
LANG_CODES = list(LANG_MAP.keys())
# Languages whose script is Latin alphabet (token-level WER applies).
LATIN_LANGS = {"en", "de", "fr"}

# Script regexes
RE_LATIN_TOK = re.compile(r"[A-Za-z']+")
RE_HANGUL = re.compile(r"[가-힣]")
RE_KANA = re.compile(r"[぀-ヿ]")
RE_HANZI = re.compile(r"[一-鿿]")


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def is_word_metric(lang: str) -> bool:
    return lang in LATIN_LANGS  # en/de/fr → word-WER; ko/ja/zh → CER


def script_filter(text: str, lang: str) -> str:
    if lang in LATIN_LANGS:
        # en/de/fr share the Latin alphabet; we cannot disambiguate by script
        # alone. WER is computed over lowercased Latin tokens for all three.
        toks = RE_LATIN_TOK.findall(text)
        return " ".join(t.lower() for t in toks)
    if lang == "ko":
        return "".join(RE_HANGUL.findall(text))
    if lang == "zh":
        chars = [c for c in text if RE_HANZI.match(c) and not RE_KANA.match(c)]
        return "".join(chars)
    if lang == "ja":
        chars = [c for c in text if RE_KANA.match(c) or RE_HANZI.match(c)]
        return "".join(chars)
    return text


def remove_phrase_chars(text: str, phrases: List[str]) -> str:
    out = text
    for p in phrases:
        if not p:
            continue
        out = out.replace(p, " ")
    return out


def err_rate(ref: str, hyp: str, lang: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        if is_word_metric(lang):
            return float(jiwer.wer(ref, hyp))
        return float(jiwer.cer(ref, hyp))
    except Exception:
        a = ref.split() if is_word_metric(lang) else list(ref)
        b = hyp.split() if is_word_metric(lang) else list(hyp)
        la, lb = len(a), len(b)
        if la == 0:
            return 1.0 if lb else 0.0
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[lb] / la


def unit_count(s: str, lang: str) -> int:
    if is_word_metric(lang):
        return len(s.split())
    return len(s)


# --- Qwen3 forced-aligner helpers (mirror of dlm_sampling_experiments.py) -----

QWEN_LANG_BY_CODE = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French",
    "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}


def resolve_qwen_aligner_language(code: str) -> str:
    if not code:
        raise ValueError("language code required for Qwen FA")
    k = code.strip().lower()
    if k in QWEN_LANG_BY_CODE:
        return QWEN_LANG_BY_CODE[k]
    by_name = {v.lower(): v for v in QWEN_LANG_BY_CODE.values()}
    if k in by_name:
        return by_name[k]
    raise ValueError(f"Unsupported Qwen FA language: {code!r}")


def _align_item_text(item: Any) -> str:
    return str(item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""))


def _align_item_start(item: Any) -> float:
    return float(item.get("start_time", 0.0) if isinstance(item, dict) else getattr(item, "start_time", 0.0))


def _align_item_end(item: Any) -> float:
    return float(item.get("end_time", 0.0) if isinstance(item, dict) else getattr(item, "end_time", 0.0))


def _normalized_chars_with_offsets(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Returns (normalized_string, [(orig_start, orig_end_excl)]) where each
    char in normalized aligns to its original text position. Spaces collapsed."""
    chars: List[str] = []
    offsets: List[Tuple[int, int]] = []
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


def align_items_to_source_chars(
    source_text: str, align_items: List[Any]
) -> List[Tuple[Any, Tuple[int, int]]]:
    """Returns [(item, (orig_char_start, orig_char_end))] -- each FA item paired
    with its character range in source_text. Walks both strings in parallel and
    finds each item's normalized form in source from a moving cursor."""
    norm_source, offsets = _normalized_chars_with_offsets(source_text)
    out: List[Tuple[Any, Tuple[int, int]]] = []
    cursor = 0
    fallback_cursor = 0
    for item in align_items:
        item_text = _align_item_text(item)
        norm_item, _ = _normalized_chars_with_offsets(item_text)
        if not norm_item:
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


def compute_fa_phrase_span(
    fa_items: List[Any],
    full_text: str,
    phrase_char_span: Optional[List[int]],
    duration_s: float,
) -> Optional[Tuple[float, float]]:
    """Given FA items + a phrase's character span in full_text, return the
    (start_s, end_s) time range that covers all FA items whose char range
    overlaps the phrase span. Returns None if no overlap (=> fall back to
    proportional). Pad to MIN_SEG_S if too short."""
    if not fa_items or phrase_char_span is None:
        return None
    if isinstance(phrase_char_span, (list, tuple)) and len(phrase_char_span) == 2:
        pc_start, pc_end = int(phrase_char_span[0]), int(phrase_char_span[1])
    else:
        return None
    aligned = align_items_to_source_chars(full_text, list(fa_items))
    t_starts: List[float] = []
    t_ends: List[float] = []
    for item, (c_start, c_end) in aligned:
        if c_end <= pc_start or c_start >= pc_end:
            continue
        ts = _align_item_start(item)
        te = _align_item_end(item)
        if te > ts:
            t_starts.append(ts)
            t_ends.append(te)
    if not t_starts:
        return None
    s = max(0.0, float(min(t_starts)))
    e = min(float(duration_s), float(max(t_ends)))
    if e - s < MIN_SEG_S:
        mid = 0.5 * (s + e)
        s = max(0.0, mid - MIN_SEG_S / 2)
        e = min(duration_s, s + MIN_SEG_S)
    return (s, e)


def run_qwen_fa_batch(
    args: argparse.Namespace,
    samples: List[Dict[str, str]],
) -> Tuple[List[List[Dict[str, Any]]], List[Optional[str]]]:
    """samples = list of dicts with keys 'audio' (path str), 'text', 'language'
    (Qwen FA language name). Returns parallel list of items_per_sample (each =
    list of {text,start_time,end_time}), plus list of error strings (None on
    success). Uses synth/qwen3_forced_aligner_worker.py."""
    if not samples:
        return [], []
    worker = REPO_ROOT / "synth" / "qwen3_forced_aligner_worker.py"
    if not worker.exists():
        raise FileNotFoundError(f"Missing FA worker: {worker}")
    cache_root = Path.home() / ".cache" / "omnivoice"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eval_fa_", dir=str(cache_root)) as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.json"
        output_path = tmp_dir / "output.json"
        req: Dict[str, Any] = {
            "samples": samples,
            "model": args.fa_aligner_model,
            "device": None,  # subprocess uses CUDA_VISIBLE_DEVICES already set by parent
            "dtype": args.fa_aligner_dtype,
        }
        if args.fa_aligner_attn_implementation:
            req["attn_implementation"] = args.fa_aligner_attn_implementation
        with input_path.open("w", encoding="utf-8") as f:
            json.dump(req, f, ensure_ascii=False)
        cmd = [
            args.fa_aligner_python,
            str(worker),
            "--input", str(input_path),
            "--output", str(output_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FA subprocess failed (exit={completed.returncode}):\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        with output_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        items_per = result.get("items_per_sample") or []
        if not isinstance(items_per, list):
            raise RuntimeError(f"Invalid FA output (missing 'items_per_sample'): {result}")
        errors = result.get("sample_errors") or [None] * len(items_per)
        return items_per, errors


# --- Phrase span estimation ---------------------------------------------------

def proportional_span(
    duration_s: float, phrase: str, full_text: str, phrase_char_span: Optional[List[int]]
) -> Tuple[float, float]:
    if not full_text or duration_s <= 0:
        return (0.0, max(0.0, duration_s))
    n = len(full_text)
    if phrase_char_span and len(phrase_char_span) == 2:
        s_char, e_char = phrase_char_span
    else:
        s_char = full_text.find(phrase)
        if s_char < 0:
            s_char = max(0, n // 2 - len(phrase) // 2)
        e_char = s_char + len(phrase)
    s = max(0.0, duration_s * s_char / n)
    e = min(duration_s, duration_s * e_char / n)
    if e - s < MIN_SEG_S:
        mid = 0.5 * (s + e)
        s = max(0.0, mid - MIN_SEG_S / 2)
        e = min(duration_s, s + MIN_SEG_S)
    return (s, e)


def find_phrase_span(
    phrase: str,
    duration_s: float,
    full_text: str,
    phrase_char_span: Optional[List[int]],
    fa_items: Optional[List[Any]] = None,
) -> Tuple[float, float, str]:
    """Estimate the (start_s, end_s) time range for ``phrase`` in the wav.

    Requires Qwen3 forced-alignment items (``fa_items``). Raises if FA is
    missing or fails to overlap — the proportional-span fallback was found
    to silently corrupt downstream metrics (see git log for the incident).
    Callers that genuinely want proportional spans must construct them
    explicitly via ``proportional_span``.
    """
    if not fa_items:
        raise RuntimeError(
            f"FA items missing for phrase {phrase!r} (proportional fallback disabled)"
        )
    fa_span = compute_fa_phrase_span(fa_items, full_text, phrase_char_span, duration_s)
    if fa_span is None:
        raise RuntimeError(
            f"FA failed to overlap phrase {phrase!r} (proportional fallback disabled)"
        )
    return float(fa_span[0]), float(fa_span[1]), "fa"


# --- Audio helpers ------------------------------------------------------------

def load_audio(path: Path) -> Tuple[np.ndarray, float]:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), len(audio) / SAMPLE_RATE


def pad30(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WHISPER_AUDIO_SAMPLES:
        pad = np.zeros(WHISPER_AUDIO_SAMPLES - len(audio), dtype=audio.dtype)
        return np.concatenate([audio, pad])
    return audio[:WHISPER_AUDIO_SAMPLES]


def cut_segment(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    s = max(0, int(start_s * SAMPLE_RATE))
    e = min(len(audio), int(end_s * SAMPLE_RATE))
    if e <= s:
        e = min(len(audio), s + int(MIN_SEG_S * SAMPLE_RATE))
    seg = audio[s:e].astype(np.float32)
    if len(seg) < int(MIN_SEG_S * SAMPLE_RATE):
        pad = np.zeros(int(MIN_SEG_S * SAMPLE_RATE) - len(seg), dtype=np.float32)
        seg = np.concatenate([seg, pad])
    return seg


def cut_matrix_audio(
    audio: np.ndarray, phrase_spans: List[Tuple[float, float]]
) -> np.ndarray:
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
        return audio.astype(np.float32)
    return np.concatenate(pieces).astype(np.float32)


def clip_audio(audio: np.ndarray, max_s: float) -> np.ndarray:
    """Clip a (possibly long) audio to max_s seconds for WavLM batching."""
    n_max = int(max_s * SAMPLE_RATE)
    if len(audio) <= n_max:
        return audio
    return audio[:n_max]


# --- Whisper engine -----------------------------------------------------------

class WhisperEngine:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=DTYPE, attn_implementation="sdpa"
            ).to(device)
        except Exception:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=DTYPE
            ).to(device)
        self.model.eval()
        tok = self.processor.tokenizer
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.all_lang_ids: List[int] = []
        for tid in range(50000, 52000):
            t = tok.convert_ids_to_tokens(tid)
            if (
                isinstance(t, str)
                and len(t) == 6
                and t.startswith("<|")
                and t.endswith("|>")
                and t[2:4].isalpha()
            ):
                self.all_lang_ids.append(tid)
        self.code_to_lang_id = {
            c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in LANG_CODES
        }

    @torch.inference_mode()
    def transcribe_with_words(
        self,
        audios: List[np.ndarray],
        language: Optional[str] = None,
        max_audio_s_hint: Optional[float] = None,
    ) -> List[Dict]:
        padded = [pad30(a) for a in audios]
        inputs = self.processor(padded, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(self.device).to(DTYPE)
        if max_audio_s_hint is not None:
            max_audio_s = max_audio_s_hint
        else:
            max_audio_s = max(len(a) / SAMPLE_RATE for a in audios)
        # ~13 tokens/sec upper bound (CJK heavier than Latin); clamp to whisper's
        # safe max_new_tokens of 440.
        cap = int(min(440, max(48, max_audio_s * 13 + 24)))
        gen_kwargs = dict(
            max_new_tokens=cap,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
        )
        if language is not None:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"
        out = self.model.generate(feats, **gen_kwargs)
        seqs = out["sequences"] if isinstance(out, dict) else out.sequences
        results = []
        for i in range(seqs.shape[0]):
            ids = seqs[i].tolist()
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            detected = self._first_lang_token(ids)
            results.append({"text": text, "lang": detected, "words": []})
        return results

    def _first_lang_token(self, ids: List[int]) -> Optional[str]:
        tok = self.processor.tokenizer
        for tid in ids[:6]:
            t = tok.convert_ids_to_tokens(tid)
            if (
                isinstance(t, str)
                and len(t) == 6
                and t.startswith("<|")
                and t.endswith("|>")
                and t[2:4].isalpha()
            ):
                return t[2:4]
        return None

    @torch.inference_mode()
    def language_posteriors_batch(self, audios: List[np.ndarray]) -> List[Dict[str, float]]:
        padded = [pad30(a) for a in audios]
        inputs = self.processor(padded, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(self.device).to(DTYPE)
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


# --- WavLM engine -------------------------------------------------------------

class WavLMEngine:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embed(self, audios: List[np.ndarray]) -> torch.Tensor:
        # Clip to safe max length to avoid OOM on long matrix concatenations.
        audios = [clip_audio(a, WAVLM_MAX_S) for a in audios]
        inputs = self.fe(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
            out = self.model(input_values, attention_mask=attn)
        else:
            out = self.model(input_values)
        emb = out.embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu()

    @torch.inference_mode()
    def embed_ref(self, ref_path: str) -> torch.Tensor:
        if ref_path in self.cache:
            return self.cache[ref_path]
        audio, _ = librosa.load(ref_path, sr=SAMPLE_RATE, mono=True)
        emb = self.embed([audio.astype(np.float32)])[0]
        self.cache[ref_path] = emb
        return emb

    @torch.inference_mode()
    def embed_batched_sorted(
        self, audios: List[np.ndarray], batch_size: int
    ) -> List[torch.Tensor]:
        """Length-sort + chunk to minimize padding waste."""
        n = len(audios)
        out: List[Optional[torch.Tensor]] = [None] * n
        order = sorted(range(n), key=lambda k: len(audios[k]))
        for start in range(0, n, batch_size):
            idxs = order[start : start + batch_size]
            sub = [audios[k] for k in idxs]
            embs = self.embed(sub)
            for jj, k in enumerate(idxs):
                out[k] = embs[jj]
        return out  # type: ignore


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    )


# --- Chunk processing ---------------------------------------------------------

def process_chunk(
    chunk_data: List[tuple],
    whisper: "WhisperEngine",
    wavlm: "WavLMEngine",
    full_batch: int,
    seg_batch: int,
    wavlm_batch: int,
    fa_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[dict]:
    """Run all Whisper passes + WavLM aggressively batched across the chunk.

    Phrase spans come from a precomputed Qwen3-ForcedAligner cache
    (``fa_cache: wav_path -> list of {text, start_time, end_time}``); when
    an entry is missing or FA fails to overlap, the wav is marked as a
    failure (see ``find_phrase_span``) instead of falling back to
    proportional spans. Whisper Pass A returns auto-ASR text + detected
    language only.

    Pass B/C/D/E use SDPA (no timestamps) to allow large batches.

    chunk_data: list of (entry, mode, wav_path, audio_np, duration_s)
    Returns: list of per-wav JSON dicts.
    """
    if fa_cache is None:
        fa_cache = {}
    n = len(chunk_data)
    if n == 0:
        return []

    audios = [c[3] for c in chunk_data]

    # --- Pass A: auto-ASR (no word timestamps; FA handles span estimation) ---
    autos: List[Optional[dict]] = [None] * n
    a_batch = full_batch
    for start in range(0, n, a_batch):
        sub_idxs = list(range(start, min(start + a_batch, n)))
        sub = [audios[i] for i in sub_idxs]
        try:
            outs = whisper.transcribe_with_words(
                sub, language=None            )
            for j, i in enumerate(sub_idxs):
                autos[i] = outs[j]
        except Exception as e:
            print(f"[ERR-PASS-A] sub@{start}: {e}", flush=True)
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Pass B: forced base_lang (no timestamps, SDPA, large batch) ---
    forced_results: List[Optional[dict]] = [None] * n
    by_base: Dict[str, List[int]] = defaultdict(list)
    for i, (entry, _m, _w, _a, _d) in enumerate(chunk_data):
        by_base[LANG_MAP.get(entry.get("base_lang", "en"), "english")].append(i)
    for lang_name, idxs in by_base.items():
        for start in range(0, len(idxs), full_batch):
            sub_idxs = idxs[start : start + full_batch]
            sub = [audios[i] for i in sub_idxs]
            try:
                outs = whisper.transcribe_with_words(
                    sub, language=lang_name                )
                for j, i in enumerate(sub_idxs):
                    forced_results[i] = outs[j]
            except Exception as e:
                print(f"[ERR-PASS-B] lang={lang_name}: {e}", flush=True)
                traceback.print_exc()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Spans + matrix metric ---
    # Per-wav FA error handling: if find_phrase_span raises for any phrase in
    # a wav, we record the wav as a synth-failure (success=False, no metrics)
    # but the OTHER wavs in the chunk continue with their normal Whisper/WavLM
    # passes. The pre-2026-05-22 chunk-level try/except propagated one wav's
    # error to all 16 wavs in the same chunk; this loop keeps the blast radius
    # at one wav.
    all_spans: List[List[Tuple[float, float]]] = []
    all_methods: List[List[str]] = []
    matrix_results: List[Dict] = []
    fa_failed_idx: Dict[int, str] = {}  # chunk-index -> error message
    for i, (entry, mode, _wav, _audio, dur) in enumerate(chunk_data):
        auto = autos[i] if autos[i] is not None else {"text": "", "lang": None, "words": []}
        forced = forced_results[i] if forced_results[i] is not None else {"text": "", "lang": None, "words": []}
        phrases = entry.get("phrases", [])
        char_spans = entry.get("phrase_char_spans") or [None] * len(phrases)
        full_text = entry.get("text", "")
        fa_items = fa_cache.get(str(_wav)) if fa_cache else None
        spans = []
        methods = []
        try:
            for ph, cs in zip(phrases, char_spans):
                s, e, meth = find_phrase_span(
                    ph, dur, full_text, cs, fa_items=fa_items
                )
                spans.append((float(s), float(e)))
                methods.append(meth)
        except RuntimeError as fa_err:
            # Synth failed for one or more phrases: FA produced no usable
            # alignment. Use placeholder (0, MIN_SEG_S) spans so the rest of
            # this chunk's batched passes don't crash; we overwrite this wav's
            # result with an explicit success=False record at the end of
            # process_chunk.
            fa_failed_idx[i] = str(fa_err)
            spans = [(0.0, MIN_SEG_S) for _ in phrases]
            methods = ["error" for _ in phrases]
        all_spans.append(spans)
        all_methods.append(methods)
        base_lang = entry.get("base_lang", "en")
        matrix_ref_raw = remove_phrase_chars(full_text, phrases)
        matrix_ref = script_filter(matrix_ref_raw, base_lang)
        matrix_hyp = script_filter(forced["text"], base_lang)
        mwer = err_rate(matrix_ref, matrix_hyp, base_lang)
        m_units = unit_count(matrix_ref, base_lang)
        matrix_results.append({
            "mWER": mwer, "m_units": m_units,
            "matrix_ref": matrix_ref, "matrix_hyp": matrix_hyp,
        })

    # --- Cut phrase segments across the whole chunk ---
    flat_seg_audios: List[np.ndarray] = []
    flat_seg_durs: List[float] = []
    flat_owner_idx: List[int] = []
    flat_owner_pos: List[int] = []  # j-th phrase within its wav
    flat_phrase_lang_name: List[str] = []
    flat_phrase_lang: List[str] = []
    flat_phrase: List[str] = []
    for i, (entry, _m, _w, audio, _d) in enumerate(chunk_data):
        phrases = entry.get("phrases", [])
        phrase_lang = entry.get("phrase_lang", "en")
        lang_name = LANG_MAP.get(phrase_lang, "english")
        for j, ph in enumerate(phrases):
            s, e = all_spans[i][j]
            seg = cut_segment(audio, s, e)
            flat_seg_audios.append(seg)
            flat_seg_durs.append(len(seg) / SAMPLE_RATE)
            flat_owner_idx.append(i)
            flat_owner_pos.append(j)
            flat_phrase_lang_name.append(lang_name)
            flat_phrase_lang.append(phrase_lang)
            flat_phrase.append(ph)

    M = len(flat_seg_audios)
    forced_seg_outs: List[Optional[dict]] = [None] * M
    auto_seg_outs: List[Optional[dict]] = [None] * M
    lid_post_outs: List[Optional[dict]] = [None] * M

    if M > 0:
        # Use seg_batch >= 16. Phrase segments are short (<=5s typically), so we
        # can pack many at once and short max_new_tokens.
        # --- Pass C: forced phrase_lang ---
        by_lang_seg: Dict[str, List[int]] = defaultdict(list)
        for k, lname in enumerate(flat_phrase_lang_name):
            by_lang_seg[lname].append(k)
        for lname, idxs in by_lang_seg.items():
            # Sort by duration to make max_audio_s tighter per batch
            idxs_sorted = sorted(idxs, key=lambda k: flat_seg_durs[k])
            for start in range(0, len(idxs_sorted), seg_batch):
                sub_idxs = idxs_sorted[start : start + seg_batch]
                sub = [flat_seg_audios[k] for k in sub_idxs]
                max_s = max(flat_seg_durs[k] for k in sub_idxs)
                try:
                    outs = whisper.transcribe_with_words(
                        sub, language=lname, max_audio_s_hint=max_s,
                    )
                    for jj, k in enumerate(sub_idxs):
                        forced_seg_outs[k] = outs[jj]
                except Exception as e:
                    print(f"[ERR-PASS-C] lang={lname}: {e}", flush=True)
                    traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Pass D: auto on segments (LA + auto_seg_lang) ---
        order = sorted(range(M), key=lambda k: flat_seg_durs[k])
        for start in range(0, M, seg_batch):
            sub_idxs = order[start : start + seg_batch]
            sub = [flat_seg_audios[k] for k in sub_idxs]
            max_s = max(flat_seg_durs[k] for k in sub_idxs)
            try:
                outs = whisper.transcribe_with_words(
                    sub, language=None, max_audio_s_hint=max_s,
                )
                for jj, k in enumerate(sub_idxs):
                    auto_seg_outs[k] = outs[jj]
            except Exception as e:
                print(f"[ERR-PASS-D] @{start}: {e}", flush=True)
                traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Pass E: 1-step LID posteriors (very cheap, larger batch) ---
        lid_batch = max(seg_batch, 32)
        for start in range(0, M, lid_batch):
            sub_idxs = list(range(start, min(start + lid_batch, M)))
            sub = [flat_seg_audios[k] for k in sub_idxs]
            try:
                outs = whisper.language_posteriors_batch(sub)
                for jj, k in enumerate(sub_idxs):
                    lid_post_outs[k] = outs[jj]
            except Exception as e:
                print(f"[ERR-PASS-E] @{start}: {e}", flush=True)
                traceback.print_exc()

    # --- WavLM embeddings: collect whole + matrix + segs into one batch ---
    wavlm_audios: List[np.ndarray] = []
    whole_offset: List[int] = []
    matrix_offset: List[int] = []
    for i, (entry, _m, _w, audio, _d) in enumerate(chunk_data):
        whole_offset.append(len(wavlm_audios))
        wavlm_audios.append(audio.astype(np.float32))
        mat_audio = cut_matrix_audio(audio, all_spans[i])
        if len(mat_audio) < int(MIN_SEG_S * SAMPLE_RATE):
            mat_audio = np.concatenate([
                mat_audio,
                np.zeros(int(MIN_SEG_S * SAMPLE_RATE) - len(mat_audio), dtype=np.float32),
            ])
        matrix_offset.append(len(wavlm_audios))
        wavlm_audios.append(mat_audio)
    flat_seg_wavlm_offset = [-1] * M
    for k, seg in enumerate(flat_seg_audios):
        flat_seg_wavlm_offset[k] = len(wavlm_audios)
        wavlm_audios.append(seg)

    wavlm_embs = wavlm.embed_batched_sorted(wavlm_audios, batch_size=wavlm_batch)

    # --- Compose per-wav results ---
    owner_to_segs: Dict[int, List[int]] = defaultdict(list)
    for k, owner in enumerate(flat_owner_idx):
        owner_to_segs[owner].append(k)

    results: List[dict] = []
    for i, (entry, mode, _wav, _audio, dur) in enumerate(chunk_data):
        auto = autos[i] if autos[i] is not None else {"text": "", "lang": None, "words": []}
        forced = forced_results[i] if forced_results[i] is not None else {"text": "", "lang": None, "words": []}
        phrases = entry.get("phrases", [])
        phrase_lang = entry.get("phrase_lang", "en")
        base_lang = entry.get("base_lang", "en")
        spans = all_spans[i]
        methods = all_methods[i]
        mr = matrix_results[i]

        seg_idxs = sorted(owner_to_segs.get(i, []), key=lambda k: flat_owner_pos[k])
        embedded_asr_texts: List[str] = []
        embedded_wers: List[float] = []
        embedded_units: List[int] = []
        la_seg: List[float] = []
        lid_conf: List[float] = []
        auto_seg_lang: List[Optional[str]] = []
        sim_emb_per_phrase: List[float] = []

        ref_emb = wavlm.embed_ref(entry["ref_audio_path"])
        for k in seg_idxs:
            ph = flat_phrase[k]
            fs = forced_seg_outs[k] or {"text": "", "lang": None, "words": []}
            au = auto_seg_outs[k] or {"text": "", "lang": None, "words": []}
            lid = lid_post_outs[k] or {}
            hyp = script_filter(fs["text"], phrase_lang)
            ref = script_filter(ph, phrase_lang)
            er = err_rate(ref, hyp, phrase_lang)
            embedded_asr_texts.append(fs["text"])
            embedded_wers.append(er)
            embedded_units.append(unit_count(ref, phrase_lang))
            la = 0.0
            seg_text = au["text"]
            seg_lang = au["lang"]
            auto_seg_lang.append(seg_lang)
            nseg = normalize(seg_text)
            nph = normalize(ph)
            if nph and nph in nseg:
                la = 1.0
            elif nph and nseg:
                sm = SequenceMatcher(a=nseg, b=nph, autojunk=False)
                m = sm.find_longest_match(0, len(nseg), 0, len(nph))
                if m.size >= max(3, int(0.5 * len(nph))):
                    if phrase_lang == "en":
                        ok = bool(RE_LATIN_TOK.search(seg_text))
                    elif phrase_lang == "ko":
                        ok = bool(RE_HANGUL.search(seg_text))
                    elif phrase_lang == "ja":
                        ok = bool(RE_KANA.search(seg_text)) or bool(RE_HANZI.search(seg_text))
                    elif phrase_lang == "zh":
                        ok = bool(RE_HANZI.search(seg_text)) and not bool(RE_KANA.search(seg_text))
                    else:
                        ok = True
                    la = 1.0 if ok else 0.0
            la_seg.append(la)
            lid_conf.append(float(lid.get(phrase_lang, 0.0)))
            seg_emb = wavlm_embs[flat_seg_wavlm_offset[k]]
            sim_emb_per_phrase.append(cosine(seg_emb, ref_emb))

        if embedded_wers:
            ewer = float(statistics.mean(embedded_wers))
            e_units_total = sum(embedded_units)
        else:
            ewer = 0.0
            e_units_total = 0
        if (mr["m_units"] + e_units_total) > 0:
            mer = (mr["mWER"] * mr["m_units"] + ewer * e_units_total) / (
                mr["m_units"] + e_units_total
            )
        else:
            mer = 0.0

        whole_emb = wavlm_embs[whole_offset[i]]
        mat_emb = wavlm_embs[matrix_offset[i]]
        sim_whole = cosine(whole_emb, ref_emb)
        sim_matrix = cosine(mat_emb, ref_emb)
        sim_emb_mean = float(statistics.mean(sim_emb_per_phrase)) if sim_emb_per_phrase else 0.0

        result = {
            "entry_id": entry["id"],
            "mode": mode,
            "setup": entry.get("setup"),
            "base_lang": base_lang,
            "phrase_lang": phrase_lang,
            "n_phrases": len(phrases),
            "audio_s": round(dur, 3),

            "asr_auto_text": auto["text"],
            "asr_auto_lang": auto["lang"],
            "asr_forced_text": forced["text"],
            "embedded_asr_texts": embedded_asr_texts,
            "auto_seg_lang_per_phrase": auto_seg_lang,

            "phrase_time_spans": spans,
            "phrase_span_methods": methods,

            "mWER": float(mr["mWER"]),
            "eWER": float(ewer),
            "MER": float(mer),
            "mWER_unit": "WER" if is_word_metric(base_lang) else "CER",
            "eWER_unit": "WER" if is_word_metric(phrase_lang) else "CER",
            "matrix_units": int(mr["m_units"]),
            "embedded_units": int(e_units_total),
            "embedded_wer_per_phrase": [float(x) for x in embedded_wers],

            "LA_segment_per_phrase": [float(x) for x in la_seg],
            "LA_segment_per_wav": float(statistics.mean(la_seg)) if la_seg else 0.0,

            "LID_conf_per_phrase": [float(x) for x in lid_conf],
            "LID_conf_per_wav": float(statistics.mean(lid_conf)) if lid_conf else 0.0,

            "SIM_whole": float(sim_whole),
            "SIM_matrix": float(sim_matrix),
            "SIM_embedded_per_phrase": [float(x) for x in sim_emb_per_phrase],
            "SIM_embedded_per_wav": float(sim_emb_mean),

            "quality_tier": entry.get("meta", {}).get("quality_tier"),
            "success": True,
            "error": None,
        }
        results.append(result)

    # Override FA-failed wavs with explicit synth-failure records so they do
    # not silently pollute aggregate metrics. The Whisper/WavLM passes above
    # ran on placeholder spans for these indices; their numeric outputs are
    # meaningless and must not be used.
    if fa_failed_idx:
        for i, err_msg in fa_failed_idx.items():
            entry, mode, _wav, _audio, dur = chunk_data[i]
            results[i] = {
                "entry_id": entry["id"],
                "mode": mode,
                "setup": entry.get("setup"),
                "base_lang": entry.get("base_lang"),
                "phrase_lang": entry.get("phrase_lang"),
                "n_phrases": len(entry.get("phrases", [])),
                "audio_s": round(dur, 3),
                "asr_auto_text": None,
                "asr_auto_lang": None,
                "asr_forced_text": None,
                "embedded_asr_texts": None,
                "auto_seg_lang_per_phrase": None,
                "phrase_time_spans": None,
                "phrase_span_methods": None,
                "mWER": None, "eWER": None, "MER": None,
                "mWER_unit": None, "eWER_unit": None,
                "matrix_units": None, "embedded_units": None,
                "embedded_wer_per_phrase": None,
                "LA_segment_per_phrase": None, "LA_segment_per_wav": None,
                "LID_conf_per_phrase": None, "LID_conf_per_wav": None,
                "SIM_whole": None, "SIM_matrix": None,
                "SIM_embedded_per_phrase": None, "SIM_embedded_per_wav": None,
                "quality_tier": entry.get("meta", {}).get("quality_tier"),
                "success": False,
                "error": f"synth_failure: {err_msg}",
            }
    return results


# --- Manifest + IO ------------------------------------------------------------

def load_manifest(manifest_path: Optional[Path] = None) -> List[dict]:
    p = Path(manifest_path) if manifest_path else MANIFEST
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def per_wav_path(entry_id: str, mode: str) -> Path:
    return PER_WAV_DIR / f"{entry_id}__{mode}.json"


def log_usage(payload: dict):
    try:
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# --- Aggregates ---------------------------------------------------------------

CSV_COLS = [
    "entry_id", "mode", "setup", "base_lang", "phrase_lang", "n_phrases",
    "quality_tier", "audio_s",
    "mWER", "eWER", "MER", "mWER_unit", "eWER_unit",
    "matrix_units", "embedded_units",
    "LA_segment_per_wav", "LID_conf_per_wav",
    "SIM_whole", "SIM_matrix", "SIM_embedded_per_wav",
    "success",
]


def write_csv(rows: List[dict]):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLS})


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (0.0, 0.0, 0)
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def write_report(rows: List[dict]):
    by_mode = defaultdict(list)
    by_setup_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
        by_setup_mode[(r.get("setup"), r["mode"])].append(r)

    lines = []
    lines.append("# Extended CS Eval Report — Synthetic CS\n\n")
    lines.append("Per-wav metrics: mWER, eWER, MER, LA(segment), LID conf, SIM-o (whole/matrix/embedded).\n\n")
    lines.append("## Per-mode aggregates\n\n")
    lines.append("| Mode | n | mWER | eWER | MER | LA_seg | LID_conf | SIM_whole | SIM_matrix | SIM_emb |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for mode in MODES:
        rs = by_mode.get(mode, [])
        m_m, _, n = _stats([r["mWER"] for r in rs])
        e_m, _, _ = _stats([r["eWER"] for r in rs])
        mer_m, _, _ = _stats([r["MER"] for r in rs])
        la_m, _, _ = _stats([r["LA_segment_per_wav"] for r in rs])
        lid_m, _, _ = _stats([r["LID_conf_per_wav"] for r in rs])
        sw, _, _ = _stats([r["SIM_whole"] for r in rs])
        sm, _, _ = _stats([r["SIM_matrix"] for r in rs])
        se, _, _ = _stats([r["SIM_embedded_per_wav"] for r in rs])
        lines.append(
            f"| {mode} | {n} | {m_m:.4f} | {e_m:.4f} | {mer_m:.4f} | {la_m:.4f} | {lid_m:.4f} | {sw:.4f} | {sm:.4f} | {se:.4f} |\n"
        )

    lines.append("\n## Per-setup × mode (mean MER / LID / SIM_whole)\n\n")
    setups = sorted({r.get("setup") for r in rows if r.get("setup")})
    lines.append("| Setup | Mode | n | mWER | eWER | MER | LID | SIM_whole | SIM_matrix | SIM_emb |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in setups:
        for mode in MODES:
            rs = by_setup_mode.get((s, mode), [])
            if not rs:
                continue
            m_m, _, n = _stats([r["mWER"] for r in rs])
            e_m, _, _ = _stats([r["eWER"] for r in rs])
            mer_m, _, _ = _stats([r["MER"] for r in rs])
            lid_m, _, _ = _stats([r["LID_conf_per_wav"] for r in rs])
            sw, _, _ = _stats([r["SIM_whole"] for r in rs])
            sm, _, _ = _stats([r["SIM_matrix"] for r in rs])
            se, _, _ = _stats([r["SIM_embedded_per_wav"] for r in rs])
            lines.append(
                f"| {s} | {mode} | {n} | {m_m:.4f} | {e_m:.4f} | {mer_m:.4f} | {lid_m:.4f} | {sw:.4f} | {sm:.4f} | {se:.4f} |\n"
            )

    lines.append("\n## Δ vs baseline_src_only (per setup, mean MER / eWER / LID / SIM_whole)\n\n")
    base_means = {}
    for s in setups:
        rs = by_setup_mode.get((s, "baseline_src_only"), [])
        if rs:
            base_means[s] = {
                "mWER": _stats([r["mWER"] for r in rs])[0],
                "eWER": _stats([r["eWER"] for r in rs])[0],
                "MER": _stats([r["MER"] for r in rs])[0],
                "LID": _stats([r["LID_conf_per_wav"] for r in rs])[0],
                "SIM": _stats([r["SIM_whole"] for r in rs])[0],
            }
    lines.append("| Setup | Mode | Δ mWER | Δ eWER | Δ MER | Δ LID | Δ SIM_whole |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for s in setups:
        bm = base_means.get(s)
        if not bm:
            continue
        for mode in MODES:
            if mode == "baseline_src_only":
                continue
            rs = by_setup_mode.get((s, mode), [])
            if not rs:
                continue
            dm = _stats([r["mWER"] for r in rs])[0] - bm["mWER"]
            de = _stats([r["eWER"] for r in rs])[0] - bm["eWER"]
            dmer = _stats([r["MER"] for r in rs])[0] - bm["MER"]
            dlid = _stats([r["LID_conf_per_wav"] for r in rs])[0] - bm["LID"]
            dsim = _stats([r["SIM_whole"] for r in rs])[0] - bm["SIM"]
            lines.append(
                f"| {s} | {mode} | {dm:+.4f} | {de:+.4f} | {dmer:+.4f} | {dlid:+.4f} | {dsim:+.4f} |\n"
            )

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


# --- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap pending tasks (sanity).")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Wav-level chunk size (Pass B). Default 16.")
    ap.add_argument("--modes-filter", type=str, default=None,
                    help="Comma-separated mode names to RESTRICT evaluation to "
                         "(intersect with auto-discovered MODES). Useful when a "
                         "synth dir contains extra modes that should be ignored "
                         "for the current ablation. Default: evaluate all "
                         "discovered modes.")
    ap.add_argument("--fa-aligner-python", type=str,
                    default=os.environ.get("QWEN_PY", "python"),
                    help="Python executable for the Qwen3-FA subprocess "
                         "(must live in an env with Qwen3-ForcedAligner "
                         "installed). Defaults to $QWEN_PY, then 'python'.")
    ap.add_argument("--fa-aligner-model", type=str,
                    default="Qwen/Qwen3-ForcedAligner-0.6B",
                    help="HF model id for the Qwen3 forced aligner.")
    ap.add_argument("--fa-aligner-dtype", type=str, default="bfloat16",
                    choices=["auto", "float16", "float32", "bfloat16"],
                    help="dtype passed to the Qwen3-FA worker.")
    ap.add_argument("--fa-aligner-attn-implementation", type=str, default=None,
                    help="Optional attn_implementation override for Qwen3-FA "
                         "(e.g. flash_attention_2). Defaults to library default.")
    ap.add_argument("--fa-disable", action="store_true", default=False,
                    help="Skip the Qwen3 FA pass. Phrase-level metrics "
                         "(LA_e / LID_e / MER_e) REQUIRE forced alignment, so "
                         "every wav is then marked as a failure; use only for "
                         "debugging the non-FA passes, never for reported runs.")
    ap.add_argument("--seg-batch", type=int, default=32,
                    help="Phrase-segment Whisper batch (Pass C/D, SDPA). Default 32.")
    ap.add_argument("--wavlm-batch", type=int, default=16,
                    help="WavLM batch size. Default 16.")
    ap.add_argument("--num-shards", type=int, default=8,
                    help="Total shards across GPUs (default 8 for B200 x 8). "
                    "Tasks are split by task_idx %% num_shards.")
    ap.add_argument("--gpu-shard", type=int, default=None,
                    help="Process tasks where task_idx %% num_shards == this. "
                    "Omit to process all tasks in this single invocation.")
    ap.add_argument("--shard-name", type=str, default="",
                    help="Log prefix tag (e.g. GPU6).")
    ap.add_argument("--device-index", type=int, default=0,
                    help="CUDA device index within visible GPUs.")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Path to manifest JSONL. Default: <repo-root>/benchmark/manifest.jsonl. "
                    "Relative paths are resolved against the repo root.")
    ap.add_argument("--synth-dir", type=str, default="synth",
                    help="Source wav directory (relative to repo root or absolute).")
    ap.add_argument("--eval-dir", type=str, default=None,
                    help="Eval output directory. Defaults to 'eval_<suffix>' "
                    "where --synth-dir=synth_<suffix>. Absolute paths accepted.")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip wavs that already have an output JSON (default: on).")
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--only-aggregate", action="store_true",
                    help="Skip processing; only build CSV + report from existing JSONs.")
    args = ap.parse_args()

    # Validate --gpu-shard against --num-shards.
    if args.gpu_shard is not None and not (0 <= args.gpu_shard < args.num_shards):
        ap.error(f"--gpu-shard must be in [0, {args.num_shards}); got {args.gpu_shard}")

    # ---- Resolve synth-dir / eval-dir and override module-level globals ----
    global SYNTH_DIR, EVAL_DIR, PER_WAV_DIR, USAGE_LOG, SUMMARY_CSV, REPORT_MD, MODES
    _synth = Path(args.synth_dir)
    SYNTH_DIR = _synth if _synth.is_absolute() else (REPO_ROOT / args.synth_dir)
    if args.eval_dir:
        _eval = Path(args.eval_dir)
        EVAL_DIR = _eval if _eval.is_absolute() else (REPO_ROOT / args.eval_dir)
    else:
        synth_name = SYNTH_DIR.name
        if synth_name == "synth":
            eval_name = "eval"
        elif synth_name.startswith("synth_"):
            eval_name = "eval_" + synth_name[len("synth_"):]
        else:
            eval_name = "eval_" + synth_name
        EVAL_DIR = REPO_ROOT / eval_name
    PER_WAV_DIR = EVAL_DIR / "extended"
    USAGE_LOG = EVAL_DIR / "extended_usage_log.jsonl"
    SUMMARY_CSV = EVAL_DIR / "extended_summary.csv"
    REPORT_MD = EVAL_DIR / "extended_report.md"
    PER_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Auto-discover MODES from the resolved SYNTH_DIR ----
    MODES = discover_modes_from_synth_dir(SYNTH_DIR)
    if not MODES:
        print(f"[eval_extended] WARNING: no wavs found under {SYNTH_DIR}", flush=True)
    else:
        print(f"[eval_extended] discovered {len(MODES)} modes: {MODES}", flush=True)
    if args.modes_filter:
        requested = [m.strip() for m in args.modes_filter.split(",") if m.strip()]
        filtered = [m for m in MODES if m in requested]
        missing = [m for m in requested if m not in MODES]
        if missing:
            print(
                f"[eval_extended] --modes-filter requested {missing} but none "
                f"present in synth dir; ignoring those.",
                flush=True,
            )
        MODES = filtered
        print(
            f"[eval_extended] after --modes-filter: {len(MODES)} modes -> {MODES}",
            flush=True,
        )
    print(
        f"[eval_extended] synth_dir={SYNTH_DIR} eval_dir={EVAL_DIR} "
        f"num_shards={args.num_shards} gpu_shard={args.gpu_shard}",
        flush=True,
    )

    tag = f"[{args.shard_name}] " if args.shard_name else ""
    device = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"

    entries = load_manifest(getattr(args, "manifest", None))
    print(f"{tag}[{time.strftime('%H:%M:%S')}] Loaded manifest: {len(entries)} entries.", flush=True)

    # Build full task list in deterministic (entry, mode) order so sharding
    # divides the same way across both processes.
    all_tasks = []
    for entry in entries:
        for mode in MODES:
            wav = SYNTH_DIR / entry["id"] / f"{mode}.wav"
            if wav.exists():
                all_tasks.append((entry, mode, wav))
    print(f"{tag}  {len(all_tasks)} wav files found.", flush=True)

    if args.gpu_shard is not None:
        my_tasks = [
            t for i, t in enumerate(all_tasks)
            if i % args.num_shards == args.gpu_shard
        ]
        print(
            f"{tag}  GPU shard {args.gpu_shard}/{args.num_shards}: "
            f"{len(my_tasks)}/{len(all_tasks)} tasks.",
            flush=True,
        )
    else:
        my_tasks = all_tasks

    rows: List[dict] = []
    pending = []
    for entry, mode, wav in my_tasks:
        out_path = per_wav_path(entry["id"], mode)
        if args.skip_existing and out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
                continue
            except Exception:
                pass
        pending.append((entry, mode, wav))
    print(f"{tag}  already done: {len(rows)}; pending: {len(pending)}", flush=True)

    if args.only_aggregate:
        rows_by_key = {(r["entry_id"], r["mode"]): r for r in rows}
        for entry, mode, _wav in all_tasks:
            key = (entry["id"], mode)
            if key in rows_by_key:
                continue
            p = per_wav_path(entry["id"], mode)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        rows_by_key[key] = json.load(f)
                except Exception:
                    pass
        final_rows = list(rows_by_key.values())
        write_csv(final_rows)
        write_report(final_rows)
        print(f"{tag}[{time.strftime('%H:%M:%S')}] Aggregates: {SUMMARY_CSV}, {REPORT_MD} "
              f"({len(final_rows)} rows).", flush=True)
        return

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"{tag}  --limit applied: {len(pending)} to process.", flush=True)

    if not pending:
        print(f"{tag}[{time.strftime('%H:%M:%S')}] Nothing to do.", flush=True)
        write_csv(rows)
        write_report(rows)
        return

    # ---- Pass F: Qwen3 forced alignment (one subprocess for the entire shard) ----
    fa_cache: Dict[str, List[Dict[str, Any]]] = {}
    if not args.fa_disable:
        samples: List[Dict[str, str]] = []
        sample_keys: List[str] = []
        for entry, _mode, wav in pending:
            text = entry.get("text", "")
            base_lang_code = entry.get("base_lang", "en")
            try:
                qwen_lang = resolve_qwen_aligner_language(base_lang_code)
            except ValueError as e:
                print(f"{tag}[FA-SKIP] {entry['id']}: {e}", flush=True)
                continue
            samples.append({
                "audio": str(wav),
                "text": text,
                "language": qwen_lang,
            })
            sample_keys.append(str(wav))
        if samples:
            print(
                f"{tag}[{time.strftime('%H:%M:%S')}] FA: aligning "
                f"{len(samples)} wavs via {args.fa_aligner_model}...",
                flush=True,
            )
            t0 = time.time()
            try:
                items_per, errors = run_qwen_fa_batch(args, samples)
            except Exception as e:
                print(f"{tag}[ERR-FA] batch failed: {e}", flush=True)
                traceback.print_exc()
                items_per, errors = [], []
            elapsed = time.time() - t0
            n_err = 0
            for key, items, err in zip(sample_keys, items_per, errors):
                if err is None and items:
                    fa_cache[key] = items
                else:
                    n_err += 1
            print(
                f"{tag}[{time.strftime('%H:%M:%S')}] FA done in {elapsed:.0f}s. "
                f"cached {len(fa_cache)}/{len(samples)} (errors={n_err}).",
                flush=True,
            )
        else:
            print(f"{tag}[{time.strftime('%H:%M:%S')}] FA: no eligible samples.", flush=True)
    else:
        print(f"{tag}[{time.strftime('%H:%M:%S')}] FA disabled (--fa-disable); "
              f"using proportional spans only.", flush=True)

    print(f"{tag}[{time.strftime('%H:%M:%S')}] Loading models on {device}...", flush=True)
    t0 = time.time()
    whisper = WhisperEngine(MODEL_ID, device)
    print(f"{tag}  Whisper loaded in {time.time()-t0:.1f}s.", flush=True)
    t0 = time.time()
    wavlm = WavLMEngine(WAVLM_ID, device)
    print(f"{tag}  WavLM-sv loaded in {time.time()-t0:.1f}s.", flush=True)
    if torch.cuda.is_available():
        print(f"{tag}  GPU mem allocated: {torch.cuda.memory_allocated(args.device_index)/1024**3:.2f} GB", flush=True)

    log_usage({
        "event": "start", "ts": time.time(), "pending": len(pending),
        "limit": args.limit, "shard": args.gpu_shard, "shard_name": args.shard_name,
        "batch_size": args.batch_size, "seg_batch": args.seg_batch,
        "wavlm_batch": args.wavlm_batch,
    })

    start = time.time()
    done = 0
    last_log = start
    full_batch = max(1, args.batch_size)
    seg_batch = max(1, args.seg_batch)
    wavlm_batch = max(1, args.wavlm_batch)

    for bi in range(0, len(pending), full_batch):
        chunk = pending[bi : bi + full_batch]
        t_chunk = time.time()
        chunk_data = []
        for entry, mode, wav in chunk:
            try:
                audio, dur = load_audio(wav)
                chunk_data.append((entry, mode, wav, audio, dur))
            except Exception as e:
                print(f"{tag}[ERR-LOAD] {entry['id']}/{mode}: {e}", flush=True)
        if not chunk_data:
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            chunk_results = process_chunk(
                chunk_data, whisper, wavlm,
                full_batch=full_batch, seg_batch=seg_batch, wavlm_batch=wavlm_batch,
                fa_cache=fa_cache,
            )
        except Exception as e:
            print(f"{tag}[ERR-CHUNK] @{bi}: {e}", flush=True)
            traceback.print_exc()
            chunk_results = []
            for (entry, mode, _w, _a, dur) in chunk_data:
                chunk_results.append({
                    "entry_id": entry["id"], "mode": mode, "setup": entry.get("setup"),
                    "base_lang": entry.get("base_lang"), "phrase_lang": entry.get("phrase_lang"),
                    "n_phrases": len(entry.get("phrases", [])),
                    "audio_s": round(dur, 3),
                    "mWER": None, "eWER": None, "MER": None,
                    "LA_segment_per_wav": None, "LID_conf_per_wav": None,
                    "SIM_whole": None, "SIM_matrix": None, "SIM_embedded_per_wav": None,
                    "quality_tier": entry.get("meta", {}).get("quality_tier"),
                    "success": False, "error": f"{type(e).__name__}: {e}",
                })

        for (entry, mode, _wav, _audio, _dur), res in zip(chunk_data, chunk_results):
            out_path = per_wav_path(entry["id"], mode)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            rows.append(res)
            done += 1
            log_usage({
                "event": "wav_done", "ts": time.time(),
                "entry_id": entry["id"], "mode": mode,
                "success": res.get("success", True),
                "shard": args.gpu_shard,
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        now = time.time()
        if done <= full_batch * 2 or (now - last_log) > 30.0:
            last_log = now
            rate = done / max(now - start, 1e-3)
            remaining = len(pending) - done
            eta_min = remaining / max(rate, 1e-3) / 60.0
            chunk_dt = now - t_chunk
            print(
                f"{tag}[{time.strftime('%H:%M:%S')}] {done}/{len(pending)} "
                f"(chunk={len(chunk_data)} in {chunk_dt:.1f}s, "
                f"{rate:.2f} wav/s, ETA {eta_min:.1f} min)",
                flush=True,
            )

    print(f"{tag}[{time.strftime('%H:%M:%S')}] Building aggregates...", flush=True)
    rows_by_key = {(r["entry_id"], r["mode"]): r for r in rows}
    for entry, mode, _wav in all_tasks:
        key = (entry["id"], mode)
        if key in rows_by_key:
            continue
        p = per_wav_path(entry["id"], mode)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rows_by_key[key] = json.load(f)
            except Exception:
                pass
    final_rows = list(rows_by_key.values())
    write_csv(final_rows)
    write_report(final_rows)
    print(f"{tag}[{time.strftime('%H:%M:%S')}] Wrote {SUMMARY_CSV}", flush=True)
    print(f"{tag}[{time.strftime('%H:%M:%S')}] Wrote {REPORT_MD}", flush=True)
    print(f"{tag}[{time.strftime('%H:%M:%S')}] Total rows: {len(final_rows)}", flush=True)
    log_usage({"event": "done", "ts": time.time(), "processed": done,
               "total_rows": len(final_rows), "shard": args.gpu_shard})


if __name__ == "__main__":
    main()


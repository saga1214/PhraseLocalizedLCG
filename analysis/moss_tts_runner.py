"""Synthesize our CS benchmark with MOSS-TTS-v1.5 (external baseline).

Reads manifest entries (id, setup, text, ref_audio_path, base_lang), clones
the per-direction reference voice, tags the MATRIX language (MOSS-TTS accepts
exactly one language tag per utterance — the same conditioning granularity as
our unguided baseline), and writes 16 kHz mono wavs in a layout the eval
pipeline consumes:  <out-dir>/<entry_id>/moss_tts_v15.wav

Run inside the `moss-tts` conda env:
    python analysis/moss_tts_runner.py --manifest benchmark/manifest.jsonl \
        --out-dir _artifacts/mosstts/synth_mosstts --setups J_to_E,K_to_E [--limit 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
WAV_NAME = "moss_tts_v15.wav"
LANG_NAME = {"en": "English", "de": "German", "fr": "French",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="benchmark/manifest.jsonl")
    ap.add_argument("--out-dir", type=str, default="_artifacts/mosstts/synth_mosstts")
    ap.add_argument("--setups", type=str, default="",
                    help="Comma-separated 'setup' filter (default: all).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max entries per setup (0 = all).")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--sr-out", type=int, default=16000)
    ap.add_argument("--max-new-tokens", type=int, default=1500,
                    help="~120 s cap at 12.5 frames/s.")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    root = manifest.parent
    entries = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if args.setups:
        keep = {s.strip() for s in args.setups.split(",") if s.strip()}
        entries = [e for e in entries if e["setup"] in keep]
    if args.limit > 0:
        by_setup: dict[str, int] = {}
        capped = []
        for e in entries:
            n = by_setup.get(e["setup"], 0)
            if n < args.limit:
                capped.append(e)
                by_setup[e["setup"]] = n + 1
        entries = capped
    entries = entries[args.shard::args.num_shards]
    log.info("Selected %d entries (shard %d/%d)",
             len(entries), args.shard, args.num_shards)

    device = "cuda"
    torch.backends.cuda.enable_cudnn_sdp(False)  # per MOSS-TTS README
    log.info("Loading %s ...", MODEL_ID)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc.audio_tokenizer = proc.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    sr_model = proc.model_config.sampling_rate
    log.info("Model loaded (sampling_rate=%d).", sr_model)

    out_root = Path(args.out_dir)
    n_done = n_fail = 0
    for i, e in enumerate(entries):
        out = out_root / e["id"] / WAV_NAME
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        language = LANG_NAME.get(e.get("base_lang", ""), None)
        t0 = time.perf_counter()
        try:
            conv = [proc.build_user_message(
                text=e["text"],
                reference=[str(root / e["ref_audio_path"])],
                language=language,
            )]
            batch = proc([conv], mode="generation")
            with torch.no_grad():
                outs = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_new_tokens=args.max_new_tokens,
                    audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
                    audio_repetition_penalty=1.0,
                )
            audio = proc.decode(outs)[0].audio_codes_list[0]
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            audio = audio.detach().cpu().to(torch.float32)
            if args.sr_out != sr_model:
                audio = torchaudio.functional.resample(
                    audio, sr_model, args.sr_out)
            # Write atomically: a kill mid-save must not leave a truncated
            # wav that the resume check would silently accept.
            tmp = out.with_name(out.stem + ".tmp.wav")
            torchaudio.save(str(tmp), audio, args.sr_out)
            tmp.replace(out)
            n_done += 1
            log.info("[%d/%d] %s: %.1fs audio in %.1fs",
                     i + 1, len(entries), e["id"],
                     audio.size(-1) / args.sr_out, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            log.error("[%d/%d] %s FAILED: %s", i + 1, len(entries), e["id"], exc)
    log.info("DONE synthesized=%d failed=%d -> %s", n_done, n_fail, out_root)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

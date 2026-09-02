#!/usr/bin/env python3
"""Subprocess worker for Qwen3 forced alignment.

Isolated from the main OmniVoice process so Qwen3-ASR can run in its own
conda env with its own transformers version.

Input JSON (single-sample and batch forms are both accepted):

  Single sample:
    {
      "audio": "<path or in-memory key>",
      "text": "...",
      "language": "...",
      "model": "...", "device": "...", "dtype": "...", "attn_implementation": ...
    }

  Batch:
    {
      "samples": [
        {"audio": "...", "text": "...", "language": "..."},
        ...
      ],
      "model": "...", "device": "...", "dtype": "...", "attn_implementation": ...
    }

Output JSON:
    {
      "items_per_sample": [[{...}, ...], ...],   # one list per input sample
      "sample_errors":    [null, "...", ...],    # parallel to items_per_sample
      "num_samples":      <int>,
      "model_load_sec":   <float>,
      "total_infer_sec":  <float>,
      "items":            [{...}, ...]           # mirrors items_per_sample[0]
    }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def _resolve_dtype(name: str):
    mapping = {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = _read_json(Path(args.input))

    # Accept either the batch form ({"samples": [...]} ) or the single-sample
    # shorthand ({"audio", "text", "language"}).
    samples = request.get("samples")
    if samples is None:
        samples = [
            {
                "audio": request["audio"],
                "text": request["text"],
                "language": request["language"],
            }
        ]
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Input must contain a non-empty 'samples' list (or audio/text/language).")

    from qwen_asr import Qwen3ForcedAligner

    kwargs = {
        "dtype": _resolve_dtype(request.get("dtype", "bfloat16")),
        "device_map": request.get("device") or "cuda:0",
    }
    attn_implementation = request.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    t_load_start = time.time()
    aligner = Qwen3ForcedAligner.from_pretrained(request["model"], **kwargs)
    t_load = time.time() - t_load_start

    items_per_sample: list[list[dict[str, Any]]] = []
    sample_errors: list[Any] = []
    t_infer_total = 0.0

    for i, sample in enumerate(samples):
        try:
            t0 = time.time()
            results = aligner.align(
                audio=sample["audio"],
                text=sample["text"],
                language=sample["language"],
            )
            t_infer_total += time.time() - t0
            if len(results) != 1:
                raise RuntimeError(
                    f"Expected one alignment result for sample {i}, got {len(results)}"
                )
            items = [
                {
                    "text": str(item.text),
                    "start_time": float(item.start_time),
                    "end_time": float(item.end_time),
                }
                for item in results[0]
            ]
            items_per_sample.append(items)
            sample_errors.append(None)
        except Exception as e:  # noqa: BLE001
            items_per_sample.append([])
            sample_errors.append(f"{type(e).__name__}: {e}")

    out_payload = {
        "items_per_sample": items_per_sample,
        "sample_errors": sample_errors,
        "num_samples": len(samples),
        "model_load_sec": t_load,
        "total_infer_sec": t_infer_total,
        # Convenience alias for single-sample callers.
        "items": items_per_sample[0] if items_per_sample else [],
    }
    _write_json(Path(args.output), out_payload)


if __name__ == "__main__":
    main()

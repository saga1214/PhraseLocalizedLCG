#!/usr/bin/env python3
"""Merge sample-shard outputs from ``cs_attention_probe_dataset.py``.

When the dataset probe is run with ``--num-sample-shards N --sample-shard k``
each shard processes a disjoint subset of the requested samples and writes
its own ``metrics.json`` + ``summary.csv``.  This script stitches N such
shard directories back together into a single merged output by:

  1. Concatenating per_sample_raw across shards (each shard sets it because
     num_sample_shards > 1 forces save_per_sample even if the user didn't).
  2. Re-running the per-setup aggregation logic (utterance-mean -> mean / std /
     percentiles, plus the per-step trajectory mean) over the union of all
     samples.
  3. Re-ranking setups by cross-dataset min_f1_mean.

Usage:
    python examples/merge_xlang_probe.py \\
        --input-dirs <shard0_out> <shard1_out> ... \\
        --output-dir <merged_out>

The merged ``metrics.json`` reuses the schema of the single-shard output, so
downstream notebooks / pandas pipelines need no changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

# Import shared helpers from the probe driver so aggregation math stays in
# lockstep with the single-shard code path.
from attention_probe_dataset import (  # noqa: E402
    METRIC_KEYS,
    _trajectory_summary,
)
from attention_probe import _sanitize_for_json  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("merge_xlang_probe")


def _load_metrics(in_dir: Path) -> dict[str, Any]:
    f = in_dir / "metrics.json"
    if not f.is_file():
        raise FileNotFoundError(f"Missing {f}")
    with f.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _aggregate_one_dataset(
    setup_names: list[str],
    per_sample_word_mean: list[dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, Any]]:
    """Mirror of the per-setup aggregation in cs_attention_probe_dataset.py."""
    aggregate: dict[str, dict[str, Any]] = {}
    for setup in setup_names:
        per_metric_utt: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
        traj_matrix: dict[str, list[list[float]]] = {k: [] for k in METRIC_KEYS}
        for sample in per_sample_word_mean:
            if setup not in sample:
                continue
            for metric in METRIC_KEYS:
                traj = sample[setup][metric] or []
                # _sanitize_for_json writes None for NaN/Inf -- treat None as NaN.
                normalized = [
                    float("nan") if v is None else float(v) for v in traj
                ]
                traj_matrix[metric].append(normalized)
                cleaned = [v for v in normalized if not math.isnan(v)]
                if cleaned:
                    per_metric_utt[metric].append(float(np.mean(cleaned)))
        agg_setup: dict[str, Any] = {}
        for metric, values in per_metric_utt.items():
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0:
                agg_setup[f"{metric}_mean"] = float("nan")
                agg_setup[f"{metric}_std"] = float("nan")
                agg_setup[f"{metric}_p25"] = float("nan")
                agg_setup[f"{metric}_p50"] = float("nan")
                agg_setup[f"{metric}_p75"] = float("nan")
                agg_setup[f"{metric}_n"] = 0
            else:
                agg_setup[f"{metric}_mean"] = float(arr.mean())
                agg_setup[f"{metric}_std"] = float(arr.std())
                agg_setup[f"{metric}_p25"] = float(np.percentile(arr, 25))
                agg_setup[f"{metric}_p50"] = float(np.percentile(arr, 50))
                agg_setup[f"{metric}_p75"] = float(np.percentile(arr, 75))
                agg_setup[f"{metric}_n"] = int(arr.size)
        for metric, mats in traj_matrix.items():
            if not mats:
                agg_setup[f"{metric}_per_step_mean"] = []
                agg_setup[f"{metric}_step_summary"] = _trajectory_summary([])
                continue
            max_S = max(len(t) for t in mats)
            full = np.full((len(mats), max_S), np.nan)
            for r, t in enumerate(mats):
                full[r, : len(t)] = t
            per_step_mean = np.nanmean(full, axis=0).tolist()
            agg_setup[f"{metric}_per_step_mean"] = [
                float(x) if not math.isnan(x) else float("nan")
                for x in per_step_mean
            ]
            agg_setup[f"{metric}_step_summary"] = _trajectory_summary(
                agg_setup[f"{metric}_per_step_mean"]
            )
        aggregate[setup] = agg_setup
    return aggregate


def _build_cross_rank(
    setup_names: list[str],
    datasets: list[str],
    per_dataset: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cross_rank: list[dict[str, Any]] = []
    for setup in setup_names:
        f1_vals: dict[str, float] = {}
        iou_vals: dict[str, float] = {}
        for ds in datasets:
            agg = per_dataset[ds]["per_setup_aggregate"].get(setup, {})
            f1_vals[ds] = float(agg.get("f1_mean", float("nan")))
            iou_vals[ds] = float(agg.get("iou_mean", float("nan")))
        valid_f1 = [v for v in f1_vals.values() if not math.isnan(v)]
        valid_iou = [v for v in iou_vals.values() if not math.isnan(v)]
        cross_rank.append({
            "setup": setup,
            "per_dataset_f1_mean": f1_vals,
            "per_dataset_iou_mean": iou_vals,
            "min_f1_mean": min(valid_f1) if valid_f1 else float("nan"),
            "avg_f1_mean": float(np.mean(valid_f1)) if valid_f1 else float("nan"),
            "min_iou_mean": min(valid_iou) if valid_iou else float("nan"),
            "avg_iou_mean": float(np.mean(valid_iou)) if valid_iou else float("nan"),
        })
    cross_rank.sort(
        key=lambda r: (-r["min_f1_mean"] if not math.isnan(r["min_f1_mean"]) else float("inf"))
    )
    return cross_rank


def _write_summary_csv(
    setup_names: list[str],
    datasets: list[str],
    per_dataset: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    csv_lines = [
        "setup,"
        + ",".join(
            f"{ds}_{m}_mean" for ds in datasets for m in METRIC_KEYS
        )
    ]
    for setup in setup_names:
        row = [setup]
        for ds in datasets:
            agg = per_dataset[ds]["per_setup_aggregate"].get(setup, {})
            for m in METRIC_KEYS:
                v = agg.get(f"{m}_mean", float("nan"))
                row.append(
                    f"{v:.4f}"
                    if (v is not None and not math.isnan(v))
                    else ""
                )
        csv_lines.append(",".join(row))
    out_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", type=str, nargs="+", required=True,
                    help="Two or more sample-shard output directories. Each "
                    "must contain a metrics.json with per_sample_raw populated.")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Destination directory for the merged metrics.json + summary.csv.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="If set, ignore missing input dirs / metrics.json instead of aborting.")
    args = ap.parse_args()

    in_dirs = [Path(p) for p in args.input_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging %d shard dir(s) -> %s", len(in_dirs), out_dir)

    shards: list[dict[str, Any]] = []
    for d in in_dirs:
        try:
            shards.append(_load_metrics(d))
        except FileNotFoundError as e:
            if args.allow_partial:
                logger.warning("Skipping missing dir: %s (%s)", d, e)
                continue
            raise
    if not shards:
        logger.error("No shard metrics loaded.")
        return 1

    # Sanity: setups + datasets agreement.
    setup_names = list(shards[0]["setups"])
    datasets = list(shards[0]["datasets"])
    for s in shards[1:]:
        if list(s["setups"]) != setup_names:
            raise ValueError(
                "Shards disagree on 'setups' list. All shards must be run with "
                "identical --setups."
            )
        if list(s["datasets"]) != datasets:
            raise ValueError("Shards disagree on 'datasets' list.")

    # Concat per_sample_raw per dataset, then re-aggregate.
    merged_per_dataset: dict[str, dict[str, Any]] = {}
    total_indices_per_ds: dict[str, list[int]] = {}
    sample_failures_per_ds: dict[str, list[dict[str, Any]]] = {}
    for ds in datasets:
        all_raw: list[dict[str, Any]] = []
        seen_idx: set[int] = set()
        indices_processed: list[int] = []
        sample_failures: list[dict[str, Any]] = []
        meta_template: dict[str, Any] = {}
        for s in shards:
            ds_block = s["per_dataset"][ds]
            if not meta_template:
                meta_template = {
                    "hf_id": ds_block.get("hf_id"),
                    "language": ds_block.get("language"),
                    "ref_idx": ds_block.get("ref_idx"),
                    "ref_text": ds_block.get("ref_text"),
                    "num_eval_requested": ds_block.get("num_eval_requested"),
                }
            raws = ds_block.get("per_sample_raw") or []
            if not raws:
                raise ValueError(
                    f"Shard at {s.get('config', {}).get('output_dir', '?')} "
                    f"has no per_sample_raw for dataset {ds!r}. Re-run shards "
                    "with --num-sample-shards > 1 (auto-enables save_per_sample) "
                    "or pass --save-per-sample True explicitly."
                )
            for r in raws:
                # idx uniqueness: skip dupes (e.g. if shards overlapped by accident).
                ri = int(r.get("idx", -1))
                if ri in seen_idx:
                    continue
                seen_idx.add(ri)
                all_raw.append(r)
                indices_processed.append(ri)
            sample_failures.extend(ds_block.get("sample_failures") or [])

        all_raw.sort(key=lambda r: int(r.get("idx", -1)))
        indices_processed = sorted(set(indices_processed))
        per_sample_word_mean = [r.get("word_mean_per_step", {}) for r in all_raw]

        aggregate = _aggregate_one_dataset(setup_names, per_sample_word_mean)

        merged_per_dataset[ds] = {
            **meta_template,
            "num_eval_completed": len(all_raw),
            "num_eval_failed": len(sample_failures),
            "sample_indices_processed": indices_processed,
            "sample_failures": sample_failures[:50],
            "per_setup_aggregate": aggregate,
            "per_sample_raw": all_raw,
        }
        total_indices_per_ds[ds] = indices_processed
        sample_failures_per_ds[ds] = sample_failures
        logger.info(
            "Dataset %s: merged %d samples from %d shards (failures: %d)",
            ds, len(all_raw), len(shards), len(sample_failures),
        )

    cross_rank = _build_cross_rank(setup_names, datasets, merged_per_dataset)

    # Carry over the first shard's config as a representative; record the
    # merge provenance.
    cfg = dict(shards[0].get("config", {}))
    cfg["__merged_from__"] = [str(d) for d in in_dirs]
    cfg["__merged_num_shards__"] = len(shards)

    final = {
        "config": cfg,
        "setups": setup_names,
        "datasets": datasets,
        "per_dataset": merged_per_dataset,
        "ranked_by_min_f1": cross_rank,
    }
    sanitized = _sanitize_for_json(final)
    (out_dir / "metrics.json").open("w", encoding="utf-8").write(
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False)
    )
    logger.info("Saved %s", out_dir / "metrics.json")

    _write_summary_csv(setup_names, datasets, merged_per_dataset, out_dir / "summary.csv")
    logger.info("Saved %s", out_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

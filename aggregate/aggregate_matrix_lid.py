"""Aggregate matrix-region LID JSONL into per-direction CSVs.

Reads `<matrix-dir>/matrix_lid.jsonl` for each requested cell tag and writes:
    <out-dir>/matrix_lid_comparison.csv          cell × mode × direction × stats
    <out-dir>/matrix_lid_comparison_overall.csv  cell × mode (averaged)

Columns: cell, mode_label, lambda_eff, direction, n,
         matrix_LID_base_mean, matrix_LID_phr_mean,
         matrix_LID_top_is_base_rate, matrix_LID_top_is_phr_rate.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mode key -> (display label, effective λ).
MODE_LABELS = {
    "baseline_src_only":                                       ("no-CS", 0),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda3":    ("λ=3",   3),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda5":    ("λ=5",   5),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda7":    ("λ=7",   7),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda9":    ("λ=9",   9),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda11":   ("λ=11", 11),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda13":   ("λ=13", 13),
}

DIRECTIONS = [
    "J_to_E", "K_to_E", "E_to_J", "E_to_K",
    "DE_to_JA", "DE_to_KO", "FR_to_JA", "FR_to_KO",
    "JA_to_DE", "KO_to_DE", "JA_to_FR", "KO_to_FR",
]
DIR_ORDER = ["Overall"] + DIRECTIONS


def direction_of(entry_id: str) -> str:
    """syncs_<HOST>_to_<EMB>_<idx> -> '<HOST>_to_<EMB>'."""
    parts = entry_id.split("_")
    if len(parts) >= 4 and parts[0] == "syncs":
        return f"{parts[1]}_to_{parts[3]}"
    return "unknown"


def load(matrix_dir: Path) -> list[dict]:
    """Read the merged ``matrix_lid.jsonl`` if present, else union all
    per-shard ``matrix_lid.shard*.jsonl`` files."""
    out: list[dict] = []
    merged = matrix_dir / "matrix_lid.jsonl"
    paths = [merged] if merged.is_file() else sorted(matrix_dir.glob("matrix_lid.shard*.jsonl"))
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def aggregate(records: list[dict], cell_name: str) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        d = direction_of(r["entry_id"])
        buckets[(r["mode"], d)].append(r)
        buckets[(r["mode"], "Overall")].append(r)
    rows: list[dict] = []
    for (mode, d), recs in buckets.items():
        if mode not in MODE_LABELS:
            continue
        label, lam = MODE_LABELS[mode]
        lid_base = [r["matrix_LID_base"] for r in recs]
        lid_phr  = [r["matrix_LID_phr"]  for r in recs]
        top_base_rate = sum(1 for r in recs if r["matrix_LID_top"] == r["base_lang"]) / len(recs)
        top_phr_rate  = sum(1 for r in recs if r["matrix_LID_top"] == r["phr_lang"])  / len(recs)
        rows.append({
            "cell":          cell_name,
            "mode_label":    label,
            "lambda_eff":    lam,
            "direction":     d,
            "n":             len(recs),
            "matrix_LID_base_mean":           round(mean(lid_base), 4),
            "matrix_LID_phr_mean":            round(mean(lid_phr),  4),
            "matrix_LID_top_is_base_rate":    round(top_base_rate, 4),
            "matrix_LID_top_is_phr_rate":     round(top_phr_rate,  4),
        })
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["m4_xu"],
                    help="Cell tags (corresponds to <matrix-prefix>_<cell>/matrix_lid.jsonl).")
    ap.add_argument("--matrix-prefix", default="matrix_lid_l8-12",
                    help="Prefix of the per-cell matrix-LID dir.")
    ap.add_argument("--out-dir", default="eval", help="Output directory (default './eval').")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [(c, REPO_ROOT / f"{args.matrix_prefix}_{c}") for c in args.cells]
    all_rows: list[dict] = []
    for cell_name, matrix_dir in cells:
        recs = load(matrix_dir)
        print(f"{cell_name}: {len(recs)} records ({matrix_dir})")
        all_rows.extend(aggregate(recs, cell_name))

    cell_order = {c: i for i, (c, _) in enumerate(cells)}
    all_rows.sort(key=lambda r: (cell_order.get(r["cell"], 99), r["lambda_eff"],
                                 DIR_ORDER.index(r["direction"]) if r["direction"] in DIR_ORDER else 99))

    out_csv = out_dir / "matrix_lid_comparison.csv"
    with out_csv.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"wrote {out_csv} ({len(all_rows)} rows)")

    overall = [r for r in all_rows if r["direction"] == "Overall"]
    out2 = out_dir / "matrix_lid_comparison_overall.csv"
    with out2.open("w", newline="") as f:
        if overall:
            w = csv.DictWriter(f, fieldnames=list(overall[0].keys()))
            w.writeheader()
            w.writerows(overall)
    print(f"wrote {out2} ({len(overall)} rows)")

    print("\n=== Overall (averaged over 12 directions) ===")
    print(f"{'cell':<14} {'mode':<12} {'λ':>3}   {'LID_base':>9}  {'LID_phr':>8}  {'top=base':>9}  {'top=phr':>8}")
    for r in overall:
        print(f"{r['cell']:<14} {r['mode_label']:<12} {r['lambda_eff']:>3}   "
              f"{r['matrix_LID_base_mean']:>9.4f}  {r['matrix_LID_phr_mean']:>8.4f}  "
              f"{r['matrix_LID_top_is_base_rate']:>9.4f}  {r['matrix_LID_top_is_phr_rate']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

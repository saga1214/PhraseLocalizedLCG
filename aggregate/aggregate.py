"""Aggregate per-entry eval outputs into per-direction summary CSVs.

Reads, for each requested `--cell` tag:
    <repo>/<eval-prefix>_<cell>/extended_summary.csv   (MER + LA + LID)
    <repo>/<utmos-prefix>_<cell>/utmos.jsonl            (UTMOS)
    <repo>/<sim-prefix>_<cell>/sim_dual.jsonl           (SIM-w + SIM-me, WavLM + ECAPA)

Writes:
    <out-dir>/all_results_final.csv
    <out-dir>/synth_failure_blacklist.{json,csv}

The blacklist is the union of entry_ids that failed in any (cell, mode); it is
applied uniformly so every comparison is on the same surviving sample set.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mode key (eval-side filename without .wav) -> (display label, effective λ).
# Filenames follow the unified ``cs_attention_plc__<setup>__lambda{λ}.wav`` scheme
# written by the engine; variant choice (langdir vs hybrid) is hidden in the wav name.
MODE_LABEL = {
    "baseline_src_only":                                            ("no-CS",  0),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda3":         ("λ=3",    3),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda5":         ("λ=5",    5),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda7":         ("λ=7",    7),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda9":         ("λ=9",    9),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda11":        ("λ=11",  11),
    "cs_attention_plc__layers_set_8-12__max__a2t__lambda13":        ("λ=13",  13),
}
MODE_ORDER = [v[0] for v in MODE_LABEL.values()]

# 12 paper-final directions across the 5 languages.
DIRECTIONS = [
    "J_to_E", "K_to_E", "E_to_J", "E_to_K",
    "DE_to_JA", "DE_to_KO", "FR_to_JA", "FR_to_KO",
    "JA_to_DE", "KO_to_DE", "JA_to_FR", "KO_to_FR",
]
DIR_ORDER = ["Overall"] + DIRECTIONS
TOTAL_PER_DIR = 100


def direction_of(entry_id: str) -> str:
    """syncs_<HOST>_to_<EMB>_<idx> -> '<HOST>_to_<EMB>'."""
    parts = entry_id.split("_")
    return f"{parts[1]}_to_{parts[3]}" if len(parts) >= 4 else "?"


def safe_mean(vs):
    vs = [v for v in vs if v is not None and v == v]
    return mean(vs) if vs else float("nan")


def fnum(s):
    if s in (None, "", "nan"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["base", "m4_xu"],
                    help="Cell tags to aggregate (corresponds to <prefix>_<cell>/ dirs).")
    ap.add_argument("--eval-prefix",  default="eval_l8-12",
                    help="Prefix of the per-cell eval dir (default 'eval_l8-12').")
    ap.add_argument("--utmos-prefix", default="utmos_l8-12",
                    help="Prefix of the per-cell utmos dir (default 'utmos_l8-12').")
    ap.add_argument("--sim-prefix",   default="sim_dual_l8-12",
                    help="Prefix of the per-cell sim dir (default 'sim_dual_l8-12').")
    ap.add_argument("--out-dir", default="eval", help="Output directory (default './eval').")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    cells = [(
        c,
        REPO_ROOT / f"{args.eval_prefix}_{c}",
        REPO_ROOT / f"{args.utmos_prefix}_{c}",
        REPO_ROOT / f"{args.sim_prefix}_{c}",
    ) for c in args.cells]

    # --- Pass 1: collect global blacklist (union of failures across cells/modes) ---
    blacklist_per_dir: dict[str, set[str]] = defaultdict(set)
    failure_records: list[dict] = []
    for cell, eval_dir, _, _ in cells:
        csv_path = eval_dir / "extended_summary.csv"
        if not csv_path.is_file():
            print(f"WARN: missing {csv_path}")
            continue
        for r in csv.DictReader(open(csv_path)):
            if r["mode"] not in MODE_LABEL:
                continue
            if str(r.get("success", "True")).lower() != "true":
                eid = r["entry_id"]
                d = direction_of(eid)
                blacklist_per_dir[d].add(eid)
                failure_records.append({
                    "cell":       cell,
                    "mode":       r["mode"],
                    "mode_label": MODE_LABEL[r["mode"]][0],
                    "direction":  d,
                    "entry_id":   eid,
                    "error":      r.get("error", ""),
                })

    global_blacklist: set[str] = set().union(*blacklist_per_dir.values()) if blacklist_per_dir else set()
    total_overall = TOTAL_PER_DIR * len(DIRECTIONS)
    direction_counts = {
        d: {
            "total":       TOTAL_PER_DIR,
            "blacklisted": len(blacklist_per_dir.get(d, set())),
            "effective":   TOTAL_PER_DIR - len(blacklist_per_dir.get(d, set())),
        } for d in DIRECTIONS
    }
    direction_counts["Overall"] = {
        "total":       total_overall,
        "blacklisted": len(global_blacklist),
        "effective":   total_overall - len(global_blacklist),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synth_failure_blacklist.json").write_text(
        json.dumps({
            "global_blacklist":        sorted(global_blacklist),
            "per_direction_blacklist": {d: sorted(eids) for d, eids in blacklist_per_dir.items()},
            "direction_counts":        direction_counts,
            "failure_records":         failure_records,
            "cells_aggregated":        [c[0] for c in cells],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "synth_failure_blacklist.csv").open("w", newline="") as f:
        if failure_records:
            w = csv.DictWriter(f, fieldnames=list(failure_records[0].keys()))
            w.writeheader()
            w.writerows(failure_records)
        else:
            f.write("(no failures)\n")

    print("=== Direction counts (effective N after union-blacklist) ===")
    for d in DIR_ORDER:
        c = direction_counts[d]
        print(f"  {d:<10}: {c['effective']:>4}/{c['total']:>4} effective ({c['blacklisted']} blacklisted)")

    # --- Pass 2: per-(cell, mode, direction) means ---
    rows_out: list[dict] = []
    for cell, eval_dir, utmos_dir, sim_dir in cells:
        csv_path = eval_dir / "extended_summary.csv"
        if not csv_path.is_file():
            continue
        rows = list(csv.DictReader(open(csv_path)))

        def _load_jsonl_dir(d: Path, stem: str) -> list[dict]:
            """Read the merged ``<stem>.jsonl`` if present, else union all
            per-shard ``<stem>.shard*.jsonl`` files."""
            rows_: list[dict] = []
            merged = d / f"{stem}.jsonl"
            paths = [merged] if merged.is_file() else sorted(d.glob(f"{stem}.shard*.jsonl"))
            for p in paths:
                for line in p.open():
                    line = line.strip()
                    if not line: continue
                    try:
                        rows_.append(json.loads(line))
                    except Exception:
                        pass
            return rows_

        utmos: dict[tuple[str, str], float] = {
            (d["entry_id"], d["mode"]): d["utmos"]
            for d in _load_jsonl_dir(utmos_dir, "utmos")
        }
        sim_dual: dict[tuple[str, str], dict] = {
            (d["entry_id"], d["mode"]): d
            for d in _load_jsonl_dir(sim_dir, "sim_dual")
        }

        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            if r["mode"] not in MODE_LABEL: continue
            eid = r["entry_id"]
            if eid in global_blacklist: continue
            if str(r.get("success", "True")).lower() != "true": continue
            d = direction_of(eid)
            buckets[(r["mode"], d)].append(r)
            buckets[(r["mode"], "Overall")].append(r)

        for mode_key, (label, lam) in MODE_LABEL.items():
            for d in DIR_ORDER:
                recs = buckets.get((mode_key, d), [])
                if not recs:
                    continue
                col = lambda c: safe_mean(fnum(r.get(c, "")) for r in recs)
                utm = [utmos.get((r["entry_id"], r["mode"]), float("nan")) for r in recs]
                sd = [sim_dual.get((r["entry_id"], r["mode"]), {}) for r in recs]
                sd_col = lambda k: safe_mean(rec.get(k, float("nan")) for rec in sd)
                rows_out.append({
                    "cell":          cell,
                    "mode_label":    label,
                    "lambda_eff":    lam,
                    "direction":     d,
                    "n":             len(recs),
                    "MER_m":         round(col("mWER"), 4),
                    "MER_e":         round(col("eWER"), 4),
                    "MER":           round(col("MER"),  4),
                    "LA_seg":        round(col("LA_segment_per_wav"), 4),
                    "LID_conf":      round(col("LID_conf_per_wav"),   4),
                    "UTMOS":         round(safe_mean(utm), 4),
                    "SIM_w_wavlm":   round(sd_col("SIM_w_wavlm"),  4),
                    "SIM_w_ecapa":   round(sd_col("SIM_w_ecapa"),  4),
                    "SIM_me_wavlm":  round(sd_col("SIM_me_wavlm"), 4),
                    "SIM_me_ecapa":  round(sd_col("SIM_me_ecapa"), 4),
                })

    cell_idx = {c[0]: i for i, c in enumerate(cells)}
    mode_idx = {m: i for i, m in enumerate(MODE_ORDER)}
    dir_idx  = {d: i for i, d in enumerate(DIR_ORDER)}
    rows_out.sort(key=lambda r: (cell_idx[r["cell"]], mode_idx[r["mode_label"]], dir_idx[r["direction"]]))

    out_path = out_dir / "all_results_final.csv"
    with out_path.open("w", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"\nwrote {out_path} ({len(rows_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

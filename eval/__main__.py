"""Unified evaluation entry-point.

Dispatches to the four metric modules — MER+LA+phrase-LID, matrix-region LID,
SIM (WavLM + ECAPA), and UTMOS — each of which preserves its own argparse
contract. The first positional argument selects which metric to run; all
remaining argv is forwarded verbatim.

Usage:
    python -m eval mer_la_lid  --synth-dir ... --gpu-shard 0
    python -m eval matrix_lid  --synth-dir ... --gpu-shard 0
    python -m eval sim         --synth-dir ... --gpu-shard 0
    python -m eval utmos       --synth-dir ... --gpu-shard 0
    python -m eval all         --synth-dir ... --gpu-shard 0
"""
from __future__ import annotations

import sys

_DISPATCH = {
    "mer_la_lid": "eval._mer_la_lid",
    "matrix_lid": "eval._matrix_lid",
    "sim":        "eval._sim",
    "utmos":      "eval._utmos",
}


def _run(metric: str, argv: list[str]) -> int:
    if metric not in _DISPATCH:
        sys.stderr.write(
            f"unknown metric '{metric}'. available: {', '.join(sorted(_DISPATCH))} | all\n"
        )
        return 2
    import importlib
    mod = importlib.import_module(_DISPATCH[metric])
    sys.argv = [f"eval.{metric}"] + argv
    rc = mod.main()
    return int(rc) if rc is not None else 0


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__ or "")
        return 2
    metric = sys.argv[1]
    argv = sys.argv[2:]
    if metric == "all":
        for m in ("mer_la_lid", "matrix_lid", "sim", "utmos"):
            rc = _run(m, argv)
            if rc != 0:
                return rc
        return 0
    return _run(metric, argv)


if __name__ == "__main__":
    raise SystemExit(main())

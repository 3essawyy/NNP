"""Score a ``results.txt`` (and optionally ``time.txt``) against a ground-truth file.

``python -m nnp.evaluation.score results.txt ground_truth.txt [time.txt]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nnp.data.io import load_ground_truth
from nnp.evaluation.metrics import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("time", type=Path, nargs="?")
    args = parser.parse_args(argv)

    pred = load_ground_truth(args.results)
    true = load_ground_truth(args.ground_truth)
    if pred.shape != true.shape:
        print(f"ERROR: {pred.size} predictions vs {true.size} labels")
        return 1
    ev = evaluate(true, pred)
    out: dict[str, object] = {
        "n": ev.n,
        "accuracy": round(ev.accuracy, 4),
        "macro_f1": round(ev.macro_f1, 4),
        "top_confusions": ev.top_confusions,
    }
    if args.time is not None:
        t = np.array([float(v) for v in args.time.read_text().split()])
        out["time_total_s"] = round(float(t.sum()), 3)
        out["time_mean_ms"] = round(float(t.mean()) * 1000, 3)
        out["time_max_ms"] = round(float(t.max()) * 1000, 3)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

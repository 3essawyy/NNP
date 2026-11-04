"""Run the experiment suite: ``uv run poe experiments [--only eda compare …] [--quick]``.

Needs the dataset (``uv run poe data``) and the ``research`` extra. Writes
``experiments/results/*.json``, ``report/figures/*.{pdf,png}`` and ``report/generated/*.tex``.
``final`` uses the committed ``models/model.pkl`` (run ``uv run poe train`` first).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

from nnp.experiments import ablation, compare, eda, final, robustness, tuning
from nnp.experiments.common import Data, load_data

STAGES: dict[str, Callable[[Data, bool], dict[str, Any]]] = {
    "eda": eda.run,
    "compare": compare.run,
    "ablation": ablation.run,
    "tuning": tuning.run,
    "robustness": robustness.run,
    "final": final.run,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--quick", action="store_true", help="fewer folds (smoke run)")
    args = parser.parse_args(argv)
    data = load_data()
    for name in args.only:
        t = time.perf_counter()
        print(f"[{name}]")
        STAGES[name](data, args.quick)
        print(f"[{name}] done in {time.perf_counter() - t:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

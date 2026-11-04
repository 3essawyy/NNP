"""Command-line entry point shared by ``infer.py`` and the ``nnp-infer`` script."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nnp.config import MODEL_PATH
from nnp.inference.predictor import infer_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Predict an activity (0-5) for every <i>.csv in DATA_DIR, in numeric order; "
        "writes results.txt and time.txt to the working directory.",
    )
    parser.add_argument("data_dir", type=Path, help="folder containing 1.csv … N.csv")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="model bundle (.pkl)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path.cwd(), help="where to write results.txt/time.txt"
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    infer_folder(args.data_dir, args.model, args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

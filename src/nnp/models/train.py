"""Train the final pipeline on all training devices and write ``models/model.pkl``.

``uv run poe train`` (or ``nnp-train --train-dir … --out …``). Uses only the training
device folders: validation labels never enter training (ADR-003).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from nnp.config import MODEL_PATH, TRAIN_DIR
from nnp.data import load_training
from nnp.models.bundle import save_bundle
from nnp.models.final import FINAL_CONFIG
from nnp.models.pipeline import HARPipeline, PipelineConfig

log = logging.getLogger(__name__)


def train(train_dir: Path, config: PipelineConfig = FINAL_CONFIG) -> HARPipeline:
    ts = load_training(train_dir)
    log.info("training on %d records from devices %s", ts.X.shape[0], ts.device_names)
    return HARPipeline(config).fit(ts.X, ts.y, ts.devices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    pipeline = train(args.train_dir)
    seconds = time.perf_counter() - t0
    save_bundle(pipeline, args.out, {"train_seconds": round(seconds, 2)})
    size_mb = args.out.stat().st_size / 1e6
    log.info("saved %s (%.2f MB) in %.1fs", args.out, size_mb, seconds)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Folder → ``results.txt`` + ``time.txt`` (brief §9, ADR-004 timing rule).

Order of work:

1. **I/O (not timed, as the brief requires):** list ``<int>.csv`` files in numeric order and
   read every sample. A file that cannot be parsed becomes an all-NaN row (logged), so the
   output still has exactly one line per input file; NaNs are imputed as the domain location.
2. **One-off adaptation (timed):** cluster the test set into devices and estimate their
   statistics, plus one warm-up prediction. Its wall time ``T_fit`` is *amortized*: every
   sample is charged ``T_fit / N``.
3. **Per-sample loop (timed):** timer starts right after the sample is in memory and stops
   right after its label is produced. ``time_i = t_i + T_fit / N``, rounded to 3 decimals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nnp.config import N_FEATURES
from nnp.data.io import DataFormatError, list_sample_files, read_sample
from nnp.models.bundle import load_bundle
from nnp.models.pipeline import HARPipeline

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class InferenceRun:
    predictions: list[int]
    seconds: list[float]
    """Per-sample time charged in ``time.txt`` (before rounding)."""
    adapt_seconds: float
    n_devices: int
    files: list[Path]


def resolve_data_dir(path: Path) -> Path:
    """Accept the ``data`` folder itself or a parent that contains ``data/``."""
    if path.is_dir() and not any(path.glob("*.csv")) and (path / "data").is_dir():
        return path / "data"
    return path


def read_all(files: list[Path]) -> FloatArray:
    rows: list[FloatArray] = []
    for f in files:
        try:
            rows.append(read_sample(f))
        except (DataFormatError, ValueError, OSError) as exc:
            log.error("could not read %s (%s); predicting from an all-missing row", f.name, exc)
            rows.append(np.full(N_FEATURES, np.nan))
    return np.vstack(rows) if rows else np.empty((0, N_FEATURES))


def run(pipeline: HARPipeline, X: FloatArray, files: list[Path]) -> InferenceRun:
    n = X.shape[0]
    if n == 0:
        return InferenceRun([], [], 0.0, 0, files)
    t0 = time.perf_counter()
    adaptation = pipeline.adapt(X)
    pipeline.predict_one(X[0], adaptation)  # warm-up, charged to the amortized cost
    adapt_seconds = time.perf_counter() - t0
    share = adapt_seconds / n

    preds: list[int] = []
    secs: list[float] = []
    for i in range(n):
        x = X[i]
        start = time.perf_counter()
        label = pipeline.predict_one(x, adaptation)
        elapsed = time.perf_counter() - start
        preds.append(label)
        secs.append(elapsed + share)
    return InferenceRun(preds, secs, adapt_seconds, adaptation.n_devices, files)


def write_outputs(result: InferenceRun, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.txt"
    time_path = out_dir / "time.txt"
    results_path.write_text("".join(f"{p}\n" for p in result.predictions))
    time_path.write_text("".join(f"{s:.3f}\n" for s in result.seconds))
    return results_path, time_path


def infer_folder(data_dir: Path, model_path: Path, out_dir: Path) -> InferenceRun:
    pipeline, meta = load_bundle(model_path)
    log.info("model: %s (%s features)", meta["config"]["model"], meta["n_selected_features"])
    files = list_sample_files(resolve_data_dir(data_dir))
    X = read_all(files)
    result = run(pipeline, X, files)
    write_outputs(result, out_dir)
    log.info(
        "%d samples, %d discovered device(s), adapt %.3fs, mean %.3f ms/sample",
        len(files),
        result.n_devices,
        result.adapt_seconds,
        1000 * float(np.mean(result.seconds)) if result.seconds else 0.0,
    )
    return result

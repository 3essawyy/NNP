"""Cross-validation and validation protocols (ADR-005).

Three complementary estimates, all reported side by side in the report:

1. **In-domain CV** — stratified k-fold on the training device(s). Upper bound; says nothing
   about device shift.
2. **Simulated-shift CV** — same folds, but every held-out fold is passed through a random
   synthetic device transform (:mod:`nnp.evaluation.shift`) before prediction. Measures
   robustness of the preprocessing to calibration/gain/noise shifts without any extra labels.
3. **Validation** — the released ``eval_data`` (labels used for model selection and tuning
   only, never for training — ADR-003). Reported overall and per *discovered* device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold

from nnp.evaluation.shift import ShiftSpec, apply_shift
from nnp.models.pipeline import HARPipeline, PipelineConfig

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CVResult:
    mean: float
    std: float
    folds: list[float]


def _folds(y: IntArray, k: int, seed: int) -> list[tuple[IntArray, IntArray]]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in skf.split(np.zeros(len(y)), y)]


def in_domain_cv(
    config: PipelineConfig,
    X: FloatArray,
    y: IntArray,
    devices: npt.NDArray[np.str_],
    k: int = 5,
    seed: int = 0,
) -> CVResult:
    accs: list[float] = []
    for tr, te in _folds(y, k, seed):
        p = HARPipeline(config).fit(X[tr], y[tr], devices[tr])
        accs.append(float((p.predict(X[te]) == y[te]).mean()))
    return CVResult(float(np.mean(accs)), float(np.std(accs)), accs)


def shifted_cv(
    config: PipelineConfig,
    X: FloatArray,
    y: IntArray,
    devices: npt.NDArray[np.str_],
    spec: ShiftSpec,
    k: int = 5,
    seed: int = 0,
) -> CVResult:
    """Held-out fold = a simulated new device (random offset/gain/noise per fold)."""
    rng = np.random.default_rng(seed)
    accs: list[float] = []
    for tr, te in _folds(y, k, seed):
        p = HARPipeline(config).fit(X[tr], y[tr], devices[tr])
        Xs = apply_shift(X[te], spec, rng)
        accs.append(float((p.predict(Xs) == y[te]).mean()))
    return CVResult(float(np.mean(accs)), float(np.std(accs)), accs)


@dataclass(frozen=True)
class ValidationResult:
    accuracy: float
    per_device: list[float]
    device_sizes: list[int]
    n_devices: int
    silhouette: float | None
    fit_seconds: float
    adapt_seconds: float
    y_pred: IntArray
    device_labels: IntArray


def validate(
    config: PipelineConfig,
    X: FloatArray,
    y: IntArray,
    devices: npt.NDArray[np.str_],
    X_val: FloatArray,
    y_val: IntArray,
) -> ValidationResult:
    t0 = time.perf_counter()
    p = HARPipeline(config).fit(X, y, devices)
    t1 = time.perf_counter()
    ad = p.adapt(X_val)
    t2 = time.perf_counter()
    pred = p.predict(X_val, ad)
    labels = ad.assignment.labels
    per = [float((pred[labels == c] == y_val[labels == c]).mean()) for c in range(ad.n_devices)]
    sizes = [int((labels == c).sum()) for c in range(ad.n_devices)]
    return ValidationResult(
        accuracy=float((pred == y_val).mean()),
        per_device=per,
        device_sizes=sizes,
        n_devices=ad.n_devices,
        silhouette=ad.assignment.silhouette,
        fit_seconds=t1 - t0,
        adapt_seconds=t2 - t1,
        y_pred=pred,
        device_labels=labels,
    )

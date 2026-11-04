"""Synthetic device shifts for robustness testing without extra labelled devices (ADR-005).

A simulated "new phone" applies, per feature ``j``:

* a **calibration offset** ``b_j`` (on a random subset of features, magnitude in units of the
  feature's spread),
* a **sensitivity gain** ``a_j = exp(N(0, gain_sigma))``,
* **non-stationary noise**: Gaussian noise whose std varies per *record*
  (``noise * spread_j * u_i``, ``u_i ~ U(0.5, 1.5)``),
* optional **outlier records** (a fraction of rows get a large spike in a few features).

These mimic the brief's §2.1 challenges and are used in simulated-shift cross-validation and
in the robustness curves of the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ShiftSpec:
    offset: float = 3.0
    """Offset magnitude in units of each feature's standard deviation."""
    offset_fraction: float = 0.5
    """Fraction of features that receive an offset."""
    gain_sigma: float = 0.2
    noise: float = 0.0
    outlier_fraction: float = 0.0
    outlier_magnitude: float = 20.0


def apply_shift(X: FloatArray, spec: ShiftSpec, rng: np.random.Generator) -> FloatArray:
    """Return a shifted copy of ``X`` (raw feature space) simulating one unseen device."""
    n, d = X.shape
    spread = X.std(axis=0)
    spread = np.where(spread > 1e-9, spread, 1.0)
    gain = np.exp(rng.normal(0.0, spec.gain_sigma, size=d))
    mask = rng.random(d) < spec.offset_fraction
    offset = np.where(mask, rng.normal(0.0, spec.offset, size=d) * spread, 0.0)
    center = X.mean(axis=0)
    Xs = (X - center) * gain + center + offset
    if spec.noise > 0:
        per_row = rng.uniform(0.5, 1.5, size=(n, 1))
        Xs = Xs + rng.normal(size=(n, d)) * spec.noise * spread * per_row
    if spec.outlier_fraction > 0:
        rows: npt.NDArray[np.bool_] = np.asarray(rng.random(n) < spec.outlier_fraction)
        cols: npt.NDArray[np.bool_] = np.asarray(rng.random((n, d)) < 0.02)
        spike = rng.choice([-1.0, 1.0], size=(n, d)) * spec.outlier_magnitude * spread
        Xs = Xs + np.where(rows[:, None] & cols, spike, 0.0)
    out: FloatArray = Xs.astype(np.float64)
    return out

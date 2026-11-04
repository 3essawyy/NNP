"""Per-domain (per-device) feature standardization — the core calibration fix (ADR-004).

A *domain* is a set of records captured by one device. Hardware calibration offsets and
sensitivity (gain) differences act, to first order, as a per-feature affine map
``x' = a * x + b``. Standardizing every domain with **its own** location/scale statistics
cancels ``a`` and ``b`` exactly (up to estimation noise), so a classifier trained on
standardized source-device features sees the same distribution on a new device.

Two estimators are offered:

* ``"zscore"``: mean / standard deviation (efficient, sensitive to outliers);
* ``"robust"``: median / IQR·0.7413 (IQR of a standard normal is 1.349, so the scale is
  comparable to a standard deviation), resistant to outlier records.

Missing values (NaN/±inf) are ignored when estimating statistics and are imputed with the
domain location, i.e. they become 0 after standardization (a neutral value).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Scaling = Literal["zscore", "robust"]

_IQR_TO_SIGMA = 1.0 / 1.3489795
_MIN_SCALE = 1e-9


def _finite(X: FloatArray) -> FloatArray:
    out: FloatArray = np.where(np.isfinite(X), X, np.nan)
    return out


@dataclass(frozen=True)
class DomainStats:
    """Location and scale of every feature within one domain."""

    location: FloatArray
    scale: FloatArray
    n_samples: int

    @classmethod
    def fit(cls, X: FloatArray, scaling: Scaling = "zscore") -> DomainStats:
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("DomainStats.fit needs a non-empty 2-D array")
        Xf = _finite(X)
        if scaling == "zscore":
            loc = np.nanmean(Xf, axis=0)
            scale = np.nanstd(Xf, axis=0)
        elif scaling == "robust":
            loc = np.nanmedian(Xf, axis=0)
            q75, q25 = np.nanpercentile(Xf, [75, 25], axis=0)
            scale = (q75 - q25) * _IQR_TO_SIGMA
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown scaling {scaling!r}")
        loc = np.nan_to_num(loc, nan=0.0)
        scale = np.nan_to_num(scale, nan=1.0)
        scale = np.where(scale < _MIN_SCALE, 1.0, scale)
        return cls(loc.astype(np.float64), scale.astype(np.float64), int(X.shape[0]))

    def subset(self, columns: npt.NDArray[np.intp]) -> DomainStats:
        """Statistics restricted to ``columns`` (used to make inference cheaper)."""
        return DomainStats(self.location[columns], self.scale[columns], self.n_samples)

    def transform(self, X: FloatArray, clip: float | None = None) -> FloatArray:
        """Standardize ``X`` (1-D or 2-D); non-finite → 0; optionally clip to ±clip."""
        Z = (X - self.location) / self.scale
        Z = np.where(np.isfinite(Z), Z, 0.0)
        if clip is not None:
            Z = np.clip(Z, -clip, clip)
        out: FloatArray = Z.astype(np.float64, copy=False)
        return out


def standardize_by_domain(
    X: FloatArray,
    domains: npt.NDArray[np.generic],
    scaling: Scaling = "zscore",
    clip: float | None = None,
) -> tuple[FloatArray, dict[object, DomainStats]]:
    """Standardize each domain of ``X`` with its own statistics.

    Returns the standardized matrix and the fitted statistics keyed by domain label.
    """
    Z = np.empty_like(X, dtype=np.float64)
    stats: dict[object, DomainStats] = {}
    for d in np.unique(domains):
        mask = domains == d
        s = DomainStats.fit(X[mask], scaling)
        stats[d.item() if hasattr(d, "item") else d] = s
        Z[mask] = s.transform(X[mask], clip)
    return Z, stats

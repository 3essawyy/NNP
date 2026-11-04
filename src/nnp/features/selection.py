"""Feature selection (ADR-006).

The released training data contains a block of features (mostly indices 40–239) whose
class signal is buried in heavy noise: their ANOVA F-score on the training set is ~5,
versus ~300–16 000 for the other features. Distance- and kernel-based classifiers do not
transfer to a new device when those features are kept (they dominate Euclidean distances
with non-transferable noise), while the remaining features carry almost all of the
transferable signal.

The selector is fit on the **standardized training data only** (no test labels):

* ``method="anova"`` keeps features with F-score > ``threshold`` (or the top ``k``);
* ``method="mutual_info"`` keeps the top ``k`` by mutual information;
* ``method="none"`` keeps everything (ablation baseline).

Near-constant features (variance < ``min_variance``) are always removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.feature_selection import f_classif, mutual_info_classif

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Method = Literal["anova", "mutual_info", "none"]


@dataclass(frozen=True)
class SelectionConfig:
    method: Method = "anova"
    threshold: float | None = 50.0
    k: int | None = None
    min_variance: float = 1e-8
    seed: int = 0


class FeatureSelector:
    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()
        self.scores_: FloatArray | None = None
        self.support_: npt.NDArray[np.intp] | None = None

    def fit(self, X: FloatArray, y: IntArray) -> FeatureSelector:
        cfg = self.config
        variance_ok = X.var(axis=0) >= cfg.min_variance
        if cfg.method == "none":
            scores = np.ones(X.shape[1])
        elif cfg.method == "anova":
            F, _ = f_classif(X, y)
            scores = np.nan_to_num(F, nan=0.0)
        elif cfg.method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=cfg.seed)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown method {cfg.method!r}")
        scores = np.where(variance_ok, scores, -np.inf)
        self.scores_ = scores.astype(np.float64)

        if cfg.method == "none":
            keep = np.flatnonzero(variance_ok)
        elif cfg.k is not None:
            order = np.argsort(-scores, kind="stable")
            keep = np.sort(order[: min(cfg.k, int(variance_ok.sum()))])
        elif cfg.threshold is not None:
            keep = np.flatnonzero(scores > cfg.threshold)
        else:
            keep = np.flatnonzero(variance_ok)
        if keep.size == 0:
            raise ValueError("feature selection removed every feature; lower the threshold")
        self.support_ = keep.astype(np.intp)
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if self.support_ is None:
            raise RuntimeError("FeatureSelector.fit must be called first")
        out: FloatArray = X[..., self.support_]
        return out

"""Optional dimensionality reduction after selection (compared in experiments, ADR-006).

* ``"none"`` — identity (default for the shipped model).
* ``"pca"``  — unsupervised projection to ``n_components`` (variance-preserving).
* ``"lda"``  — supervised Fisher projection to at most ``n_classes - 1`` = 5 dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ReductionMethod = Literal["none", "pca", "lda"]


@dataclass(frozen=True)
class ReductionConfig:
    method: ReductionMethod = "none"
    n_components: int | float | None = None
    seed: int = 0


class Reducer:
    def __init__(self, config: ReductionConfig | None = None) -> None:
        self.config = config or ReductionConfig()
        self.model_: Any = None

    def fit(self, X: FloatArray, y: IntArray) -> Reducer:
        cfg = self.config
        if cfg.method == "pca":
            self.model_ = PCA(n_components=cfg.n_components, random_state=cfg.seed).fit(X)
        elif cfg.method == "lda":
            n = cfg.n_components if isinstance(cfg.n_components, int) else None
            self.model_ = LinearDiscriminantAnalysis(n_components=n, solver="svd").fit(X, y)
        else:
            self.model_ = None
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if self.model_ is None:
            return X
        out: FloatArray = np.asarray(self.model_.transform(X), dtype=np.float64)
        return out

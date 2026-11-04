"""Unsupervised discovery of devices in an unlabeled test set (ADR-004).

Why not cluster on all features? Activities themselves form strong clusters (static vs
dynamic postures), so naive clustering would split the test set by *activity* and then
per-cluster standardization would erase the very signal we want to classify.

Instead we cluster on **probe features**: the features whose training distribution is
almost independent of the activity (lowest ANOVA F-scores on the training set). Device
calibration offsets move those features, activities do not. Probe values are robust-z-scored
with the training statistics, projected to a few principal components (fit on the test set),
and clustered with k-means for ``k = 1 … k_max``. A split is accepted only if it is
**clear** (silhouette ≥ ``min_silhouette``) and every cluster is **large**
(≥ ``min_fraction`` of the samples and ≥ ``min_size``) — small groups of outlier records are
not devices. The best accepted ``k`` (highest silhouette) wins; otherwise ``k = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import silhouette_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ClusterConfig:
    probe_quantile: float = 0.25
    """Use the features whose training F-score is in the lowest ``probe_quantile``."""
    n_components: int = 2
    k_max: int = 4
    min_silhouette: float = 0.8
    min_fraction: float = 0.05
    min_size: int = 30
    silhouette_sample: int = 2000
    seed: int = 0


@dataclass
class DeviceAssignment:
    """Result of clustering one test set: cluster labels and the model to assign new rows."""

    labels: IntArray
    k: int
    silhouette: float | None
    _probe: npt.NDArray[np.intp] = field(repr=False)
    _loc: FloatArray = field(repr=False)
    _scale: FloatArray = field(repr=False)
    _pca: PCA | None = field(default=None, repr=False)
    _centroids: FloatArray | None = field(default=None, repr=False)

    def assign(self, x: FloatArray) -> int:
        """Cluster of a single raw sample (nearest centroid in probe-PCA space)."""
        if self.k == 1 or self._pca is None or self._centroids is None:
            return 0
        z = _probe_z(x[None, :], self._probe, self._loc, self._scale)
        p = self._pca.transform(z)
        return int(np.argmin(((self._centroids - p) ** 2).sum(axis=1)))


def _probe_z(
    X: FloatArray, probe: npt.NDArray[np.intp], loc: FloatArray, scale: FloatArray
) -> FloatArray:
    Z = (X[:, probe] - loc) / scale
    out: FloatArray = np.where(np.isfinite(Z), Z, 0.0)
    return out


class DeviceClusterer:
    """Fit on labelled training data once; then :meth:`partition` any unlabeled set."""

    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = config or ClusterConfig()
        self.probe_: npt.NDArray[np.intp] | None = None
        self.loc_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, X: FloatArray, y: IntArray) -> DeviceClusterer:
        Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        F, _ = f_classif(Xf, y)
        F = np.nan_to_num(F, nan=0.0)
        cut = np.quantile(F, self.config.probe_quantile)
        self.probe_ = np.flatnonzero(cut >= F).astype(np.intp)
        P = Xf[:, self.probe_]
        q75, q25 = np.percentile(P, [75, 25], axis=0)
        self.loc_ = np.median(P, axis=0)
        self.scale_ = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        return self

    def partition(self, X: FloatArray) -> DeviceAssignment:
        if self.probe_ is None or self.loc_ is None or self.scale_ is None:
            raise RuntimeError("DeviceClusterer.fit must be called first")
        cfg = self.config
        n = X.shape[0]
        single = DeviceAssignment(
            np.zeros(n, dtype=np.int64), 1, None, self.probe_, self.loc_, self.scale_
        )
        min_size = max(cfg.min_size, int(np.ceil(cfg.min_fraction * n)))
        if n < 2 * min_size:
            return single
        Z = _probe_z(X, self.probe_, self.loc_, self.scale_)
        pca = PCA(n_components=min(cfg.n_components, Z.shape[1]), random_state=cfg.seed)
        Zp = pca.fit_transform(Z)
        best: DeviceAssignment = single
        best_sil = -1.0
        for k in range(2, cfg.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Zp)
            labels = km.labels_.astype(np.int64)
            if np.bincount(labels, minlength=k).min() < min_size:
                continue
            sil = float(
                silhouette_score(
                    Zp, labels, sample_size=min(cfg.silhouette_sample, n), random_state=cfg.seed
                )
            )
            if sil >= cfg.min_silhouette and sil > best_sil:
                best_sil = sil
                best = DeviceAssignment(
                    labels,
                    k,
                    sil,
                    self.probe_,
                    self.loc_,
                    self.scale_,
                    pca,
                    km.cluster_centers_.astype(np.float64),
                )
        return best

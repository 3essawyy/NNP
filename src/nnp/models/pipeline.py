"""The end-to-end HAR pipeline: preprocessing → selection → reduction → classifier.

Training (labelled source devices)::

    raw X ──per-device standardize──▶ Z ──clip──▶ select(F-score) ──▶ reduce ──▶ classifier

Inference (unlabeled target set, ADR-004)::

    raw X_test ──DeviceClusterer.partition──▶ k discovered devices
               ──per-discovered-device standardize (own stats)──▶ same chain as training

``adapt`` is the one-off transductive step (cluster + statistics over the whole test set);
``predict_one`` is the cheap per-sample step that ``infer.py`` times individually.
With ``adapt=False`` the pipeline is a classic model: one global standardizer fit on the
training data and applied unchanged to every test record (ablation baseline).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from nnp.features import FeatureSelector, Reducer, ReductionConfig, SelectionConfig
from nnp.models.fast import GenericPredictor, Predictor, compile_predictor
from nnp.models.zoo import make_model
from nnp.preprocessing import (
    ClusterConfig,
    DeviceAssignment,
    DeviceClusterer,
    DomainStats,
    Scaling,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PipelineConfig:
    scaling: Scaling = "zscore"
    clip: float | None = 8.0
    adapt: bool = True
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    model: str = "svc_rbf"
    model_params: Mapping[str, Any] = field(default_factory=dict)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    min_stats_size: int = 30
    """A discovered device needs at least this many records for its own statistics;
    smaller groups use the training reference statistics (``Adaptation.fallback``)."""
    fallback_model: str | None = None
    """Classifier used for records whose device fell back to reference statistics
    (no adaptation possible, e.g. tiny test sets). ``None`` = use the main model (ADR-013)."""
    fallback_params: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "scaling": self.scaling,
            "clip": self.clip,
            "adapt": self.adapt,
            "selection": vars(self.selection),
            "reduction": vars(self.reduction),
            "model": self.model,
            "model_params": dict(self.model_params),
            "cluster": vars(self.cluster),
            "min_stats_size": self.min_stats_size,
            "fallback_model": self.fallback_model,
            "fallback_params": dict(self.fallback_params),
            "seed": self.seed,
        }


@dataclass
class Adaptation:
    """Everything ``predict_one`` needs about one test set (output of ``adapt``)."""

    assignment: DeviceAssignment
    stats: list[DomainStats]
    """One entry per discovered device, restricted to the selected columns."""
    fallback: list[bool]
    """True where a cluster was too small and the training reference stats were used."""

    @property
    def n_devices(self) -> int:
        return len(self.stats)


class HARPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.selector = FeatureSelector(self.config.selection)
        self.reducer = Reducer(self.config.reduction)
        self.clusterer = DeviceClusterer(self.config.cluster)
        self.model: Any = None
        self.reference_: DomainStats | None = None
        self.train_device_stats_: dict[str, DomainStats] = {}
        self.n_train_: int = 0
        self._fast: Predictor | None = None
        self.fallback_: Any = None
        self._fast_fallback: Predictor | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fast"] = None  # rebuilt lazily after unpickling; keeps model.pkl small
        state["_fast_fallback"] = None
        return state

    # ------------------------------------------------------------------ training
    def _standardize_train(self, X: FloatArray, devices: npt.NDArray[np.str_]) -> FloatArray:
        cfg = self.config
        Z = np.empty_like(X, dtype=np.float64)
        self.reference_ = DomainStats.fit(X, cfg.scaling)
        if not cfg.adapt:
            return self.reference_.transform(X, cfg.clip)
        self.train_device_stats_ = {}
        for d in np.unique(devices):
            mask = devices == d
            s = DomainStats.fit(X[mask], cfg.scaling)
            self.train_device_stats_[str(d)] = s
            Z[mask] = s.transform(X[mask], cfg.clip)
        return Z

    def fit(
        self, X: FloatArray, y: IntArray, devices: npt.NDArray[np.str_] | None = None
    ) -> HARPipeline:
        devices = devices if devices is not None else np.full(X.shape[0], "train")
        Z = self._standardize_train(X, devices)
        self.selector.fit(Z, y)
        Zs = self.selector.transform(Z)
        self.reducer.fit(Zs, y)
        self.model = make_model(self.config.model, self.config.model_params, self.config.seed)
        Zr = self.reducer.transform(Zs)
        self.model.fit(Zr, y)
        self._fast = self._compile(Zr[: min(500, Zr.shape[0])])
        if self.config.adapt:
            self.clusterer.fit(X, y)
            if self.config.fallback_model is not None and self.reference_ is not None:
                # trained on the *reference* standardization: exactly what it sees at test time
                Zref = self.reducer.transform(
                    self.selector.transform(self.reference_.transform(X, self.config.clip))
                )
                self.fallback_ = make_model(
                    self.config.fallback_model, self.config.fallback_params, self.config.seed
                ).fit(Zref, y)
                self._fast_fallback = self._compile_for(self.fallback_, Zref[:500])
        self.n_train_ = int(X.shape[0])
        return self

    # ------------------------------------------------------------------ inference
    @property
    def support(self) -> npt.NDArray[np.intp]:
        if self.selector.support_ is None:
            raise RuntimeError("pipeline is not fitted")
        return self.selector.support_

    def adapt(self, X: FloatArray) -> Adaptation:
        """One-off, label-free step over the whole test set (ADR-004)."""
        if self.reference_ is None:
            raise RuntimeError("pipeline is not fitted")
        sup = self.support
        ref = self.reference_.subset(sup)
        if not self.config.adapt or X.shape[0] == 0:
            assignment = DeviceAssignment(
                np.zeros(X.shape[0], dtype=np.int64),
                1,
                None,
                np.empty(0, dtype=np.intp),
                np.empty(0),
                np.empty(0),
            )
            return Adaptation(assignment, [ref], [not self.config.adapt])
        assignment = self.clusterer.partition(X)
        stats: list[DomainStats] = []
        fallback: list[bool] = []
        for c in range(assignment.k):
            members = X[assignment.labels == c][:, sup]
            if members.shape[0] < self.config.min_stats_size:
                stats.append(ref)
                fallback.append(True)
            else:
                stats.append(DomainStats.fit(members, self.config.scaling))
                fallback.append(False)
        return Adaptation(assignment, stats, fallback)

    def _compile(self, check: FloatArray | None = None) -> Predictor:
        """Fast single-row predictor for the main model."""
        return self._compile_for(self.model, check)

    @staticmethod
    def _compile_for(model: Any, check: FloatArray | None = None) -> Predictor:
        """Fast single-row predictor; verified against ``model.predict`` when ``check`` given."""
        fast = compile_predictor(model)
        # Only a re-implementation needs verifying; the generic fallback *is* model.predict.
        if check is not None and check.shape[0] > 0 and not isinstance(fast, GenericPredictor):
            reference = model.predict(check)
            if any(fast(row) != ref for row, ref in zip(check, reference, strict=True)):
                fast = GenericPredictor(model)  # pragma: no cover - safety net
        return fast

    def _uses_fallback(self, adaptation: Adaptation, c: int) -> bool:
        return self.fallback_ is not None and adaptation.fallback[c] and self.config.adapt

    def _features(self, Xsel: FloatArray, stats: DomainStats) -> FloatArray:
        return self.reducer.transform(stats.transform(Xsel, self.config.clip))

    def predict_one(self, x: FloatArray, adaptation: Adaptation) -> int:
        """Predict one raw ``(561,)`` sample; this is the per-sample step ``infer.py`` times."""
        c = adaptation.assignment.assign(x) if adaptation.n_devices > 1 else 0
        z = self._features(x[self.support][None, :], adaptation.stats[c])
        if self._uses_fallback(adaptation, c):
            if self._fast_fallback is None:
                self._fast_fallback = self._compile_for(self.fallback_)
            return self._fast_fallback(z[0])
        if self._fast is None:
            self._fast = self._compile()
        return self._fast(z[0])

    def predict(self, X: FloatArray, adaptation: Adaptation | None = None) -> IntArray:
        """Batch prediction (experiments); same result as ``predict_one`` row by row."""
        adaptation = adaptation or self.adapt(X)
        labels = adaptation.assignment.labels
        out = np.empty(X.shape[0], dtype=np.int64)
        Xsel = X[:, self.support]
        for c in range(adaptation.n_devices):
            mask = labels == c
            if mask.any():
                model = self.fallback_ if self._uses_fallback(adaptation, c) else self.model
                out[mask] = model.predict(self._features(Xsel[mask], adaptation.stats[c]))
        return out

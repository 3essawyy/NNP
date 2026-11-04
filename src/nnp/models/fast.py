"""Low-overhead single-sample prediction for the shipped model (runtime is graded).

``sklearn.svm.SVC.predict`` on one row spends most of its ~1 ms in input validation and
libsvm call overhead. :class:`FastRBFSVC` re-implements the exact same decision rule in
numpy from the fitted attributes:

* RBF kernel to every support vector: ``k_s = exp(-γ‖sv_s − z‖²)``,
* one-vs-one decision for each class pair ``(i, j)`` with libsvm's coefficient layout
  (``dual_coef_[j-1]`` for class *i*'s SVs, ``dual_coef_[i]`` for class *j*'s SVs),
* majority vote; ties go to the lowest class index (same as libsvm).

The equivalence with ``SVC.predict`` is checked by ``tests/test_fast.py`` and by
``HARPipeline.fit`` on the training data. Any other estimator falls back to its own
``predict``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC

FloatArray = npt.NDArray[np.float64]
Predictor = Callable[[FloatArray], int]


class FastRBFSVC:
    def __init__(self, svc: Any) -> None:
        if not isinstance(svc, SVC) or svc.kernel != "rbf":
            raise TypeError("FastRBFSVC needs a fitted SVC with an RBF kernel")
        self.sv: FloatArray = np.ascontiguousarray(svc.support_vectors_, dtype=np.float64)
        self.sv_sq: FloatArray = (self.sv**2).sum(axis=1)
        self.gamma = float(svc._gamma)
        self.classes = np.asarray(svc.classes_)
        n_class = self.classes.size
        starts = np.concatenate([[0], np.cumsum(svc.n_support_)]).astype(int)
        dual = np.asarray(svc.dual_coef_, dtype=np.float64)
        intercept = np.asarray(svc.intercept_, dtype=np.float64)
        # Pair p = (i, j), i < j, in libsvm order; build one weight vector per pair over all SVs
        pairs: list[tuple[int, int]] = []
        W = np.zeros((n_class * (n_class - 1) // 2, self.sv.shape[0]))
        p = 0
        for i in range(n_class):
            for j in range(i + 1, n_class):
                si, sj = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
                W[p, si] = dual[j - 1, si]
                W[p, sj] = dual[i, sj]
                pairs.append((i, j))
                p += 1
        self.W = W
        self.b = intercept
        self.pos = np.array([i for i, _ in pairs])
        self.neg = np.array([j for _, j in pairs])
        self.n_class = n_class

    def __call__(self, z: FloatArray) -> int:
        d2 = self.sv_sq - 2.0 * (self.sv @ z) + float(z @ z)
        k = np.exp(-self.gamma * d2)
        dec = self.W @ k + self.b
        winners = np.where(dec > 0, self.pos, self.neg)
        votes = np.bincount(winners, minlength=self.n_class)
        return int(self.classes[int(np.argmax(votes))])


class GenericPredictor:
    """Fallback: the estimator's own ``predict`` on a 1-row matrix."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, z: FloatArray) -> int:
        return int(self.model.predict(z[None, :])[0])


def compile_predictor(model: Any) -> Predictor:
    """Return a fast single-row predictor for ``model`` (exact for RBF SVC)."""
    if isinstance(model, SVC) and model.kernel == "rbf":
        return FastRBFSVC(model)
    return GenericPredictor(model)

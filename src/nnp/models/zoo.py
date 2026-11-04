"""Model zoo: every classifier family compared in the report (ADR-007).

``make_model(name, params, seed)`` returns an unfitted scikit-learn-compatible estimator.
LightGBM is imported lazily so the inference image never needs it unless the shipped model
is a LightGBM model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

MODEL_NAMES: tuple[str, ...] = (
    "gaussian_nb",
    "lda",
    "qda",
    "logreg",
    "linear_svc",
    "svc_rbf",
    "knn",
    "random_forest",
    "extra_trees",
    "lightgbm",
    "mlp",
    "ensemble",
)

DEFAULTS: dict[str, dict[str, Any]] = {
    "gaussian_nb": {},
    "lda": {"solver": "lsqr", "shrinkage": "auto"},
    "qda": {"reg_param": 0.5},
    "logreg": {"C": 1.0, "max_iter": 5000},
    "linear_svc": {"C": 0.05, "max_iter": 20000},
    "svc_rbf": {"C": 10.0, "gamma": "scale"},
    "knn": {"n_neighbors": 10, "weights": "distance"},
    "random_forest": {"n_estimators": 300, "n_jobs": 4},
    "extra_trees": {"n_estimators": 300, "n_jobs": 4},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1},
    "mlp": {"hidden_layer_sizes": (256,), "alpha": 1e-2, "max_iter": 400, "early_stopping": True},
    "ensemble": {},
}


def make_model(name: str, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
    """Build an unfitted estimator; ``params`` override :data:`DEFAULTS`."""
    if name not in DEFAULTS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    p: dict[str, Any] = {**DEFAULTS[name], **(params or {})}
    if name == "gaussian_nb":
        return GaussianNB(**p)
    if name == "lda":
        return LinearDiscriminantAnalysis(**p)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(**p)
    if name == "logreg":
        return LogisticRegression(random_state=seed, **p)
    if name == "linear_svc":
        return LinearSVC(random_state=seed, **p)
    if name == "svc_rbf":
        return SVC(random_state=seed, **p)
    if name == "knn":
        return KNeighborsClassifier(**p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, **p)
    if name == "lightgbm":
        import lightgbm

        return lightgbm.LGBMClassifier(random_state=seed, **p)
    if name == "mlp":
        return MLPClassifier(random_state=seed, **p)
    # ensemble: hard majority vote of three strong, diverse linear/kernel models
    members = p.pop("members", ("svc_rbf", "linear_svc", "logreg"))
    member_params: Mapping[str, Mapping[str, Any]] = p.pop("member_params", {})
    return VotingClassifier(
        estimators=[(m, make_model(m, member_params.get(m), seed)) for m in members],
        voting="hard",
        **p,
    )

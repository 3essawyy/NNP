"""Classification metrics and error analysis used by experiments, the report and ``score``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from nnp.config import CLASS_NAMES, N_CLASSES

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ClassReport:
    label: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_f1: float
    n: int
    confusion: list[list[int]]
    per_class: list[ClassReport]
    top_confusions: list[tuple[str, str, int]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate(y_true: IntArray, y_pred: IntArray, top: int = 5) -> Evaluation:
    """Accuracy, macro-F1, confusion matrix (rows = true, cols = predicted), per-class P/R/F1."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    labels = np.arange(N_CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = [
        ClassReport(int(i), CLASS_NAMES[i], float(p[i]), float(r[i]), float(f[i]), int(s[i]))
        for i in labels
    ]
    off = [
        (CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j]))
        for i in labels
        for j in labels
        if i != j and cm[i, j] > 0
    ]
    off.sort(key=lambda t: -t[2])
    return Evaluation(
        accuracy=float((y_true == y_pred).mean()) if y_true.size else 0.0,
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        n=int(y_true.size),
        confusion=cm.astype(int).tolist(),
        per_class=per_class,
        top_confusions=off[:top],
    )

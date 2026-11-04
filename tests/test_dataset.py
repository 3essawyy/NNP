"""Checks against the real dataset — skipped when ``data/`` is absent (e.g. in CI)."""

from __future__ import annotations

import numpy as np
import pytest

from nnp.config import MODEL_PATH, VALIDATION_DIR, VALIDATION_LABELS
from nnp.data import load_ground_truth, load_samples
from nnp.models import load_bundle

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(not VALIDATION_DIR.is_dir(), reason="dataset not downloaded"),
]


def test_shipped_model_accuracy_on_validation() -> None:
    pipeline, _ = load_bundle(MODEL_PATH)
    _, X = load_samples(VALIDATION_DIR)
    y = load_ground_truth(VALIDATION_LABELS)
    pred = pipeline.predict(X)
    assert (pred == y).mean() >= 0.95
    assert pipeline.adapt(X).n_devices == 2
    assert np.bincount(pred, minlength=6).min() > 0

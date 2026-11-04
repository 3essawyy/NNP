"""Saving / loading the trained pipeline as ``models/model.pkl`` (brief §5.3, ADR-008).

The bundle is a plain pickle of a dict holding the fitted :class:`HARPipeline` and metadata
(library versions, training size, config). Only load bundles you produced yourself: pickle
executes code on load.
"""

from __future__ import annotations

import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

import nnp
from nnp.models.pipeline import HARPipeline

BUNDLE_FORMAT = 1


def save_bundle(pipeline: HARPipeline, path: Path, extra: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": BUNDLE_FORMAT,
        "pipeline": pipeline,
        "meta": {
            "nnp_version": nnp.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train": pipeline.n_train_,
            "n_selected_features": int(pipeline.support.size),
            "config": pipeline.config.describe(),
            **(extra or {}),
        },
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_bundle(path: Path) -> tuple[HARPipeline, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"model bundle not found: {path} (run `uv run poe train`)")
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path}: unsupported model bundle format")
    pipeline = payload["pipeline"]
    if not isinstance(pipeline, HARPipeline):
        raise TypeError(f"{path}: bundle does not contain a HARPipeline")
    meta: dict[str, Any] = payload["meta"]
    return pipeline, meta

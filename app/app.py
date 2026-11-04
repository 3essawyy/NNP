"""Gradio demo of the HAR model (bonus, ADR-010). Run locally: ``uv run poe app``.

Upload one or more sample CSVs (the ``1.csv … N.csv`` format of the brief). The same
pipeline as ``infer.py`` runs: with enough samples (≥ 60) the set is clustered into devices
and each device is standardized with its own statistics; with fewer samples the training
reference statistics are used (documented limitation — adaptation needs a batch).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
for candidate in (_ROOT / "src", _ROOT.parent / "src"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

from nnp.config import CLASS_NAMES  # noqa: E402
from nnp.data.io import read_sample  # noqa: E402
from nnp.models.bundle import load_bundle  # noqa: E402

MODEL_FILE = next(
    p for p in (_ROOT / "models" / "model.pkl", _ROOT.parent / "models" / "model.pkl") if p.exists()
)
PIPELINE, META = load_bundle(MODEL_FILE)


def _sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), "") if path.stem.isdigit() else (10**12, path.name)


def predict_files(files: list[Any] | None) -> tuple[list[list[Any]], str]:
    if not files:
        return [], "Upload at least one CSV."
    paths = sorted((Path(getattr(f, "name", f)) for f in files), key=_sort_key)
    rows: list[list[Any]] = []
    X = []
    ok: list[Path] = []
    for p in paths:
        try:
            X.append(read_sample(p))
            ok.append(p)
        except ValueError as exc:
            rows.append([p.name, "—", f"unreadable: {exc}", ""])
    if not X:
        return rows, "No readable sample."
    Xa = np.vstack(X)
    t0 = time.perf_counter()
    adaptation = PIPELINE.adapt(Xa)
    PIPELINE.predict_one(Xa[0], adaptation)  # warm-up (builds the fast path), charged below
    adapt_s = time.perf_counter() - t0
    for p, x in zip(ok, Xa, strict=True):
        t = time.perf_counter()
        label = PIPELINE.predict_one(x, adaptation)
        ms = (time.perf_counter() - t + adapt_s / len(Xa)) * 1000
        rows.append([p.name, label, CLASS_NAMES[label], f"{ms:.2f}"])
    note = f"{len(Xa)} sample(s); {adaptation.n_devices} device(s) discovered" + (
        " — using training reference statistics (upload ≥ 60 samples to adapt)."
        if any(adaptation.fallback)
        else "."
    )
    return rows, note


def build() -> Any:
    import gradio as gr

    with gr.Blocks(title="NNP — robust HAR") as demo:
        gr.Markdown(
            "# Robust Human Activity Recognition\n"
            "Smartphone inertial features (561) → one of six activities. "
            f"Model: `{META['config']['model']}` on {META['n_selected_features']} features, "
            "per-device test-time standardization. Source: the NNP repository."
        )
        files = gr.File(file_count="multiple", file_types=[".csv"], label="Sample CSVs")
        run = gr.Button("Predict", variant="primary")
        note = gr.Markdown()
        table = gr.Dataframe(
            headers=["file", "label", "activity", "ms (charged)"], interactive=False, wrap=True
        )
        run.click(predict_files, inputs=files, outputs=[table, note])
    return demo


if __name__ == "__main__":
    build().launch()

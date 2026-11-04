"""E6 — the shipped model on the validation set through the real inference path.

Uses :func:`nnp.inference.predictor.run`, i.e. the exact code ``infer.py`` executes (cluster,
adapt, per-sample timed loop), so the reported accuracy and timings are the ones a grader
would see on similar data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nnp.config import CLASS_NAMES, MODEL_PATH, N_CLASSES
from nnp.evaluation.metrics import evaluate
from nnp.experiments.common import (
    SEQUENTIAL,
    SERIES,
    Data,
    figure,
    latex_macros,
    latex_table,
    pct,
    save_figure,
    save_json,
)
from nnp.inference.predictor import run as infer_run
from nnp.models.bundle import load_bundle


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    pipeline, meta = load_bundle(MODEL_PATH)
    files = [Path(f"{i + 1}.csv") for i in range(data.X_val.shape[0])]
    res = infer_run(pipeline, data.X_val, files)
    pred = np.array(res.predictions, dtype=np.int64)
    ev = evaluate(data.y_val, pred)
    ad = pipeline.adapt(data.X_val)
    dev = ad.assignment.labels
    per_device = [
        evaluate(data.y_val[dev == c], pred[dev == c]).accuracy for c in range(ad.n_devices)
    ]
    secs = np.array(res.seconds)
    rounded = np.round(secs, 3)
    out = {
        "model_meta": meta,
        "accuracy": ev.accuracy,
        "macro_f1": ev.macro_f1,
        "per_device_accuracy": per_device,
        "per_class": [vars(c) for c in ev.per_class],
        "confusion": ev.confusion,
        "top_confusions": ev.top_confusions,
        "adapt_seconds": res.adapt_seconds,
        "time_mean_ms": float(secs.mean() * 1000),
        "time_p95_ms": float(np.percentile(secs, 95) * 1000),
        "time_max_ms": float(secs.max() * 1000),
        "time_total_s": float(secs.sum()),
        "time_rounded_total_s": float(rounded.sum()),
        "n_devices": res.n_devices,
    }
    save_json("final", out)
    print(
        f"  final: acc {ev.accuracy:.4f} macroF1 {ev.macro_f1:.4f} per-device {per_device} "
        f"mean {out['time_mean_ms']:.3f} ms"
    )

    cm = np.array(ev.confusion)
    fig, ax = figure(4.4, 3.6)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.grid(False)
    short = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]
    ax.set_xticks(range(N_CLASSES), short)
    ax.set_yticks(range(N_CLASSES), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    ax.set_title(f"Confusion matrix, validation (acc. {pct(ev.accuracy)}%)")
    save_figure(fig, "confusion")

    fig, ax = figure(4.4, 2.4)
    ax.hist(secs * 1000, bins=50, color=SERIES[0])
    ax.set_xlabel("charged time per sample (ms, incl. amortized adaptation)")
    ax.set_ylabel("samples")
    ax.set_title("Per-sample inference time")
    save_figure(fig, "timing")

    latex_table(
        "per_class",
        ["Activity", "Precision", "Recall", "F1", "Support"],
        [
            [
                f"{c.label} {c.name}",
                pct(c.precision),
                pct(c.recall),
                pct(c.f1),
                str(c.support),
            ]
            for c in ev.per_class
        ],
        "Shipped model on the validation set, through the real \\texttt{infer.py} path (\\%).",
        "tab:per_class",
    )
    latex_macros(
        {
            "FinalAcc": pct(ev.accuracy),
            "FinalMacroF": pct(ev.macro_f1),
            "FinalDevZero": pct(per_device[0]),
            "FinalDevOne": pct(per_device[1] if len(per_device) > 1 else per_device[0]),
            "FinalMeanMs": f"{out['time_mean_ms']:.2f}",
            "FinalPNinetyFiveMs": f"{out['time_p95_ms']:.2f}",
            "FinalAdaptS": f"{res.adapt_seconds:.2f}",
            "FinalTotalS": f"{out['time_total_s']:.2f}",
            "SitStand": str(int(cm[3, 4] + cm[4, 3])),
            "NSelected": str(int(pipeline.support.size)),
            "ModelMB": f"{MODEL_PATH.stat().st_size / 1e6:.1f}",
            "NSV": str(int(pipeline.model.support_vectors_.shape[0]))
            if hasattr(pipeline.model, "support_vectors_")
            else "--",
        }
    )
    names = dict(enumerate(CLASS_NAMES))
    out["class_names"] = names
    return out

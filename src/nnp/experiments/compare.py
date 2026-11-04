"""E2 — model family comparison under one fixed preprocessing (report §Model selection)."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from nnp.evaluation.shift import ShiftSpec
from nnp.experiments.common import (
    SERIES,
    Data,
    figure,
    latex_macros,
    latex_table,
    pct,
    save_figure,
    save_json,
)
from nnp.models.cv import in_domain_cv, shifted_cv, validate
from nnp.models.final import FINAL_CONFIG
from nnp.models.pipeline import HARPipeline
from nnp.models.zoo import MODEL_NAMES

SIM_SHIFT = ShiftSpec(
    offset=3.0, offset_fraction=0.5, gain_sigma=0.2, noise=0.1, outlier_fraction=0.01
)

LABELS = {
    "gaussian_nb": "Gaussian NB",
    "lda": "LDA (shrinkage)",
    "qda": "QDA (reg. 0.5)",
    "logreg": "Logistic regression",
    "linear_svc": "Linear SVM",
    "svc_rbf": "RBF SVM",
    "knn": "kNN (k=10)",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "lightgbm": "LightGBM",
    "mlp": "MLP (256)",
    "ensemble": "Vote (RBF+lin. SVM+LR)",
}


def per_sample_ms(pipe: HARPipeline, X: np.ndarray[Any, Any], n: int = 300) -> float:
    ad = pipe.adapt(X)
    rows = X[:n]
    pipe.predict_one(rows[0], ad)
    t = time.perf_counter()
    for r in rows:
        pipe.predict_one(r, ad)
    return 1000 * (time.perf_counter() - t) / len(rows)


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    k = 3 if quick else 5
    results: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        cfg = dataclasses.replace(FINAL_CONFIG, model=name, model_params={})
        t0 = time.perf_counter()
        cv = in_domain_cv(cfg, tr.X, tr.y, tr.devices, k=k)
        sh = shifted_cv(cfg, tr.X, tr.y, tr.devices, SIM_SHIFT, k=k)
        val = validate(cfg, tr.X, tr.y, tr.devices, data.X_val, data.y_val)
        noad = validate(
            dataclasses.replace(cfg, adapt=False), tr.X, tr.y, tr.devices, data.X_val, data.y_val
        )
        pipe = HARPipeline(cfg).fit(tr.X, tr.y, tr.devices)
        ms = per_sample_ms(pipe, data.X_val)
        row = {
            "model": name,
            "label": LABELS[name],
            "in_domain_cv": cv.mean,
            "in_domain_cv_std": cv.std,
            "shifted_cv": sh.mean,
            "shifted_cv_std": sh.std,
            "validation": val.accuracy,
            "validation_per_device": val.per_device,
            "validation_no_adapt": noad.accuracy,
            "fit_seconds": val.fit_seconds,
            "per_sample_ms": ms,
            "wall_seconds": time.perf_counter() - t0,
        }
        print(
            f"  {name:14s} in-dom {cv.mean:.3f}  shifted {sh.mean:.3f}  val {val.accuracy:.3f} "
            f"(no-adapt {noad.accuracy:.3f})  {ms:.2f} ms/sample"
        )
        results.append(row)

    results.sort(key=lambda r: -float(r["validation"]))
    save_json(
        "compare", {"folds": k, "sim_shift": dataclasses.asdict(SIM_SHIFT), "models": results}
    )

    rows = [
        [
            r["label"],
            f"{pct(r['in_domain_cv'])} $\\pm$ {pct(r['in_domain_cv_std'])}",
            f"{pct(r['shifted_cv'])}",
            pct(r["validation_no_adapt"]),
            f"\\textbf{{{pct(r['validation'])}}}" if i == 0 else pct(r["validation"]),
            f"{r['per_sample_ms']:.2f}",
        ]
        for i, r in enumerate(results)
    ]
    latex_table(
        "compare",
        [
            "Model",
            "In-domain CV",
            "Sim.-shift CV",
            "Val. (global scaler)",
            "Val. (adapted)",
            "ms/sample",
        ],
        rows,
        "Model families with the same preprocessing (ANOVA "
        f"$F>{FINAL_CONFIG.selection.threshold:g}$,"
        f" clip $\\pm{FINAL_CONFIG.clip:g}$). Accuracy in \\%; {k}-fold CV on Device\\_B; "
        "validation = released eval\\_data (never trained on).",
        "tab:compare",
        wide=True,
    )

    fig, ax = figure(6.4, 3.6)
    ys = np.arange(len(results))[::-1]
    for j, (key, lab) in enumerate(
        [
            ("validation_no_adapt", "val., global scaler"),
            ("shifted_cv", "simulated-shift CV"),
            ("validation", "val., per-device adapted"),
        ]
    ):
        ax.scatter([r[key] for r in results], ys, s=26, color=SERIES[j], label=lab, zorder=3)
    ax.set_yticks(ys, [r["label"] for r in results])
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Model families: robustness to device shift")
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig, "compare")

    best = results[0]
    latex_macros(
        {
            "BestCompareModel": best["label"],
            "BestCompareVal": pct(best["validation"]),
        }
    )
    return {"models": results}

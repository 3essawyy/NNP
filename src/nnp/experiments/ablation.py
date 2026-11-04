"""E3 — ablation of every preprocessing decision with the shipped classifier (report §Ablation)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

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
from nnp.experiments.compare import SIM_SHIFT
from nnp.features import ReductionConfig, SelectionConfig
from nnp.models.cv import shifted_cv, validate
from nnp.models.final import FINAL_CONFIG
from nnp.models.pipeline import HARPipeline, PipelineConfig


def variants(n_selected: int) -> list[tuple[str, PipelineConfig]]:
    """Ablation variants; MI keeps as many features as the shipped ANOVA filter."""
    f = FINAL_CONFIG
    r = dataclasses.replace
    return [
        ("Shipped pipeline", f),
        ("No per-device adaptation (global scaler)", r(f, adapt=False)),
        ("No feature selection (all 561)", r(f, selection=SelectionConfig(method="none"))),
        ("No adaptation and no selection", r(f, adapt=False, selection=SelectionConfig("none"))),
        ("No outlier clipping", r(f, clip=None)),
        ("Robust scaling (median/IQR)", r(f, scaling="robust")),
        (
            f"Mutual-information top-{n_selected}",
            r(f, selection=SelectionConfig("mutual_info", threshold=None, k=n_selected)),
        ),
        ("+ PCA (95\\% variance)", r(f, reduction=ReductionConfig("pca", 0.95))),
        ("+ LDA projection (5-D)", r(f, reduction=ReductionConfig("lda", 5))),
    ]


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    k = 3 if quick else 5
    rows: list[dict[str, Any]] = []
    n_sel = int(HARPipeline(FINAL_CONFIG).fit(tr.X, tr.y, tr.devices).support.size)
    for name, cfg in variants(n_sel):
        val = validate(cfg, tr.X, tr.y, tr.devices, data.X_val, data.y_val)
        sh = shifted_cv(cfg, tr.X, tr.y, tr.devices, SIM_SHIFT, k=k)
        rows.append(
            {
                "variant": name,
                "validation": val.accuracy,
                "validation_per_device": val.per_device,
                "n_devices": val.n_devices,
                "shifted_cv": sh.mean,
            }
        )
        print(f"  {name:45s} val {val.accuracy:.3f} {val.per_device}  shifted {sh.mean:.3f}")
    save_json("ablation", {"variants": rows})

    def dev(r: dict[str, Any], i: int) -> str:
        per = r["validation_per_device"]
        return pct(per[i]) if r["n_devices"] > 1 and i < len(per) else "--"

    latex_table(
        "ablation",
        ["Variant", "Val. dev.\\,0", "Val. dev.\\,1", "Val. overall", "Sim.-shift CV"],
        [
            [r["variant"], dev(r, 0), dev(r, 1), pct(r["validation"]), pct(r["shifted_cv"])]
            for r in rows
        ],
        "Ablation with the shipped RBF SVM (accuracy \\%). Per-device columns use the devices "
        "discovered on the validation set; variants without adaptation treat the set as one "
        "domain (--).",
        "tab:ablation",
        wide=True,
    )
    fig, ax = figure(6.2, 3.0)
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r["validation"] for r in rows], height=0.6, color=SERIES[0])
    for y_, r in zip(ys, rows, strict=True):
        ax.text(r["validation"] + 0.01, y_, pct(r["validation"]), va="center", fontsize=8)
    ax.set_yticks(ys, [r["variant"].replace("\\%", "%") for r in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("validation accuracy")
    ax.set_title("Ablation: what each preprocessing step buys")
    save_figure(fig, "ablation")
    base = rows[0]["validation"]
    latex_macros(
        {
            "AblNoAdapt": pct(rows[1]["validation"]),
            "AblNoSelect": pct(rows[2]["validation"]),
            "AblNeither": pct(rows[3]["validation"]),
            "AblShipped": pct(base),
        }
    )
    return {"variants": rows}

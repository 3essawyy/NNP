"""E4 — hyperparameter tuning of the shipped family (report §Tuning).

Grid: SVM ``C`` × ANOVA F-threshold (``γ = "scale"``), then ``γ`` multipliers at the best
cell. Selection rule (ADR-005/007): highest validation accuracy; ties within 0.2 pp are
broken by simulated-shift CV, then by the simpler model (smaller C, fewer features).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from nnp.experiments.common import (
    SEQUENTIAL,
    Data,
    figure,
    latex_macros,
    pct,
    save_figure,
    save_json,
)
from nnp.experiments.compare import SIM_SHIFT
from nnp.features import SelectionConfig
from nnp.models.cv import shifted_cv, validate
from nnp.models.final import FINAL_CONFIG

C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
F_GRID = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0)
GAMMA_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    k = 3
    cells: list[dict[str, Any]] = []
    grid = np.zeros((len(C_GRID), len(F_GRID)))
    for i, C in enumerate(C_GRID):
        for j, thr in enumerate(F_GRID):
            cfg = dataclasses.replace(
                FINAL_CONFIG,
                selection=SelectionConfig("anova", threshold=thr),
                model_params={"C": C, "gamma": "scale"},
            )
            val = validate(cfg, tr.X, tr.y, tr.devices, data.X_val, data.y_val)
            sh = (
                shifted_cv(cfg, tr.X, tr.y, tr.devices, SIM_SHIFT, k=k).mean
                if not quick
                else float("nan")
            )
            grid[i, j] = val.accuracy
            cells.append({"C": C, "f_threshold": thr, "validation": val.accuracy, "shifted_cv": sh})
            print(f"  C={C:<6g} F>{thr:<5g} val {val.accuracy:.4f}  shifted {sh:.4f}")

    best_val = max(c["validation"] for c in cells)
    near = [c for c in cells if c["validation"] >= best_val - 0.002]
    near.sort(key=lambda c: (-np.nan_to_num(c["shifted_cv"]), c["C"], -c["f_threshold"]))
    chosen = near[0]

    # gamma sweep at the chosen cell; gamma "scale" = 1 / (n_features * Var(X))
    gammas: list[dict[str, Any]] = []
    for g in GAMMA_MULT:
        thr = chosen["f_threshold"]
        n_feat_probe = dataclasses.replace(
            FINAL_CONFIG, selection=SelectionConfig("anova", threshold=thr)
        )
        from nnp.models.pipeline import HARPipeline

        p = HARPipeline(n_feat_probe).fit(tr.X, tr.y, tr.devices)
        base_gamma = float(p.model._gamma)
        cfg = dataclasses.replace(
            n_feat_probe, model_params={"C": chosen["C"], "gamma": base_gamma * g}
        )
        val = validate(cfg, tr.X, tr.y, tr.devices, data.X_val, data.y_val)
        gammas.append({"gamma_mult": g, "gamma": base_gamma * g, "validation": val.accuracy})
        print(f"  gamma x{g:<5g} val {val.accuracy:.4f}")

    save_json(
        "tuning",
        {
            "C_grid": C_GRID,
            "F_grid": F_GRID,
            "cells": cells,
            "chosen": chosen,
            "gamma_sweep": gammas,
            "rule": "max validation; ties (<=0.2pp) -> max simulated-shift CV -> smaller C -> "
            "higher F threshold",
        },
    )

    fig, ax = figure(4.8, 3.0)
    im = ax.imshow(grid, cmap=SEQUENTIAL, aspect="auto", vmin=grid.min() - 0.01, vmax=grid.max())
    ax.grid(False)
    ax.set_xticks(range(len(F_GRID)), [f"{f:g}" for f in F_GRID])
    ax.set_yticks(range(len(C_GRID)), [f"{c:g}" for c in C_GRID])
    ax.set_xlabel("ANOVA F-threshold")
    ax.set_ylabel("SVM C")
    for i in range(len(C_GRID)):
        for j in range(len(F_GRID)):
            v = grid[i, j]
            ax.text(
                j,
                i,
                pct(v),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v > grid.min() + 0.6 * (grid.max() - grid.min()) else "black",
            )
    ax.set_title("Validation accuracy (%)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save_figure(fig, "tuning")
    latex_macros(
        {
            "TunedC": f"{chosen['C']:g}",
            "TunedF": f"{chosen['f_threshold']:g}",
            "TunedVal": pct(chosen["validation"]),
            "GridMin": pct(float(grid.min())),
            "GridMax": pct(float(grid.max())),
        }
    )
    return {"chosen": chosen}

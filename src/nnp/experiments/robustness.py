"""E5 — robustness curves under controlled synthetic device shifts (report §Robustness).

One stratified 80/20 split of Device_B. The held-out 20 % is passed through
:func:`nnp.evaluation.shift.apply_shift` with increasing offset, gain spread, noise and outlier
rate; the shipped pipeline (per-device adaptation) is compared with the same pipeline using a
single global scaler.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from nnp.evaluation.shift import ShiftSpec, apply_shift
from nnp.experiments.common import SERIES, Data, figure, save_figure, save_json
from nnp.models.final import FINAL_CONFIG
from nnp.models.pipeline import HARPipeline

SWEEPS: dict[str, tuple[str, list[float]]] = {
    "offset": ("calibration offset (× feature std)", [0, 0.5, 1, 2, 5, 10, 20]),
    "gain_sigma": ("log-gain spread σ", [0, 0.1, 0.2, 0.4, 0.8]),
    "noise": ("additive noise (× feature std)", [0, 0.1, 0.25, 0.5, 1.0]),
    "outlier_fraction": ("outlier-record rate", [0, 0.01, 0.05, 0.1, 0.2]),
}


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    idx = np.arange(tr.X.shape[0])
    itr, ite = train_test_split(idx, test_size=0.2, stratify=tr.y, random_state=0)
    pipes = {
        "adapted": HARPipeline(FINAL_CONFIG).fit(tr.X[itr], tr.y[itr], tr.devices[itr]),
        "global scaler": HARPipeline(dataclasses.replace(FINAL_CONFIG, adapt=False)).fit(
            tr.X[itr], tr.y[itr], tr.devices[itr]
        ),
    }
    base = ShiftSpec(offset=0, offset_fraction=0.5, gain_sigma=0, noise=0, outlier_fraction=0)
    out: dict[str, Any] = {}
    for key, (label, values) in SWEEPS.items():
        curves: dict[str, list[float]] = {n: [] for n in pipes}
        for v in values:
            spec = dataclasses.replace(base, **{key: v})
            Xs = apply_shift(tr.X[ite], spec, np.random.default_rng(1))
            for n, p in pipes.items():
                curves[n].append(float((p.predict(Xs) == tr.y[ite]).mean()))
        out[key] = {"label": label, "values": values, "curves": curves}
        print(f"  {key}: " + "; ".join(f"{n} {np.round(c, 3).tolist()}" for n, c in curves.items()))

        fig, ax = figure(3.0, 2.3)
        for j, (n, c) in enumerate(curves.items()):
            ax.plot(values, c, marker="o", ms=4, lw=2, color=SERIES[j], label=n)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(label)
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7)
        save_figure(fig, f"robustness_{key}")
    save_json("robustness", out)
    return out

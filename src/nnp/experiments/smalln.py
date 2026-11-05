"""E7 — small test sets: what happens when ``infer.py`` gets few files (ADR-013).

Adaptation needs a batch: a device split needs enough records per cluster, and per-device
statistics need enough records to be estimated. This stage draws random subsets of the
validation set (N = 1 … 200), either **mixed** (both devices) or **single-device**, and
compares the pipeline variants below. The full validation score must not change.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from nnp.experiments.common import SERIES, Data, figure, latex_macros, pct, save_figure, save_json
from nnp.models.final import FINAL_CONFIG
from nnp.models.pipeline import HARPipeline, PipelineConfig

NS = (1, 5, 10, 20, 30, 45, 60, 100, 200)
HALF = 1545  # validation files 1-1545 / 1546-3090 are the two devices (EXP-001)


def variants() -> list[tuple[str, PipelineConfig]]:
    f = FINAL_CONFIG
    r = dataclasses.replace
    before = r(
        f,
        fallback_model=None,
        min_stats_size=30,
        cluster=r(f.cluster, min_size=30),
    )
    return [
        ("no fallback, min 30 (S002)", before),
        ("+ LR fallback, min 30", r(before, fallback_model="logreg")),
        ("shipped: LR fallback, min 5/10", f),
    ]


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    Xv, yv = data.X_val, data.y_val
    out: dict[str, Any] = {"N": NS, "results": []}
    for name, cfg in variants():
        p = HARPipeline(cfg).fit(tr.X, tr.y, tr.devices)
        full = float((p.predict(Xv) == yv).mean())
        res: dict[str, Any] = {"variant": name, "full_validation": full}
        for mode in ("mixed", "single"):
            rng = np.random.default_rng(0)
            accs = []
            for n in NS:
                draws = (20 if quick else 100) if n < 100 else (5 if quick else 20)
                a = []
                for _ in range(draws):
                    if mode == "mixed":
                        idx = rng.choice(len(yv), n, replace=False)
                    else:
                        idx = rng.choice(HALF, n, replace=False) + HALF * int(rng.integers(2))
                    a.append(float((p.predict(Xv[idx]) == yv[idx]).mean()))
                accs.append(float(np.mean(a)))
            res[mode] = accs
        print(f"  {name:32s} full {full:.4f}")
        print(f"    mixed  {[round(100 * a, 1) for a in res['mixed']]}")
        print(f"    single {[round(100 * a, 1) for a in res['single']]}")
        out["results"].append(res)
    save_json("smalln", out)

    for mode in ("mixed", "single"):
        fig, ax = figure(3.2, 2.4)
        for j, r in enumerate(out["results"]):
            ax.plot(
                range(len(NS)), r[mode], marker="o", ms=3, lw=2, color=SERIES[j], label=r["variant"]
            )
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set_xlabel("files in the test folder (N)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.6, 1.0)
        ax.set_title(f"{mode}-device test sets")
        ax.legend(fontsize=6)
        save_figure(fig, f"smalln_{mode}")

    shipped = out["results"][-1]
    before = out["results"][0]
    i30 = NS.index(30)
    latex_macros(
        {
            "SmallNBeforeMixedThirty": pct(before["mixed"][i30]),
            "SmallNAfterMixedThirty": pct(shipped["mixed"][i30]),
            "SmallNBeforeSingleTwenty": pct(before["single"][NS.index(20)]),
            "SmallNAfterSingleTwenty": pct(shipped["single"][NS.index(20)]),
            "SmallNAfterOne": pct(shipped["mixed"][0]),
        }
    )
    return out

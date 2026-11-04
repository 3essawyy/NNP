"""E1 — exploratory data analysis and shift diagnosis (report §Dataset / §Shift analysis)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from nnp.config import CLASS_NAMES, N_CLASSES
from nnp.experiments.common import (
    SERIES,
    TEXT_SECONDARY,
    Data,
    figure,
    latex_macros,
    latex_table,
    pct,
    save_figure,
    save_json,
)
from nnp.models.final import FINAL_CONFIG
from nnp.preprocessing import DeviceClusterer, DomainStats


def run(data: Data, quick: bool = False) -> dict[str, Any]:
    tr = data.train
    X, y = tr.X, tr.y
    Xv, yv = data.X_val, data.y_val
    out: dict[str, Any] = {
        "train_shape": list(X.shape),
        "train_devices": list(tr.device_names),
        "validation_shape": list(Xv.shape),
        "train_class_counts": np.bincount(y, minlength=N_CLASSES).tolist(),
        "validation_class_counts": np.bincount(yv, minlength=N_CLASSES).tolist(),
        "train_missing": int((~np.isfinite(X)).sum()),
        "validation_missing": int((~np.isfinite(Xv)).sum()),
    }
    in_range = (X.min(0) >= -1.0001) & (X.max(0) <= 1.0001)
    out["train_features_in_unit_range"] = int(in_range.sum())

    # ---- ANOVA F-scores on training data (train only, no validation labels)
    F, _ = f_classif(X, y)
    out["fscore_median_by_block"] = {
        f"{s}-{min(s + 39, 560)}": float(np.median(F[s : s + 40])) for s in range(0, 561, 40)
    }
    thr = FINAL_CONFIG.selection.threshold or 0.0
    out["n_features_above_threshold"] = int((thr < F).sum())

    fig, ax = figure(6.2, 2.6)
    idx = np.arange(F.size)
    ax.scatter(idx[thr < F], F[thr < F], s=6, color=SERIES[0], label=f"kept (F > {thr:g})")
    ax.scatter(idx[thr >= F], F[thr >= F], s=6, color=SERIES[1], label="dropped")
    ax.axhline(thr, color=TEXT_SECONDARY, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("feature index")
    ax.set_ylabel("ANOVA F-score (train)")
    ax.set_title("Per-feature class separability on Device_B")
    ax.legend(loc="lower right", markerscale=2)
    save_figure(fig, "fscore")

    # ---- per-block kNN accuracy (noise evidence), in-domain 3-fold
    Z = DomainStats.fit(X).transform(X)
    blocks: list[dict[str, Any]] = []
    step = 40
    for s in range(0, 561, step):
        cols = np.arange(s, min(s + step, 561))
        acc = cross_val_score(KNeighborsClassifier(10), Z[:, cols], y, cv=3).mean()
        blocks.append({"block": f"{s}-{cols[-1]}", "start": s, "knn_cv": float(acc)})
    out["block_knn"] = blocks
    fig, ax = figure(6.2, 2.4)
    ax.bar(
        [b["start"] + step / 2 for b in blocks],
        [b["knn_cv"] for b in blocks],
        width=step - 4,
        color=SERIES[0],
    )
    ax.axhline(1 / N_CLASSES, color=TEXT_SECONDARY, lw=1, ls="--")
    ax.text(560, 1 / N_CLASSES + 0.02, "chance", ha="right", color=TEXT_SECONDARY, fontsize=8)
    ax.set_xlabel("feature block (40 features each, start index)")
    ax.set_ylabel("kNN accuracy (3-fold, train)")
    ax.set_ylim(0, 1)
    ax.set_title("Feature blocks 40-239 carry almost no distance-based signal")
    save_figure(fig, "block_knn")

    # ---- device discovery on the validation set
    clus = DeviceClusterer(FINAL_CONFIG.cluster).fit(X, y)
    assignment = clus.partition(Xv)
    out["validation_devices_found"] = assignment.k
    out["validation_device_sizes"] = np.bincount(assignment.labels).tolist()
    out["validation_device_silhouette"] = assignment.silhouette
    # where are the devices in file order?
    first = [int(np.flatnonzero(assignment.labels == c)[0]) + 1 for c in range(assignment.k)]
    out["validation_device_first_file"] = first
    # calibration offset of discovered device vs train, per feature
    offsets = []
    for c in range(assignment.k):
        m = Xv[assignment.labels == c].mean(0) - X.mean(0)
        offsets.append(
            {
                "device": c,
                "n_features_offset_gt_1": int((np.abs(m) > 1).sum()),
                "median_offset_on_those": float(np.median(m[np.abs(m) > 1]))
                if (np.abs(m) > 1).any()
                else 0.0,
            }
        )
    out["validation_device_offsets"] = offsets

    from sklearn.decomposition import PCA

    if clus.probe_ is not None and clus.loc_ is not None and clus.scale_ is not None:
        Zp = PCA(2, random_state=0).fit_transform((Xv[:, clus.probe_] - clus.loc_) / clus.scale_)
        Zt = PCA(2, random_state=0).fit((Xv[:, clus.probe_] - clus.loc_) / clus.scale_)
        Ztrain = Zt.transform((X[:, clus.probe_] - clus.loc_) / clus.scale_)
        fig, ax = figure(4.6, 3.2)
        ax.scatter(Ztrain[:, 0], Ztrain[:, 1], s=4, color=SERIES[2], alpha=0.4, label="train")
        for c in range(assignment.k):
            m = assignment.labels == c
            ax.scatter(
                Zp[m, 0], Zp[m, 1], s=4, color=SERIES[c % 2], alpha=0.5, label=f"val. device {c}"
            )
        ax.set_xlabel("probe PC1")
        ax.set_ylabel("probe PC2")
        ax.set_title("Devices discovered from activity-independent features")
        ax.legend(markerscale=3, loc="best")
        save_figure(fig, "device_clusters")

    # ---- class balance figure
    fig, ax = figure(6.2, 2.4)
    w = 0.38
    xs = np.arange(N_CLASSES)
    tc = np.array(out["train_class_counts"]) / len(y)
    vc = np.array(out["validation_class_counts"]) / len(yv)
    ax.bar(xs - w / 2 - 0.01, tc, w, color=SERIES[0], label="train (Device_B)")
    ax.bar(xs + w / 2 + 0.01, vc, w, color=SERIES[1], label="validation")
    ax.set_xticks(xs, [n.replace("Walking ", "Walk. ") for n in CLASS_NAMES])
    ax.set_ylabel("share of records")
    ax.set_title("Class balance")
    ax.legend()
    save_figure(fig, "class_balance")

    rows = [
        [
            name.replace("_", " "),
            str(out["train_class_counts"][i]),
            str(out["validation_class_counts"][i]),
        ]
        for i, name in enumerate(CLASS_NAMES)
    ]
    rows.append(["Total", str(len(y)), str(len(yv))])
    latex_table(
        "dataset",
        ["Activity", "Train (Device\\_B)", "Validation"],
        [[f"{i} {r[0]}" if i < N_CLASSES else r[0], r[1], r[2]] for i, r in enumerate(rows)],
        "Records per class. Device\\_A is empty in the release.",
        "tab:dataset",
    )
    latex_macros(
        {
            "NTrain": str(len(y)),
            "NVal": str(len(yv)),
            "FThreshold": f"{thr:g}",
            "ValDevices": str(assignment.k),
            "ValSilhouette": f"{assignment.silhouette or 0:.3f}",
            "NoiseKnn": pct(
                float(np.mean([b["knn_cv"] for b in blocks if 40 <= b["start"] < 240]))
            ),
        }
    )
    save_json("eda", out)
    return out

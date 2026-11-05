"""The configuration of the shipped model (chosen by experiments E4 — see ADR-007, EXP-004).

Change this only together with a new experiments run and an ADR update: the report, the
committed ``models/model.pkl`` and this file must agree.
"""

from __future__ import annotations

from nnp.features import SelectionConfig
from nnp.models.pipeline import PipelineConfig
from nnp.preprocessing import ClusterConfig

FINAL_CONFIG = PipelineConfig(
    scaling="zscore",
    clip=3.0,
    adapt=True,
    selection=SelectionConfig(method="anova", threshold=100.0),
    model="svc_rbf",
    model_params={"C": 100.0, "gamma": "scale"},
    # Small test sets (ADR-013, EXP-007): split devices from 5 records per cluster, trust a
    # device's own statistics from 12 records, otherwise use training statistics with a
    # logistic-regression fallback (87.7 % per single file vs 75.9 % for the SVM).
    cluster=ClusterConfig(min_size=5),
    min_stats_size=12,
    fallback_model="logreg",
)

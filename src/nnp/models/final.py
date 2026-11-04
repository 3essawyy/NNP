"""The configuration of the shipped model (chosen by experiments — see ADR-007).

Change this only together with a new experiments run and an ADR update: the report, the
committed ``models/model.pkl`` and this file must agree.
"""

from __future__ import annotations

from nnp.features import SelectionConfig
from nnp.models.pipeline import PipelineConfig

FINAL_CONFIG = PipelineConfig(
    scaling="zscore",
    clip=8.0,
    adapt=True,
    selection=SelectionConfig(method="anova", threshold=100.0),
    model="svc_rbf",
    model_params={"C": 100.0, "gamma": "scale"},
)

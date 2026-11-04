"""Feature selection / dimensionality reduction (see docs/modules/features, ADR-006)."""

from nnp.features.reduction import Reducer, ReductionConfig
from nnp.features.selection import FeatureSelector, SelectionConfig

__all__ = ["FeatureSelector", "Reducer", "ReductionConfig", "SelectionConfig"]

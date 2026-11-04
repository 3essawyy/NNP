from __future__ import annotations

import numpy as np
import pytest

from nnp.features import FeatureSelector, Reducer, ReductionConfig, SelectionConfig
from nnp.preprocessing import DomainStats
from tests.conftest import NOISY, Synthetic


def test_anova_threshold_drops_noise_block(synthetic: Synthetic) -> None:
    Z = DomainStats.fit(synthetic.X).transform(synthetic.X)
    sel = FeatureSelector(SelectionConfig("anova", threshold=50)).fit(Z, synthetic.y)
    assert sel.support_ is not None
    kept_noise = np.isin(sel.support_, NOISY).mean()
    assert kept_noise == 0.0
    assert sel.support_.size > 250


def test_top_k_and_none(synthetic: Synthetic) -> None:
    Z = DomainStats.fit(synthetic.X).transform(synthetic.X)
    assert FeatureSelector(SelectionConfig("anova", k=17)).fit(Z, synthetic.y).transform(
        Z
    ).shape == (Z.shape[0], 17)
    assert (
        FeatureSelector(SelectionConfig("none")).fit(Z, synthetic.y).transform(Z).shape == Z.shape
    )


def test_selection_that_removes_everything_is_an_error(synthetic: Synthetic) -> None:
    Z = DomainStats.fit(synthetic.X).transform(synthetic.X)
    with pytest.raises(ValueError, match="removed every feature"):
        FeatureSelector(SelectionConfig("anova", threshold=1e12)).fit(Z, synthetic.y)


@pytest.mark.parametrize(("method", "dims"), [("pca", 10), ("lda", 5), ("none", 561)])
def test_reducer_shapes(synthetic: Synthetic, method: str, dims: int) -> None:
    Z = DomainStats.fit(synthetic.X).transform(synthetic.X)
    r = Reducer(ReductionConfig(method, 10 if method == "pca" else 5)).fit(Z, synthetic.y)  # type: ignore[arg-type]
    assert r.transform(Z).shape == (Z.shape[0], dims)

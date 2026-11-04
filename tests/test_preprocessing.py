from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from nnp.preprocessing import DeviceClusterer, DomainStats, standardize_by_domain
from tests.conftest import Synthetic


def test_per_domain_standardization_cancels_affine_calibration() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 30))
    a = rng.uniform(0.5, 2.0, size=30)
    b = rng.normal(0, 10, size=30)
    for scaling in ("zscore", "robust"):
        z1 = DomainStats.fit(X, scaling).transform(X)
        z2 = DomainStats.fit(a * X + b, scaling).transform(a * X + b)
        np.testing.assert_allclose(z1, z2, atol=1e-9)


def test_constant_feature_and_nan_are_safe() -> None:
    X = np.ones((10, 3))
    X[:, 1] = np.arange(10)
    X[2, 2] = np.nan
    s = DomainStats.fit(X)
    Z = s.transform(X)
    assert np.isfinite(Z).all()
    assert (Z[:, 0] == 0).all()  # constant → scale 1 → 0
    assert Z[2, 2] == 0  # NaN imputed with the location


def test_clip() -> None:
    X = np.r_[np.zeros(99), [1000.0]][:, None]
    assert DomainStats.fit(X).transform(X, clip=3.0).max() == 3.0


def test_standardize_by_domain_keys() -> None:
    X = np.r_[np.zeros((5, 2)), np.ones((5, 2)) * 7]
    d = np.array(["A"] * 5 + ["B"] * 5)
    _, stats = standardize_by_domain(X, d)
    assert set(stats) == {"A", "B"}


def test_clusterer_finds_two_devices(synthetic: Synthetic) -> None:
    clus = DeviceClusterer().fit(synthetic.X, synthetic.y)
    mixed = np.r_[synthetic.X[:300], synthetic.X_shift]
    truth = np.r_[np.zeros(300), np.ones(len(synthetic.X_shift))]
    a = clus.partition(mixed)
    assert a.k == 2
    assert adjusted_rand_score(truth, a.labels) == 1.0
    # single-sample assignment agrees with the batch labels
    assert all(a.assign(mixed[i]) == a.labels[i] for i in range(0, len(mixed), 37))


def test_clusterer_does_not_split_one_device_by_activity_or_outliers(
    synthetic: Synthetic,
) -> None:
    clus = DeviceClusterer().fit(synthetic.X, synthetic.y)
    X = synthetic.X_shift.copy()
    X[:8] += 50.0  # a handful of outlier records is not a device
    assert clus.partition(X).k == 1


def test_clusterer_tiny_set_is_one_device(synthetic: Synthetic) -> None:
    clus = DeviceClusterer().fit(synthetic.X, synthetic.y)
    assert clus.partition(synthetic.X_shift[:10]).k == 1

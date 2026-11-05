from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pytest

from nnp.evaluation.shift import ShiftSpec
from nnp.features import SelectionConfig
from nnp.models import MODEL_NAMES, HARPipeline, PipelineConfig, load_bundle, save_bundle
from nnp.models.cv import in_domain_cv, shifted_cv, validate
from nnp.models.fast import FastRBFSVC, GenericPredictor
from tests.conftest import Synthetic


@pytest.fixture(scope="module")
def fitted(synthetic: Synthetic) -> HARPipeline:
    return HARPipeline().fit(synthetic.X, synthetic.y)


def test_adaptation_beats_global_scaler_under_shift(synthetic: Synthetic) -> None:
    adapted = HARPipeline().fit(synthetic.X, synthetic.y)
    plain = HARPipeline(PipelineConfig(adapt=False)).fit(synthetic.X, synthetic.y)
    acc_a = (adapted.predict(synthetic.X_shift) == synthetic.y_shift).mean()
    acc_p = (plain.predict(synthetic.X_shift) == synthetic.y_shift).mean()
    assert acc_a > 0.95
    assert acc_a > acc_p + 0.3


def test_predict_one_matches_batch(fitted: HARPipeline, synthetic: Synthetic) -> None:
    mixed = np.r_[synthetic.X[:150], synthetic.X_shift[:150]]
    ad = fitted.adapt(mixed)
    assert ad.n_devices == 2
    batch = fitted.predict(mixed, ad)
    single = np.array([fitted.predict_one(x, ad) for x in mixed])
    np.testing.assert_array_equal(batch, single)


def test_fast_svc_is_exact(fitted: HARPipeline, synthetic: Synthetic) -> None:
    fast = FastRBFSVC(fitted.model)
    ad = fitted.adapt(synthetic.X_shift)
    Z = ad.stats[0].transform(synthetic.X_shift[:, fitted.support], fitted.config.clip)
    np.testing.assert_array_equal([fast(z) for z in Z], fitted.model.predict(Z))


def test_pickle_round_trip_drops_and_rebuilds_fast_path(
    fitted: HARPipeline, synthetic: Synthetic, tmp_path: Path
) -> None:
    clone = pickle.loads(pickle.dumps(fitted))  # noqa: S301 - our own object
    assert clone._fast is None
    ad = clone.adapt(synthetic.X_shift)
    assert clone.predict_one(synthetic.X_shift[0], ad) == fitted.predict_one(
        synthetic.X_shift[0], fitted.adapt(synthetic.X_shift)
    )
    path = save_bundle(fitted, tmp_path / "m.pkl")
    loaded, meta = load_bundle(path)
    assert meta["n_train"] == synthetic.X.shape[0]
    assert meta["config"]["model"] == "svc_rbf"
    np.testing.assert_array_equal(
        loaded.predict(synthetic.X_shift), fitted.predict(synthetic.X_shift)
    )


def test_bad_bundle_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps({"format": 999}))
    with pytest.raises(ValueError, match="unsupported"):
        load_bundle(p)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.pkl")


def test_tiny_cluster_falls_back_to_reference(fitted: HARPipeline, synthetic: Synthetic) -> None:
    ad = fitted.adapt(synthetic.X_shift[:5])
    assert ad.n_devices == 1
    assert ad.fallback == [True]


@pytest.mark.parametrize("name", [m for m in MODEL_NAMES if m not in {"lightgbm", "mlp"}])
def test_every_model_family_trains(synthetic: Synthetic, name: str) -> None:
    # top-40 features: every family (incl. QDA, which needs n_class_samples > n_features) fits
    cfg = PipelineConfig(model=name, selection=SelectionConfig("anova", threshold=None, k=40))
    p = HARPipeline(cfg).fit(synthetic.X[:300], synthetic.y[:300])
    pred = p.predict(synthetic.X_shift[:60])
    assert pred.shape == (60,)
    assert set(pred) <= set(range(6))
    from nnp.models.fast import FastLinear, FastRBFSVC

    expected = {"svc_rbf": FastRBFSVC, "logreg": FastLinear, "linear_svc": FastLinear}
    assert isinstance(p._fast, expected.get(name, GenericPredictor))


def test_cv_protocols_run(synthetic: Synthetic) -> None:
    cfg = PipelineConfig()
    dev = np.full(synthetic.X.shape[0], "B")
    assert in_domain_cv(cfg, synthetic.X, synthetic.y, dev, k=3).mean > 0.9
    sh = shifted_cv(cfg, synthetic.X, synthetic.y, dev, ShiftSpec(offset=5, noise=0.1), k=3)
    assert sh.mean > 0.9
    val = validate(cfg, synthetic.X, synthetic.y, dev, synthetic.X_shift, synthetic.y_shift)
    assert val.accuracy > 0.95
    plain = validate(
        dataclasses.replace(cfg, adapt=False),
        synthetic.X,
        synthetic.y,
        dev,
        synthetic.X_shift,
        synthetic.y_shift,
    )
    assert plain.n_devices == 1


def test_fast_linear_is_exact(synthetic: Synthetic) -> None:
    from nnp.models.fast import FastLinear, compile_predictor

    for name in ("logreg", "linear_svc"):
        cfg = PipelineConfig(model=name, selection=SelectionConfig("anova", threshold=None, k=40))
        p = HARPipeline(cfg).fit(synthetic.X, synthetic.y)
        fast = compile_predictor(p.model)
        assert isinstance(fast, FastLinear)
        ad = p.adapt(synthetic.X_shift)
        Z = ad.stats[0].transform(synthetic.X_shift[:, p.support], p.config.clip)
        np.testing.assert_array_equal([fast(z) for z in Z], p.model.predict(Z))


def test_fallback_model_used_only_when_adaptation_falls_back(synthetic: Synthetic) -> None:
    cfg = PipelineConfig(fallback_model="logreg")
    p = HARPipeline(cfg).fit(synthetic.X, synthetic.y)
    assert p.fallback_ is not None
    tiny = synthetic.X_shift[:5]
    ad_tiny = p.adapt(tiny)
    assert ad_tiny.fallback == [True]
    # tiny set: fallback model on reference-standardized features, batch == single
    z = ad_tiny.stats[0].transform(tiny[:, p.support], p.config.clip)
    expected = p.fallback_.predict(z)
    np.testing.assert_array_equal(p.predict(tiny, ad_tiny), expected)
    np.testing.assert_array_equal([p.predict_one(x, ad_tiny) for x in tiny], expected)
    # large set: main model, identical to a pipeline without a fallback model
    plain = HARPipeline(PipelineConfig()).fit(synthetic.X, synthetic.y)
    np.testing.assert_array_equal(p.predict(synthetic.X_shift), plain.predict(synthetic.X_shift))


def test_fallback_survives_pickle(synthetic: Synthetic) -> None:
    p = HARPipeline(PipelineConfig(fallback_model="logreg")).fit(synthetic.X, synthetic.y)
    clone = pickle.loads(pickle.dumps(p))  # noqa: S301 - our own object
    assert clone._fast_fallback is None
    ad = clone.adapt(synthetic.X_shift[:3])
    assert clone.predict_one(synthetic.X_shift[0], ad) == p.predict_one(
        synthetic.X_shift[0], p.adapt(synthetic.X_shift[:3])
    )

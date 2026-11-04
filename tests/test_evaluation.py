from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nnp.evaluation import ShiftSpec, apply_shift, evaluate
from nnp.evaluation.score import main as score_main


def test_evaluate_basic() -> None:
    y = np.array([0, 0, 3, 4, 4, 5])
    p = np.array([0, 1, 4, 4, 4, 5])
    ev = evaluate(y, p)
    assert ev.accuracy == pytest.approx(4 / 6)
    assert ev.confusion[3][4] == 1
    assert ev.top_confusions[0][2] == 1
    assert len(ev.per_class) == 6


def test_evaluate_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate(np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64))


def test_zero_shift_is_identity() -> None:
    X = np.random.default_rng(0).normal(size=(50, 8))
    spec = ShiftSpec(offset=0, gain_sigma=0, noise=0, outlier_fraction=0)
    np.testing.assert_allclose(apply_shift(X, spec, np.random.default_rng(1)), X)


def test_shift_changes_data_and_keeps_shape() -> None:
    X = np.random.default_rng(0).normal(size=(50, 8))
    Xs = apply_shift(X, ShiftSpec(noise=0.5, outlier_fraction=0.2), np.random.default_rng(1))
    assert Xs.shape == X.shape
    assert not np.allclose(Xs, X)


def test_score_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "r.txt").write_text("0\n1\n2\n")
    (tmp_path / "g.txt").write_text("0\n1\n3\n")
    (tmp_path / "t.txt").write_text("0.001\n0.002\n0.003\n")
    assert score_main([str(tmp_path / f) for f in ("r.txt", "g.txt", "t.txt")]) == 0
    assert '"accuracy": 0.6667' in capsys.readouterr().out
    (tmp_path / "g.txt").write_text("0\n")
    assert score_main([str(tmp_path / "r.txt"), str(tmp_path / "g.txt")]) == 1

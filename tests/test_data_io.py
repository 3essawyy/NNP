from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nnp.config import N_FEATURES
from nnp.data.io import (
    DataFormatError,
    list_sample_files,
    load_ground_truth,
    load_samples,
    load_training,
    read_sample,
)
from tests.conftest import write_device_folder, write_sample_folder

HEADER = ",".join(str(i) for i in range(N_FEATURES))


def test_numeric_order_not_lexicographic(tmp_path: Path) -> None:
    for i in (1, 2, 10, 3, 11):
        (tmp_path / f"{i}.csv").write_text(HEADER + "\n" + ",".join(["0"] * N_FEATURES) + "\n")
    assert [p.stem for p in list_sample_files(tmp_path)] == ["1", "2", "3", "10", "11"]


def test_non_numeric_sample_name_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "1.csv").write_text("x")
    (tmp_path / "sample.csv").write_text("x")
    with pytest.raises(DataFormatError):
        list_sample_files(tmp_path)


def test_non_csv_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "1.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in list_sample_files(tmp_path)] == ["1.csv"]


def test_read_sample_plain(tmp_path: Path) -> None:
    values = np.arange(N_FEATURES, dtype=float) / 7
    p = tmp_path / "1.csv"
    p.write_text(HEADER + "\n" + ",".join(f"{v:.6f}" for v in values) + "\n")
    np.testing.assert_allclose(read_sample(p), values, atol=1e-6)


def test_read_sample_drops_target_and_reorders(tmp_path: Path) -> None:
    cols = [*(str(i) for i in range(N_FEATURES - 1, -1, -1)), "target"]
    vals = [*(str(int(c)) for c in cols[:-1]), "3"]
    p = tmp_path / "1.csv"
    p.write_text(",".join(cols) + "\n" + ",".join(vals) + "\n")
    np.testing.assert_array_equal(read_sample(p), np.arange(N_FEATURES, dtype=float))


def test_read_sample_empty_field_is_nan(tmp_path: Path) -> None:
    cells = ["1"] * N_FEATURES
    cells[5] = ""
    p = tmp_path / "1.csv"
    p.write_text(HEADER + "\n" + ",".join(cells) + "\n")
    row = read_sample(p)
    assert np.isnan(row[5])
    assert np.isfinite(np.delete(row, 5)).all()


def test_read_sample_rejects_two_rows(tmp_path: Path) -> None:
    line = ",".join(["0"] * N_FEATURES)
    p = tmp_path / "1.csv"
    p.write_text(f"{HEADER}\n{line}\n{line}\n")
    with pytest.raises(DataFormatError):
        read_sample(p)


def test_read_sample_rejects_wrong_width(tmp_path: Path) -> None:
    p = tmp_path / "1.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(DataFormatError):
        read_sample(p)


def test_load_training_skips_empty_device(tmp_path: Path) -> None:
    X = np.random.default_rng(0).normal(size=(20, N_FEATURES))
    y = np.arange(20) % 6
    (tmp_path / "Device_A").mkdir()
    write_device_folder(tmp_path, "Device_B", X, y)
    ts = load_training(tmp_path)
    assert ts.device_names == ("Device_B",)
    assert ts.X.shape == (20, N_FEATURES)
    assert set(ts.devices) == {"Device_B"}
    np.testing.assert_allclose(ts.X, X)


def test_load_training_rejects_bad_labels(tmp_path: Path) -> None:
    X = np.zeros((3, N_FEATURES))
    write_device_folder(tmp_path, "Device_B", X, np.array([0, 1, 9]))
    with pytest.raises(DataFormatError):
        load_training(tmp_path)


def test_load_samples_and_ground_truth(tmp_path: Path) -> None:
    X = np.random.default_rng(1).normal(size=(12, N_FEATURES))
    write_sample_folder(tmp_path / "data", X)
    files, Xr = load_samples(tmp_path / "data")
    assert [int(f.stem) for f in files] == list(range(1, 13))
    np.testing.assert_allclose(Xr, X, rtol=1e-12)
    gt = tmp_path / "gt.txt"
    gt.write_text("0\n5\n3\n")
    np.testing.assert_array_equal(load_ground_truth(gt), [0, 5, 3])

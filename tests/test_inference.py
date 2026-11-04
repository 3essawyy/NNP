"""End-to-end contract of ``infer.py`` (brief §9): ordering, file formats, timing rule."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nnp.config import REPO_ROOT
from nnp.inference.predictor import infer_folder, resolve_data_dir
from nnp.models import HARPipeline, save_bundle
from tests.conftest import Synthetic, write_sample_folder


@pytest.fixture(scope="module")
def model_path(synthetic: Synthetic, tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = HARPipeline().fit(synthetic.X, synthetic.y)
    return save_bundle(p, tmp_path_factory.mktemp("model") / "model.pkl")


@pytest.fixture(scope="module")
def data_dir(synthetic: Synthetic, tmp_path_factory: pytest.TempPathFactory) -> Path:
    # 120 samples → names 1..120 exercise numeric (not lexicographic) ordering
    return write_sample_folder(tmp_path_factory.mktemp("eval") / "data", synthetic.X_shift[:120])


def test_infer_cli_writes_brief_formats(model_path: Path, data_dir: Path, tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, str(REPO_ROOT / "infer.py"), str(data_dir), "--model", str(model_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    results = (tmp_path / "results.txt").read_text()
    times = (tmp_path / "time.txt").read_text()
    r_lines = results.splitlines()
    t_lines = times.splitlines()
    assert len(r_lines) == len(t_lines) == 120
    assert all(re.fullmatch(r"[0-5]", ln) for ln in r_lines)
    assert all(re.fullmatch(r"\d+\.\d{3}", ln) for ln in t_lines)
    assert results.endswith("\n")


def test_predictions_follow_numeric_file_order(
    synthetic: Synthetic, model_path: Path, data_dir: Path, tmp_path: Path
) -> None:
    run = infer_folder(data_dir, model_path, tmp_path)
    assert [int(f.stem) for f in run.files] == list(range(1, 121))
    acc = (np.array(run.predictions) == synthetic.y_shift[:120]).mean()
    assert acc > 0.9  # only possible if line i really is file i.csv


def test_amortized_adaptation_is_charged(model_path: Path, data_dir: Path, tmp_path: Path) -> None:
    run = infer_folder(data_dir, model_path, tmp_path)
    total = sum(run.seconds)
    assert total >= run.adapt_seconds  # every millisecond of adaptation shows up in time.txt


def test_unreadable_file_still_yields_one_line(
    model_path: Path, tmp_path: Path, synthetic: Synthetic
) -> None:
    d = write_sample_folder(tmp_path / "data", synthetic.X_shift[:40])
    (d / "41.csv").write_text("garbage\n")
    run = infer_folder(d, model_path, tmp_path)
    assert len(run.predictions) == 41
    assert len((tmp_path / "results.txt").read_text().splitlines()) == 41


def test_parent_folder_is_accepted(data_dir: Path) -> None:
    assert resolve_data_dir(data_dir.parent) == data_dir
    assert resolve_data_dir(data_dir) == data_dir


def test_empty_folder_gives_empty_outputs(model_path: Path, tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    run = infer_folder(tmp_path / "data", model_path, tmp_path)
    assert run.predictions == []
    assert (tmp_path / "results.txt").read_text() == ""

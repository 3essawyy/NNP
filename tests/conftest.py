"""Synthetic HAR-like data so the suite runs without the (gitignored) course dataset.

The generator mimics the structure found in the real data (docs/modules/data):
informative features, a block of heavy-noise features with weak class signal, and a second
"device" that applies a calibration offset + gain to a subset of features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from nnp.config import N_CLASSES, N_FEATURES

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

NOISY = np.arange(40, 240)


@dataclass(frozen=True)
class Synthetic:
    X: FloatArray
    y: IntArray
    X_shift: FloatArray
    y_shift: IntArray


def make_synthetic(n: int = 600, n_shift: int = 300, seed: int = 0) -> Synthetic:
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1.0, size=(N_CLASSES, N_FEATURES))
    means[:, NOISY] *= 0.05  # weak signal …
    noise_sd = np.ones(N_FEATURES) * 0.6
    noise_sd[NOISY] = 3.0  # … buried in heavy noise

    def sample(m: int) -> tuple[FloatArray, IntArray]:
        y = rng.integers(0, N_CLASSES, size=m).astype(np.int64)
        X = means[y] + rng.normal(size=(m, N_FEATURES)) * noise_sd
        return X.astype(np.float64), y

    X, y = sample(n)
    X = X + 10.0 * (np.arange(N_FEATURES) < 250)  # source device calibration offset
    Xs, ys = sample(n_shift)
    gain = np.exp(rng.normal(0, 0.2, size=N_FEATURES))
    Xs = Xs * gain + 3.0 * (np.arange(N_FEATURES) < 120)  # different offset + gain
    return Synthetic(X, y, Xs, ys)


@pytest.fixture(scope="session")
def synthetic() -> Synthetic:
    return make_synthetic()


def write_device_folder(root: Path, name: str, X: FloatArray, y: IntArray) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[str(i) for i in range(N_FEATURES)])
    df["target"] = y
    df.to_csv(folder / "train_data.csv", index=False)
    return folder


def write_sample_folder(folder: Path, X: FloatArray) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    cols = [str(i) for i in range(N_FEATURES)]
    for i, row in enumerate(X, start=1):
        pd.DataFrame([row], columns=cols).to_csv(folder / f"{i}.csv", index=False)
    return folder

"""Reading training device folders, evaluation sample folders and label files.

Contracts (brief §2.2, §9.1):

* A training *device folder* (e.g. ``Device_B/``) holds one or more CSVs with columns
  ``"0"`` … ``"560"`` plus ``"target"``. Empty device folders are skipped with a warning
  (the released ``Device_A/`` is empty — ADR-003).
* An evaluation folder holds ``1.csv`` … ``N.csv``; each file is **one** sample (header row +
  one data row). Files must be processed in increasing *numerical* order, so ``10.csv`` comes
  after ``9.csv`` (a plain string sort would put it after ``1.csv``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from nnp.config import N_CLASSES, N_FEATURES, TARGET_COLUMN

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_NUMERIC_STEM = re.compile(r"^\d+$")
_FEATURE_COLUMNS = [str(i) for i in range(N_FEATURES)]


class DataFormatError(ValueError):
    """A file does not match the documented dataset format."""


@dataclass(frozen=True)
class TrainingSet:
    """All labelled training records, with the device each record came from."""

    X: FloatArray
    y: IntArray
    devices: npt.NDArray[np.str_]
    device_names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.y.shape != (n,) or self.devices.shape != (n,):
            raise DataFormatError("X, y and devices must have the same number of rows")


def _feature_matrix(df: pd.DataFrame, source: Path) -> FloatArray:
    missing = [c for c in _FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataFormatError(f"{source}: missing feature columns, e.g. {missing[:5]}")
    return df[_FEATURE_COLUMNS].to_numpy(dtype=np.float64)


def load_device_folder(folder: Path) -> tuple[FloatArray, IntArray]:
    """Read every CSV in one device folder; returns ``(X, y)`` (possibly 0 rows)."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        return np.empty((0, N_FEATURES)), np.empty(0, dtype=np.int64)
    Xs: list[FloatArray] = []
    ys: list[IntArray] = []
    for f in files:
        df = pd.read_csv(f)
        if TARGET_COLUMN not in df.columns:
            raise DataFormatError(f"{f}: no '{TARGET_COLUMN}' column")
        y = df[TARGET_COLUMN].to_numpy()
        if not np.isin(y, np.arange(N_CLASSES)).all():
            raise DataFormatError(f"{f}: labels outside 0..{N_CLASSES - 1}")
        Xs.append(_feature_matrix(df, f))
        ys.append(y.astype(np.int64))
    return np.vstack(Xs), np.concatenate(ys)


def load_training(root: Path) -> TrainingSet:
    """Load every ``Device_*`` sub-folder of ``root`` into one :class:`TrainingSet`."""
    if not root.is_dir():
        raise FileNotFoundError(f"training root not found: {root} (run `uv run poe data`)")
    Xs: list[FloatArray] = []
    ys: list[IntArray] = []
    devs: list[npt.NDArray[np.str_]] = []
    names: list[str] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        X, y = load_device_folder(folder)
        if X.shape[0] == 0:
            log.warning("device folder %s is empty; skipped", folder.name)
            continue
        Xs.append(X)
        ys.append(y)
        devs.append(np.full(X.shape[0], folder.name))
        names.append(folder.name)
    if not Xs:
        raise DataFormatError(f"no training records under {root}")
    return TrainingSet(np.vstack(Xs), np.concatenate(ys), np.concatenate(devs), tuple(names))


def list_sample_files(folder: Path) -> list[Path]:
    """Return ``folder``'s ``<int>.csv`` files sorted by their integer stem.

    Raises :class:`DataFormatError` for a ``.csv`` whose stem is not an integer, because
    silently skipping or mis-ordering a sample would shift every later prediction line.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"evaluation folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    bad = [p.name for p in files if not _NUMERIC_STEM.match(p.stem)]
    if bad:
        raise DataFormatError(f"non-numeric sample file names in {folder}: {bad[:5]}")
    return sorted(files, key=lambda p: int(p.stem))


def read_sample(path: Path) -> FloatArray:
    """Read one sample CSV into a ``(561,)`` float vector.

    A plain-text parser (≈20× faster than ``pandas.read_csv`` for a 1-row, 561-column file).
    The header decides column order: if it names ``"0"`` … ``"560"`` the values are reordered
    by name, and a ``target`` column is dropped. Empty fields become NaN. A file with more than
    one data row is an error: one file must give exactly one line in ``results.txt``.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) != 2:
        raise DataFormatError(f"{path}: expected a header + 1 data row, found {len(lines)} lines")
    header = [h.strip().strip('"') for h in lines[0].split(",")]
    cells = [c.strip() for c in lines[1].split(",")]
    if len(cells) != len(header):
        raise DataFormatError(f"{path}: {len(header)} header fields but {len(cells)} values")
    values = np.array([float(c) if c else np.nan for c in cells], dtype=np.float64)
    if header[: len(_FEATURE_COLUMNS)] == _FEATURE_COLUMNS and len(header) in (
        N_FEATURES,
        N_FEATURES + 1,
    ):
        row = values[:N_FEATURES]
    elif set(_FEATURE_COLUMNS) <= set(header):
        index = {h: i for i, h in enumerate(header)}
        row = values[[index[c] for c in _FEATURE_COLUMNS]]
    else:
        keep = [i for i, h in enumerate(header) if h != TARGET_COLUMN]
        if len(keep) != N_FEATURES:
            raise DataFormatError(f"{path}: expected {N_FEATURES} features, got {len(keep)}")
        row = values[keep]
    return np.asarray(row, dtype=np.float64)


def load_samples(folder: Path) -> tuple[list[Path], FloatArray]:
    """Read every sample of an evaluation folder in numeric order → ``(files, X)``."""
    files = list_sample_files(folder)
    if not files:
        return files, np.empty((0, N_FEATURES))
    return files, np.vstack([read_sample(f) for f in files])


def load_ground_truth(path: Path) -> IntArray:
    """Read a ``ground_truth.txt``/``results.txt`` style file: one integer label per line."""
    tokens = path.read_text().split()
    return np.array([int(t) for t in tokens], dtype=np.int64)

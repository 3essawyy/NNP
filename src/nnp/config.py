"""Project-wide constants and default paths.

Everything here is plain data so every module can import it (it sits below ``nnp.data``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

N_FEATURES: Final = 561
"""Number of numeric feature columns per record (columns ``"0"`` … ``"560"``)."""

TARGET_COLUMN: Final = "target"

CLASS_NAMES: Final[tuple[str, ...]] = (
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
)
N_CLASSES: Final = len(CLASS_NAMES)

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
"""Repository root (``src/nnp/config.py`` → two levels up from ``src``)."""

DATA_DIR: Final = REPO_ROOT / "data"
DATASET_DIR: Final = DATA_DIR / "radar_project_data"
TRAIN_DIR: Final = DATASET_DIR / "student_realease"  # sic: folder name as shipped
VALIDATION_DIR: Final = DATASET_DIR / "eval_data" / "data"
VALIDATION_LABELS: Final = DATASET_DIR / "eval_data" / "ground_truth.txt"

MODEL_PATH: Final = REPO_ROOT / "models" / "model.pkl"
RESULTS_DIR: Final = REPO_ROOT / "experiments" / "results"
FIGURES_DIR: Final = REPO_ROOT / "report" / "figures"
GENERATED_TEX_DIR: Final = REPO_ROOT / "report" / "generated"

DATASET_URL: Final = (
    "https://drive.usercontent.google.com/download"
    "?id=1oZWHi8zYDAn_FJ55iYq3tAJLaRlzU5Jn&export=download&confirm=t"
)
"""Course dataset (radar_project_data.7z, ~18.7 MB) from the brief's Drive link."""

SEED: Final = 0

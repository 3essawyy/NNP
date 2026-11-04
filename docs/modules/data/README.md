# Module: data

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/data/` (`io.py`, `download.py`), constants in `src/nnp/config.py`
- **Tasks:** T-003
- **ADRs:** ADR-003

## Responsibility
Turn files on disk into numpy arrays, with the brief's ordering guarantees. Does **not** clean, scale or interpret values (that is preprocessing).

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `load_training(root) -> TrainingSet(X, y, devices, device_names)` | every `Device_*` sub-folder; empty folders skipped with a warning; labels must be 0–5 |
| Provides | `list_sample_files(folder) -> list[Path]` | `<int>.csv` sorted by **integer** stem; a non-numeric `.csv` name raises `DataFormatError`; non-CSV files ignored |
| Provides | `read_sample(path) -> (561,)` | header + exactly 1 data row; reorders by header names `"0"…"560"`; drops `target`; empty cell → NaN |
| Provides | `load_samples(folder)`, `load_ground_truth(path)` | convenience |
| Provides | `nnp-download` / `poe data` | downloads + extracts the course archive into `data/` |
| Consumes | filesystem | — |

## Data
- Training CSV: 561 float columns `"0"…"560"` + `target` (int).
- Sample CSV: same header without `target`, one row.
- Constants: `N_FEATURES=561`, `CLASS_NAMES`, default paths (`DATASET_DIR`, `TRAIN_DIR`, `VALIDATION_DIR`, `MODEL_PATH`…).

## Internal design
`read_sample` is a hand-rolled parser (split lines/commas, `float()`): ~1 ms per file vs ~23 ms with `pandas.read_csv` for a 561-column single row — the full 3090-file validation run went from 72 s to ~5 s wall time (I/O is not in `time.txt`, but graders still wait for it). Predictions are byte-identical (checked in S002).

## Validation and failure modes
| Failure | Detection | Behavior |
|---|---|---|
| `10.csv` sorted before `2.csv` | numeric sort | guaranteed order (test `test_numeric_order_not_lexicographic`) |
| Stray `notes.txt` in data/ | suffix filter | ignored |
| `sample.csv` (non-numeric) | regex | `DataFormatError` (refuse to guess the order) |
| Malformed sample in inference | exception | inference module substitutes an all-NaN row so the line count stays N |
| Missing dataset | `FileNotFoundError` | message says `uv run poe data` |

## Tests
`tests/test_data_io.py` (ordering, parsing variants, empty device folder, bad labels).

## Known limitations
Assumes comma separators and a header row (as released).

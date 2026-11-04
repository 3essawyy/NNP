# Module: inference

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/inference/` (`predictor.py`, `cli.py`), root `infer.py`
- **Tasks:** T-009
- **ADRs:** ADR-004 (timing), ADR-008

## Responsibility
Implement the brief's §9 contract: `python infer.py <data_dir>` → `results.txt` and `time.txt` in the working directory.

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `infer.py DATA_DIR [--model PATH] [--out-dir DIR] [-q]` | default model `models/model.pkl` next to `infer.py`; default out = CWD; accepts the `data` folder or its parent |
| Provides | `infer_folder`, `run`, `write_outputs` | used by experiments (E6) and the HF app |
| Consumes | data (`list_sample_files`, `read_sample`), models (`load_bundle`, `HARPipeline`) | — |

## Internal design (order of work)
1. **Untimed I/O:** list `<int>.csv` numerically; read all rows. Unreadable file → all-NaN row (logged) so line *i* still ↔ `i.csv`.
2. **Timed one-off:** `pipeline.adapt(X)` + one warm-up `predict_one` → `T_adapt`.
3. **Timed loop:** for each sample, `perf_counter` right before `predict_one` (sample already in memory), right after the label. Charged time `t_i + T_adapt/N`.
4. Write `"{label}\n"` and `"{time:.3f}\n"` per sample.

`infer.py` prepends `src/` to `sys.path`, so it runs from a plain checkout or the zip without installing the package.

## Validation and failure modes
| Failure | Behavior |
|---|---|
| Missing model | `FileNotFoundError` naming `poe train` |
| Empty folder | empty output files |
| Non-numeric `.csv` name | `DataFormatError` (refuse to guess order) |

## Tests
`tests/test_inference.py` (CLI subprocess formats, numeric order via accuracy, amortized timing, unreadable file, parent folder, empty folder); `tests/test_dataset.py` on the real data when present; CI Docker job.

## Measured (validation, 3090 files)
96.7 % accuracy; adaptation ≈ 0.2–0.4 s; ≈ 0.4–0.5 ms charged per sample; whole run ≈ 4–6 s wall time.

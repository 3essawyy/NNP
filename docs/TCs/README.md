# Acceptance Test Cases

These prove the system meets the brief. Definitions: [acceptance.md](acceptance.md). Template for long TCs: [_TEMPLATE.md](_TEMPLATE.md). Update **Status** with date + commit whenever a TC is run.

## Traceability matrix

| Requirement | TCs | Tasks |
|---|---|---|
| REQ-S1 report sections | TC-012 | T-012 |
| REQ-S2 code zip, readme, requirements, infer.py | TC-002, TC-009, TC-013 | T-002, T-009, T-013 |
| REQ-S3 model file loaded automatically | TC-009 | T-009 |
| REQ-S4 Dockerfile | TC-010 | T-010 |
| REQ-S5 HF Space (bonus) | TC-011 | T-011 |
| REQ-I1…I4 inference contract | TC-004, TC-009 | T-003, T-009 |
| REQ-M1, M5, M7 preprocessing | TC-005 | T-005 |
| REQ-M2 feature selection | TC-006 | T-006 |
| REQ-M3 model selection / CV | TC-007 | T-007 |
| REQ-M4 error analysis | TC-008 | T-008 |
| REQ-G1 accuracy + runtime | TC-014 | T-007, T-009 |
| REQ-G2 docs / explainability | TC-001 | T-001 |

## Index

| ID | Title | Type | Status |
|---|---|---|---|
| TC-001 | Every brief requirement traced to a task and a design doc | Manual | Pass (2026-11-04) |
| TC-002 | Fresh clone: `uv sync` + `uv run poe check` green | Automated (CI) | Pass (2026-11-04, CI) |
| TC-003 | Import contracts hold; requirements.txt in sync | Automated (`tests/test_architecture.py`) | Pass (2026-11-04) |
| TC-004 | Numeric ordering + parser edge cases | Automated (`tests/test_data_io.py`) | Pass (2026-11-04) |
| TC-005 | Affine calibration cancelled; devices discovered; no false devices | Automated (`tests/test_preprocessing.py`) | Pass (2026-11-04) |
| TC-006 | Noise features removed by train-only selector | Automated (`tests/test_features.py`) | Pass (2026-11-04) |
| TC-007 | Adaptation beats global scaler; batch = single; fast SVM exact; every family trains | Automated (`tests/test_pipeline.py`) | Pass (2026-11-04) |
| TC-008 | Metrics, shifts, score CLI | Automated (`tests/test_evaluation.py`) | Pass (2026-11-04) |
| TC-009 | `infer.py` writes brief-format files in CWD, numeric order, amortized timing, robust to bad files | Automated (`tests/test_inference.py`) | Pass (2026-11-04) |
| TC-010 | Docker image builds and runs offline; outputs correct | Automated (CI docker job) + Manual | Pass (2026-11-04, manual in session; CI on PR) |
| TC-011 | HF Space serves predictions | Manual | Partial: app tested end-to-end locally via `gradio_client` (80 files → 2 devices); public deploy TODO(team) |
| TC-012 | Report contains every brief section; numbers generated | Manual | Partial: compiles (6 pages, no undefined refs); team sections + course template TODO |
| TC-013 | Submission zip contents | Manual | Pass (2026-11-04, `NNP_TEAM=TEST`: 179 files, 4.1 MB; unzipped `infer.py` in a plain-pip venv → 97.31 %); re-run with the real team number |
| TC-014 | Shipped model ≥ 95 % on validation through `infer.py` | Automated (`tests/test_dataset.py`, dataset only) + `poe infer && poe score` | Pass (2026-11-04: 97.31 %) |

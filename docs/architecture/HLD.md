# High-Level Design — NNP (Robust HAR)

Status: **v1.0** (2026-11-04, sessions S001–S002). Satisfies REQ-M1…M4 and REQ-S2…S4. Numbers quoted here come from `experiments/results/*.json` (regenerate with `uv run poe experiments`).

## 1. Purpose and scope
Classify one smartphone-telemetry record (561 features) into one of six activities, **on a phone the model has never seen**. The released data makes the shift concrete (see [EXP-001](../experiments/EXP-001-shift-diagnosis.md)):

* training = one device (`Device_B`, 3604 records; `Device_A` is empty),
* validation = 3090 records from **two** devices: one with Device_B's calibration, one with a ≈ −10 offset on features 0–250 — and both shifted enough that a standard pipeline (global scaler + RBF SVM) scores **~55 %**,
* a block of ~200 features (indices ≈ 40–239) whose class signal is buried in heavy noise.

The design goal is therefore *transfer*, not in-domain accuracy: every choice is judged on held-out-device behaviour.

## 2. Pipeline

```mermaid
flowchart LR
  subgraph Train["Training (labelled source devices) — nnp.models.train"]
    D1["Device_* folders"] --> S1["per-device<br/>standardize (own stats)"]
    S1 --> C1["clip ±8σ"]
    C1 --> F1["ANOVA-F selection<br/>(fit on train)"]
    F1 --> M1["RBF SVM<br/>(C=10, γ=scale)"]
    D1 --> P1["DeviceClusterer.fit<br/>(probe features = lowest F)"]
  end
  subgraph Infer["Inference (unlabeled test folder) — infer.py"]
    T0["1.csv … N.csv<br/>(numeric order, I/O untimed)"] --> A1["adapt(): cluster test set<br/>into devices (probe PCA + k-means,<br/>silhouette ≥ 0.8, size ≥ 5 %)"]
    A1 --> A2["per discovered device:<br/>mean/std on selected features"]
    A2 --> L["per-sample loop (timed):<br/>assign → standardize → clip → FastRBFSVC"]
    L --> R["results.txt / time.txt"]
  end
  M1 -. model.pkl .-> L
  P1 -. model.pkl .-> A1
  F1 -. support .-> A2
```

| Stage | Module | Brief requirement | Key decision |
|---|---|---|---|
| Read / order samples | [data](../modules/data/README.md) | REQ-I1 | [ADR-003](../decisions/ADR-003-data-usage-policy.md) |
| Per-device standardization, clipping | [preprocessing](../modules/preprocessing/README.md) | REQ-M1, M5, M7 | [ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md) |
| Device discovery in the test set | [preprocessing](../modules/preprocessing/README.md) | REQ-M5, M6 | [ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md) |
| Feature selection / reduction | [features](../modules/features/README.md) | REQ-M2, M6 | [ADR-006](../decisions/ADR-006-feature-selection.md) |
| Model zoo, CV protocols, tuning, training | [models](../modules/models/README.md) | REQ-M3 | [ADR-005](../decisions/ADR-005-validation-protocol.md), [ADR-007](../decisions/ADR-007-final-model.md) |
| Metrics, error analysis, synthetic shifts | [evaluation](../modules/evaluation/README.md) | REQ-M4 | [ADR-005](../decisions/ADR-005-validation-protocol.md) |
| Folder → results.txt/time.txt | [inference](../modules/inference/README.md) | REQ-I1…I4, S2, S3 | [ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md) (timing), [ADR-008](../decisions/ADR-008-model-artifact.md) |
| Experiments → report numbers/figures | [experiments](../modules/experiments/README.md) | REQ-S1, M8 | — |
| Docker, HF Space, submission zip | [deploy](../modules/deploy/README.md) | REQ-S2, S4, S5 | [ADR-009](../decisions/ADR-009-docker-image.md), [ADR-010](../decisions/ADR-010-hf-space.md) |
| LaTeX report | [report](../modules/report/README.md) | REQ-S1 | [ADR-012](../decisions/ADR-012-report-in-latex.md) |

## 3. Why this shape (the argument in four lines)
1. **Calibration offsets/gains are per-feature affine maps** → standardizing each device with its own statistics cancels them exactly ([ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md)); the classic "fit scaler on train, apply to test" cannot.
2. **We don't know which test record came from which phone** → discover devices without labels, on features that do not depend on the activity (otherwise we would cluster activities).
3. **Noisy features break distance-based models across devices** → drop them with a train-only ANOVA filter ([ADR-006](../decisions/ADR-006-feature-selection.md)).
4. **Runtime is graded** → RBF SVM with a numpy one-vs-one decision (~0.3 ms/sample), adaptation cost amortized into every `time.txt` line.

## 4. Data flow and artefacts

| Artefact | Producer | Consumer | In git? |
|---|---|---|---|
| `data/radar_project_data/…` | `uv run poe data` | train, experiments, dataset tests | **No** (ADR-003) |
| `models/model.pkl` | `uv run poe train` | `infer.py`, Docker, HF Space | Yes (ADR-008) |
| `experiments/results/*.json` | `uv run poe experiments` | report, docs | Yes |
| `report/figures/*`, `report/generated/*.tex` | `uv run poe experiments` | `report/main.tex` | Yes |
| `results.txt`, `time.txt` | `infer.py` | graders / `poe score` | No |

## 5. Dependency direction (enforced)
`experiments → inference → models → {features, preprocessing, evaluation} → data`. `inference` never imports `experiments`/`evaluation` (keeps the runtime image small). Checked by import-linter (`tests/test_architecture.py`).

## 6. Resource and runtime assumptions
| Item | Value (validation set, this container: 4 vCPU, 15 GB RAM) |
|---|---|
| Training time | ≈ 2 s |
| Model size | ≈ 3.4 MB |
| One-off adaptation (3090 samples) | ≈ 0.2–0.4 s, amortized (≈ 0.1 ms/sample) |
| Per-sample time charged in `time.txt` | ≈ 0.4–0.5 ms mean |
| Docker image | `python:3.12-slim` + numpy/pandas/scipy/scikit-learn ≈ 650 MB; runs with `--network none` |

## 7. Assumptions and limitations (also in the report)
* Device shift ≈ per-feature affine (offset + gain) + noise; strong non-linear distortions are not undone.
* Per-device statistics assume roughly similar class mixes per device (prior shift biases the means).
* Adaptation needs a batch: a test device with < 30 samples (or < 5 % of the set) falls back to training reference statistics.
* Validation labels guided model selection (ADR-003) → the validation score is optimistic for truly new devices; simulated-shift CV is the unbiased-ish companion number.

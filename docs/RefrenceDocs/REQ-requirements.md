# Requirements (extracted from the brief)

Source: [Pattern_Recognition_Project_Fall_2026.pdf](Pattern_Recognition_Project_Fall_2026.pdf) (Cairo University, CMP Pattern Recognition & Neural Networks, "Robust HAR System"). Paraphrased into traceable IDs. If this file and the PDF disagree, the PDF wins — fix this file.

## Problem (P)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-P1 | Classify each tabular record (561 numeric features from smartphone accelerometer + gyroscope; time- and frequency-domain aggregates) into one of 6 activities: 0 Walking, 1 Upstairs, 2 Downstairs, 3 Sitting, 4 Standing, 5 Laying. (The brief says "562 features": 561 features + the `target` column.) | 2 |
| REQ-P2 | Be robust to real-world variation and **multi-device sensor heterogeneity**; generalize to unseen hardware. | 1, 2.1 |

## Pipeline modules (M)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-M1 | Preprocessing and feature normalization module. | 2.1 |
| REQ-M2 | Feature selection / dimensionality reduction module. | 2.1 |
| REQ-M3 | Model selection, hyperparameter tuning and cross-validation training module. | 2.1 |
| REQ-M4 | Performance analysis and error evaluation module. | 2.1 |
| REQ-M5 | Preprocessing handles hardware calibration offsets and sensitivity variations across manufacturers. | 2.1 |
| REQ-M6 | Preprocessing handles non-stationary noise variance and feature drift. | 2.1 |
| REQ-M7 | Preprocessing includes outlier detection and robust scaling. | 2.1 |
| REQ-M8 | Literature survey on domain generalization and inertial telemetry informs the design. | 2.1 |

## Data (D)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-D1 | Training set = `Device_A` and `Device_B` folders. (Observed: `Device_A` is empty — ADR-003.) | 2.2 |
| REQ-D2 | A validation split for local CV and tuning. (Observed: `eval_data/` with labels — ADR-003.) | 2.2 |
| REQ-D3 | Hidden test set `\data` used for the leaderboard. | 2.2 |

## Deliverables (S)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-S1 | PDF report (course Overleaf template): Abstract, Introduction, System Overview (pipeline, models evaluated, architectural decisions), Experimental Setup (preprocessing, feature selection criteria, CV strategy, hyperparameters, metrics), Results (comparative + error analysis, confusion insights, failure modes), Conclusion (capabilities, limitations, future work), Workload Distribution. | 5.1 |
| REQ-S2 | `code_[team number].zip` with complete source, `readme.md` (setup, dependencies, execution), `requirements.txt`, and `infer.py` taking the evaluation data directory as argument and generating the prediction files. | 5.2 |
| REQ-S3 | Pre-trained model file (preferably `.pkl`) at the right relative path, loaded automatically. | 5.3 |
| REQ-S4 | A valid `Dockerfile` replicating the environment. | 5.4 |
| REQ-S5 | **Bonus:** deploy on a public cloud service (Hugging Face Spaces recommended); link in report and readme. | 5.5 |

## Inference contract (I)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-I1 | Input folder `data` with `1.csv … N.csv`, one sample each; process in **strictly increasing numerical order**. | 9.1 |
| REQ-I2 | Write `results.txt` in the working directory: one integer 0–5 per line, line *i* ↔ `i.csv`; no header, no indices, not all on one line. | 9.2.1 |
| REQ-I3 | Write `time.txt` in the working directory: one time in seconds per line, **3 decimals**, per sample, including preprocessing + inference; timer starts after the CSV is loaded and stops after the prediction (disk I/O excluded). Only numbers. | 9.2.2 |
| REQ-I4 | Files generated automatically when the script completes. | 9.2.2 |

## Grading and rules (G)

| ID | Requirement | Brief § |
|---|---|---|
| REQ-G1 | 70 %: hidden-test performance; ranking by accuracy (dominant) and running time. | 7, 8.5 |
| REQ-G2 | 30 %: report + oral discussion; all members must understand all phases (problem/design, preprocessing/features, model selection/training, evaluation/analysis, conclusion). | 7 |
| REQ-G3 | Any technique allowed; deep models ≤ 2 billion parameters running concurrently. | 8.2 |
| REQ-G4 | Team of 3–4; all attend the discussion; fair workload distribution. | 3, 6 |
| REQ-G5 | Academic integrity: no plagiarism/cheating. | 6.4 |
| REQ-G6 | Electronic submission via the course Drive link; late penalty applies. | 6 |

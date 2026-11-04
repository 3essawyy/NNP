# ADR-004: Unlabeled test-time adaptation (per-device standardization) and honest timing

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001 (policy), S002 (method)
- **Related:** REQ-M1, REQ-M5…M7, REQ-I3, T-005, T-009, EXP-001, EXP-003

## Context
A Device_B-trained pipeline with a global scaler scores ≈ 55–65 % on validation (EXP-002) although the same models reach ≈ 96 % in-domain. Diagnosis (EXP-001): per-feature offsets (≈ ±10 on features 0–250) and gains differ between devices — an affine distortion per feature. A scaler fitted on the source device cannot undo a target device's offset. Each test file is one record, but `infer.py` receives the whole folder.

## Decision
1. **Policy (user, S001):** `infer.py` may use the **unlabeled** features of the whole test folder (transductive adaptation). No labels, no information beyond the files given.
2. **Method:** standardize each *device* with its own statistics (mean/std on the selected features), both at training (per device folder) and at test time (per *discovered* device). Clip standardized values to ±8 (outlier control).
3. **Device discovery:** k-means (k = 1…4) on a 2-D PCA of **probe features** — the 25 % of features with the lowest training ANOVA F-score, i.e. activity-independent — robust-z-scored with training stats. Accept a split only if silhouette ≥ 0.8 **and** every cluster ≥ max(30, 5 % of N). Otherwise one device. Tiny clusters (< 30) fall back to training reference statistics.
4. **Timing (user, S001):** the one-off adaptation (clustering + statistics + one warm-up prediction) is timed and **amortized**: each `time.txt` line = that sample's own preprocess+predict time + adaptation time / N. Disk I/O stays untimed, as the brief specifies.

## Alternatives considered
| Option | Why not |
|---|---|
| Per-sample only (pure domain generalization) | One record can't reveal its device's offset; ceiling ≈ 65 % with any model (EXP-002) |
| Cluster on all features | Splits by activity (static vs dynamic are well separated), which per-cluster standardization would erase |
| Hard-code "files 1–1545 / 1546–3090" | Overfits the validation layout; the hidden test may mix devices differently |
| CORAL / covariance alignment | More parameters, unstable with few samples per device; first-order alignment already reaches ≈ 96–97 % — future work |
| Exclude adaptation from `time.txt` | Smaller numbers, but hides compute the pipeline needs — could be seen as gaming the timer |
| Add adaptation time to line 1 only | Honest but produces one outlier line; amortization states the same total |

## Consequences
- Accuracy ≈ 96.7 % validation vs ≈ 55 % without (EXP-003).
- Assumes similar class mixes per device; documented as a limitation.
- Behaviour with < 60 test files: one device, possibly reference stats → reduced accuracy on a shifted phone (documented in the report and the HF app).

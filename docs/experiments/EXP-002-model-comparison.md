# EXP-002: Model family comparison (E2)

- **Date:** 2026-11-04 (S002; final run with the shipped preprocessing)
- **Code:** `src/nnp/experiments/compare.py`; results `experiments/results/compare.json`; report `tab:compare`, `fig:compare`
- **Decision:** ADR-007

## Question
With the same preprocessing (per-device z-score, clip ±3, ANOVA F > 100), which classifier family transfers best to new devices, and what does each cost per sample?

## Setup
- Each family runs with its **default** hyperparameters (`zoo.DEFAULTS`; RBF SVM C = 10 here, the shipped C = 100 comes from EXP-004).
- 5-fold in-domain CV and simulated-shift CV on Device_B. The synthetic phone has a 3σ offset on 50 % of features, gain σ 0.2, noise 0.1σ and 1 % outliers.
- Validation with adaptation and with a global scaler.
- Per-sample latency over 300 `predict_one` calls on 4 vCPU.

## Result (accuracy %)

| Model | In-domain CV | Sim.-shift CV | Val. global scaler | Val. adapted | Per device (0 / 1) | ms/sample |
|---|---|---|---|---|---|---|
| RBF SVM | 96.3 | 95.9 | 75.9 | **96.7** | 96.4 / 97.0 | 0.51 |
| Vote (RBF+lin. SVM+LR) | 95.9 | 95.4 | 82.6 | 96.4 | 96.6 / 96.2 | 1.47 |
| Logistic regression | 95.9 | 95.1 | 87.7 | 96.2 | 96.6 / 95.9 | 0.30 |
| LightGBM | 95.6 | 94.1 | 60.0 | 96.1 | 94.9 / 97.3 | 1.43 |
| Linear SVM | 95.6 | 95.1 | 75.9 | 96.1 | 96.3 / 95.8 | 0.34 |
| Extra trees | 94.2 | 91.5 | 80.1 | 95.7 | 96.1 / 95.4 | 87.82 |
| LDA (shrinkage) | 94.5 | 93.5 | 58.5 | 95.0 | 94.8 / 95.1 | 0.29 |
| MLP (256) | 94.7 | 94.4 | 86.7 | 94.2 | 94.4 / 94.0 | 0.53 |
| kNN (k=10) | 91.6 | 91.4 | 85.4 | 93.0 | 93.7 / 92.4 | 1.26 |
| QDA (reg. 0.5) | 90.9 | 91.2 | 72.5 | 92.3 | 91.7 / 92.9 | 1.54 |
| Random forest | 92.6 | 89.6 | 79.6 | 91.6 | 93.6 / 89.6 | 87.53 |
| Gaussian NB | 68.2 | 47.1 | 51.5 | 63.0 | 70.4 / 55.7 | 0.65 |

## Interpretation
- **Adaptation helps every family:** from +7.5 pp (MLP) to +36 pp (LightGBM); the RBF SVM gains 21 pp.
- **Clipping at ±3 already blunts the shift for a global scaler.** It caps how far a 10-unit offset can move a standardized feature, so LR reaches 87.7 % with no adaptation, against 64 % at clip ±8 (S002 earlier run). Adaptation is still worth +7.5 to +36 pp.
- **RBF SVM is best** on validation, simulated-shift CV and in-domain CV, at about 0.5 ms/sample.
- **The vote ensemble does not help.** It scores 96.4 % against 96.7 % for the SVM alone, at 3× the latency.
- **Trees cost about 88 ms per single-row prediction** (300 trees, thread dispatch per call), about 170× the SVM. They also transfer worse.
- **Simulated-shift CV ranks families much like validation.** 4 of the top 5 agree, and Gaussian NB, QDA and kNN sit near the bottom in both. However, it cannot see the noise-block failure (EXP-003).

## Decision
Ship the RBF SVM; tune C, γ, F-threshold and clipping in EXP-004.

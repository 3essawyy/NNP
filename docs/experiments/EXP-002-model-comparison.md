# EXP-002: Model family comparison (E2)

- **Date:** 2026-11-04 (S002)
- **Code:** `src/nnp/experiments/compare.py`; results `experiments/results/compare.json`; report `tab:compare`, `fig:compare`
- **Decision:** ADR-007

## Question
With the same preprocessing (per-device z-score, clip ±8, ANOVA F > 50), which classifier family transfers best to new devices, and what does each cost per sample?

## Setup
5-fold in-domain CV and simulated-shift CV on Device_B (offset 3σ on 50 % of features, gain σ 0.2, noise 0.1σ, 1 % outliers); validation with adaptation and with a global scaler; per-sample latency over 300 `predict_one` calls (4 vCPU container).

## Result (accuracy %)

| Model | In-domain CV | Sim.-shift CV | Val. global scaler | Val. adapted | Per device (0 / 1) | ms/sample |
|---|---|---|---|---|---|---|
| RBF SVM | 96.1 | 95.6 | 55.4 | **96.7** | 96.4 / 97.0 | 0.34 |
| Vote (RBF+lin. SVM+LR) | 96.0 | 95.6 | 69.2 | 96.4 | 96.6 / 96.2 | 1.36 |
| Logistic regression | 95.9 | 95.2 | 64.2 | 96.2 | 96.6 / 95.9 | 0.33 |
| Linear SVM | 95.7 | 95.2 | 63.0 | 96.2 | 96.6 / 95.7 | 0.30 |
| MLP (256) | 94.6 | 94.5 | 72.9 | 96.1 | 96.3 / 95.9 | 0.45 |
| LightGBM | 96.1 | 94.0 | 58.9 | 95.9 | 94.9 / 97.0 | 1.34 |
| Extra trees | 94.2 | 92.1 | 79.9 | 95.9 | 96.4 / 95.3 | 73.86 |
| LDA (shrinkage) | 94.2 | 93.6 | 57.4 | 95.2 | 95.4 / 95.1 | 0.34 |
| Random forest | 92.8 | 90.2 | 79.9 | 94.3 | 95.0 / 93.7 | 70.56 |
| kNN (k=10) | 90.4 | 90.5 | 65.0 | 92.1 | 92.4 / 91.8 | 1.68 |
| QDA (reg. 0.5) | 89.8 | 90.0 | 60.4 | 91.0 | 90.9 / 91.0 | 1.62 |
| Gaussian NB | 68.8 | 47.5 | 41.7 | 64.1 | 70.2 / 58.1 | 0.47 |

## Interpretation
- **Adaptation matters more than the classifier.** Every family gains 14–41 pp from per-device adaptation. The best global-scaler models (trees, 79.9 %) are still more than 10 pp below every adapted model except Gaussian NB.
- **RBF SVM is best** on validation, simulated-shift CV (tied with the vote) and in-domain CV (tied with LightGBM), and is among the fastest.
- **The vote ensemble does not help.** It scores 96.4 % against 96.7 % for the SVM alone, at 4× the latency.
- **Trees are least hurt by the global scaler** but end lower with adaptation and cost ~70 ms per single-row prediction (300 trees, thread dispatch per call). That would multiply the runtime component ~200×.
- **Simulated-shift CV ranks the families almost like validation.** The top 5 are the same and Gaussian NB/QDA/kNN sit at the bottom in both. It is a reasonable label-free proxy.
- **Gaussian NB collapses under simulated shift** (47.5 %). Its independence assumption plus the heavy-tailed features makes per-class variances fragile.

## Decision
Ship the RBF SVM; tune C, γ and the F-threshold in EXP-004.

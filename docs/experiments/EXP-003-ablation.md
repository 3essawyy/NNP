# EXP-003: Ablation of the preprocessing (E3)

- **Date:** 2026-11-04 (S002)
- **Code:** `src/nnp/experiments/ablation.py`; results `experiments/results/ablation.json`; report `tab:ablation`, `fig:ablation`
- **Decisions:** ADR-004, ADR-006 (corrected after this run)

## Question
What does each preprocessing step contribute, with the shipped RBF SVM?

## Result (accuracy %)

| Variant | Val. device 0 | Val. device 1 | Val. overall | Sim.-shift CV |
|---|---|---|---|---|
| Shipped (per-device z, clip ±8, ANOVA F > 50) | 96.4 | 97.0 | **96.7** | 95.6 |
| No per-device adaptation (global scaler) | — | — | 55.4 | 13.7 |
| No feature selection (all 561) | 36.5 | 33.6 | 35.0 | **97.3** |
| No adaptation and no selection | — | — | 28.0 | 19.7 |
| No outlier clipping | 96.1 | 96.8 | 96.4 | 95.3 |
| Robust scaling (median/IQR) | 96.6 | 96.6 | 96.6 | 95.4 |
| Mutual-information top-336 | 75.1 | 73.8 | 74.4 | 95.4 |
| + PCA (95 % variance) | 95.4 | 95.1 | 95.3 | 94.5 |
| + LDA projection (5-D) | 74.4 | 92.8 | 83.6 | 18.6 |

## Interpretation
- **Both core steps are necessary.** Removing adaptation costs 41 pp and removing selection costs 62 pp. Without either, the model is at 28 %.
- **The two steps interact.** Adaptation alone (no selection) reaches only 35.0 % and selection alone (no adaptation) 55.4 %; only together 96.7 %. Offsets must be removed *and* the noise dimensions dropped: after z-scoring, the ~225 noise dimensions have the same unit variance as the informative ones, so they dominate RBF distances.
- **Clipping helps a little** (+0.3 pp). Robust and z-score scaling are equivalent here, since few records are extreme.
- **Mutual information fails** (74.4 %). Of its top 336 features, 53 come from the noise block (checked directly in S002), because MI picks up non-linear dependence that survives the noise. ANOVA's linear between/within ratio is the right criterion for *transferable* signal here. ADR-006 was corrected; it had claimed similar accuracy.
- **Reduction does not pay.** PCA costs 1.4 pp. The LDA projection is fragile: one device at 74 %, and synthetic shift breaks it (18.6 %).
- **Important limitation of simulated-shift CV.** It rates "no selection" at 97.3 %, the best of all variants, while validation gives 35 %. The synthetic phone model (offsets, gains, i.i.d.-like noise) does not reproduce whatever makes the noise block non-transferable across real devices. Simulated-shift CV is therefore evidence about calibration robustness only. The feature-selection decision rests on EXP-001 (train-only kNN/F diagnosis) and validation.

## Decision
Keep the shipped pipeline. Record the proxy's blind spot in ADR-005/006 and in the report's limitations.

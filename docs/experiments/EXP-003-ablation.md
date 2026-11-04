# EXP-003: Ablation of the preprocessing (E3)

- **Date:** 2026-11-04 (S002; final run with the shipped config)
- **Code:** `src/nnp/experiments/ablation.py`; results `experiments/results/ablation.json`; report `tab:ablation`, `fig:ablation`
- **Decisions:** ADR-004, ADR-005, ADR-006 (the last two were corrected after this experiment)

## Question
What does each preprocessing step contribute to the shipped RBF SVM (C = 100, F > 100, clip ±3)?

## Result (accuracy %)

| Variant | Val. device 0 | Val. device 1 | Val. overall | Sim.-shift CV |
|---|---|---|---|---|
| Shipped pipeline | 97.2 | 97.4 | **97.3** | 95.6 |
| No per-device adaptation (global scaler) | — | — | 75.9 | 18.4 |
| No feature selection (all 561) | 31.1 | 29.1 | 30.1 | **98.2** |
| No adaptation and no selection | — | — | 24.7 | 19.7 |
| No outlier clipping | 97.0 | 96.8 | 96.9 | 94.4 |
| Robust scaling (median/IQR) | 96.2 | 97.1 | 96.7 | 95.4 |
| Mutual-information top-315 | 81.4 | 80.6 | 81.0 | 96.2 |
| + PCA (95 % variance) | 95.1 | 96.1 | 95.6 | 93.8 |
| + LDA projection (5-D) | 95.6 | 96.2 | 95.9 | 58.2 |

"—": without adaptation the validation set is treated as one domain.

## Interpretation
- **Both core steps are necessary, and only their combination works.**
  - Adaptation alone, with all features, gives 30.1 %.
  - Selection alone, with a global scaler, gives 75.9 %.
  - Neither gives 24.7 %. That pipeline scores **99.1 % in in-domain 5-fold CV** (measured in S002), the clearest illustration of the problem.
- **Why the combination matters:** after per-device z-scoring, the ~246 noise-dominated dimensions have the same unit variance as the informative ones and dominate RBF distances. Removing them only helps once the offsets are gone.
- **Clipping (±3) adds 0.4 pp.** Robust median/IQR scaling is 0.6 pp lower here, but more outlier-proof (EXP-005).
- **Mutual information fails (81.0 %).** Its top 315 include 36 features that ANOVA drops, 32 of them from the noise block. MI detects non-linear dependence that survives heavy noise and doesn't transfer. ANOVA's between/within variance ratio picks transferable, linearly separable signal. ADR-006 was corrected; it had claimed similar accuracy.
- **Reduction doesn't pay.** PCA costs 1.7 pp. The LDA projection holds on validation (95.9 %) but breaks under synthetic shift (58.2 %).
- **Blind spot of simulated-shift CV.** It rates "no selection" at 98.2 %, the best of all variants, while validation gives 30.1 %. The synthetic phone (offsets, gains, i.i.d.-like noise) does not reproduce what makes the noise block non-transferable. Simulated-shift CV is evidence about calibration robustness only. The selection decision rests on the train-only diagnosis (EXP-001: kNN at chance on the block within Device_B) and on validation.

## Decision
Keep the shipped pipeline. Document the blind spot in ADR-005/006 and in the report.

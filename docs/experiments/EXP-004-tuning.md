# EXP-004: Hyperparameter tuning of the RBF SVM (E4)

- **Date:** 2026-11-04 (S002)
- **Code:** `src/nnp/experiments/tuning.py`; results `experiments/results/tuning.json`; report `fig:tuning`
- **Decision:** ADR-007

## Question
Which C, ANOVA threshold and γ should the shipped model use, and how sensitive is the result to them?

## Setup
- **Grid:** C ∈ {1, 3, 10, 30, 100} × F-threshold ∈ {10, 25, 50, 100, 200, 400}, γ = `scale`.
- **Metrics:** validation accuracy plus 3-fold simulated-shift CV.
- **γ sweep:** multipliers 0.25…4 of the `scale` value at the chosen cell.
- **Selection rule:** ADR-005.

## Result: validation accuracy (%) / simulated-shift CV (%)

| C \ F > | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| 1 | 90.8 / 93.9 | 95.3 / 93.8 | 95.3 / 93.3 | 94.9 / 93.3 | 94.5 / 92.1 | 94.4 / 92.2 |
| 3 | 89.3 / 94.8 | 96.2 / 94.9 | 96.4 / 94.6 | 96.1 / 94.5 | 95.4 / 93.1 | 95.1 / 93.9 |
| 10 | 87.3 / 95.1 | 96.4 / 95.1 | 96.7 / 95.1 | 96.5 / 95.2 | 96.4 / 94.0 | 95.9 / 94.7 |
| 30 | 87.0 / 95.3 | 96.7 / 95.1 | **97.1** / 95.1 | 96.9 / 95.3 | 96.6 / 94.6 | 96.6 / 94.9 |
| 100 | 87.0 / 95.3 | 96.6 / 95.0 | 96.8 / 95.0 | **97.1** / **95.5** | 96.9 / 94.2 | 96.6 / 94.6 |

**γ sweep at C = 100, F > 100:** ×0.25 → 96.6 · ×0.5 → 96.8 · **×1 → 97.1** · ×2 → 96.1 · ×4 → 93.5.

## Interpretation
- **F > 10 is too permissive.** It keeps part of the noise block, and validation drops to 87–91 %. Simulated-shift CV again fails to see this (it *rises* with C at F > 10), consistent with the EXP-003 blind spot.
- **Plateau.** For F ∈ [25, 400] and C ≥ 3, accuracy is 95.1–97.1 %. The optimum is broad, not a sharp validation artefact.
- **Larger C helps slightly.** Device-standardized features are clean enough for a harder margin.
- **γ = `scale`** (1 / (n_features · Var)) is already optimal.

## Decision
C = 100, F > 100, γ = `scale` (ADR-007): validation 97.06 %, verified through `infer.py`.

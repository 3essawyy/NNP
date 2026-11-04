# EXP-004: Hyperparameter tuning of the RBF SVM (E4)

- **Date:** 2026-11-04 (S002)
- **Code:** `src/nnp/experiments/tuning.py`; results `experiments/results/tuning.json`; report `fig:tuning`, `tab:clip`
- **Decision:** ADR-007

## Question
Which C, ANOVA threshold, γ and outlier clip should the shipped model use, and how sensitive is the result to them?

## Procedure (as it happened)
1. First pass at clip ±8: the C × F grid picked **C = 100, F > 100** (97.06 %).
2. The robustness study (EXP-005) then showed the adapted pipeline was fragile to outlier records, so a clip × scaling sweep was added at C = 100, F > 100. **Clip ±3** was chosen (97.31 %).
3. Final re-run of the whole stage at clip ±3: the C × F grid again picks C = 100, F > 100, and the clip sweep again picks ±3. The configuration is a fixed point of the procedure.

Selection rule (ADR-005): highest validation accuracy. Ties within 0.2 pp go to the higher simulated-shift CV, then to the simpler model.

## Result: C × F grid at clip ±3 — validation % / simulated-shift CV % (3-fold)

| C \ F > | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| 1 | 88.2 / 94.8 | 95.5 / 94.8 | 95.7 / 94.5 | 95.4 / 94.1 | 94.6 / 93.0 | 94.5 / 92.8 |
| 3 | 86.2 / 95.9 | 96.5 / 95.8 | 96.6 / 95.6 | 96.5 / 95.3 | 96.0 / 93.9 | 95.7 / 94.4 |
| 10 | 84.7 / 96.2 | 96.7 / 96.0 | 97.0 / 95.9 | 96.7 / 96.0 | 96.7 / 94.9 | 96.5 / 94.8 |
| 30 | 84.7 / 96.1 | 96.8 / 96.2 | 97.1 / 95.8 | 97.0 / 96.2 | 97.0 / 95.3 | 96.7 / 95.0 |
| 100 | 84.7 / 96.1 | 96.8 / 95.9 | 97.0 / 95.7 | **97.3 / 96.2** | 97.1 / 95.0 | 96.9 / 95.1 |

**γ sweep at C = 100, F > 100** (multiplier of `scale`): ×0.25 → 96.9 · ×0.5 → 97.0 · **×1 → 97.3** · ×2 → 96.9 · ×4 → 95.4.

**Clip × scaling at C = 100, F > 100:**

| Scaling | ±2 | ±3 | ±4 | ±5 | ±8 |
|---|---|---|---|---|---|
| z-score, validation | 97.38 | **97.31** | 97.25 | 97.25 | 97.06 |
| z-score, sim.-shift CV | 96.14 | **96.20** | 95.95 | 96.00 | 95.53 |
| robust, validation | 96.57 | 96.67 | 96.73 | 96.76 | 96.60 |
| robust, sim.-shift CV | 94.48 | 94.67 | 94.92 | 94.62 | 93.95 |

Rule: the best validation is ±2 (97.38). ±3, ±4 and ±5 fall within 0.2 pp, and ±3 has the best simulated-shift CV, so ±3 is chosen.

## Interpretation
- **F > 10 is too permissive.** It keeps part of the noise block, and validation drops to 85–88 %. Simulated-shift CV, blind to this (EXP-003), even prefers it.
- **Plateau.** For F ∈ [25, 400] and C ≥ 3, validation is 95.7–97.3 %. The optimum is broad, not a sharp artefact.
- **Larger C helps slightly.** Device-standardized features support a harder margin.
- **γ = `scale`** is already optimal.
- **z-score beats robust scaling on the real devices** by 0.5–0.8 pp at every clip, although robust is more outlier-proof (EXP-005).
- **Selection bias.** The shipped config beats the untuned default (C = 10, F > 50, clip ±8: 96.7 %) by 0.6 pp, about 19 of 3090 files. That is of the same order as the optimism that selecting on validation introduces, as stated in the report.

## Decision
C = 100, F > 100, γ = `scale`, z-score, clip ±3 (ADR-007): validation **97.31 %** through `infer.py`.

# ADR-006: Feature selection — train-only ANOVA F filter

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S002
- **Related:** REQ-M2, REQ-M6, T-006, EXP-001, EXP-003, EXP-004

## Context
EXP-001: features ≈ 40–239 have median training F-score ≈ 3–9 (others ≈ 300–1300) and kNN on those blocks is at chance **even within Device_B** (kurtosis is identical across the block — injected heavy noise). With them, kernel/distance models collapse under shift (RBF SVM ≈ 36 % with per-device scaling), while linear models survive (≈ 93 %).

## Decision
- After per-device standardization and clipping, keep features with **ANOVA F > 50** computed on the training data only (336 of 561 on the released data). Remove near-constant features.
- Threshold tuned jointly with C on validation (E4); the surface is flat over 25–200, so the choice is not fragile.
- Dimensionality reduction (PCA, LDA projection) and mutual-information selection are implemented and compared (E3) but not shipped.

## Alternatives considered
| Option | Why not |
|---|---|
| Keep all features | In-domain CV is higher (≈ 98 %) but transfer collapses for RBF/kNN models |
| Hard-code "drop 40–239" | Not principled; the F filter finds the same block automatically and keeps the informative ones inside it (e.g. gravity means 40–44) |
| Select by marginal shift (KS train vs test) | Tested in S002 exploration: removes informative frequency features, worse accuracy |
| Mutual information top-k | Similar accuracy, slower, randomness (E3) |
| PCA | Mixes noisy and clean features before the classifier; no gain (E3) |

## Consequences
- 40 % fewer features → faster per-sample inference.
- If a new dataset has no noisy block, the filter keeps almost everything (safe default).

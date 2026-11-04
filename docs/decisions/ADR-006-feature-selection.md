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
- Dimensionality reduction (PCA, LDA projection) and mutual-information selection are implemented and compared (EXP-003) but not shipped.
- **Caveat found in EXP-003:** simulated-shift CV does *not* detect the noise-block failure (97.3 % without selection vs 35.0 % on validation) — the choice to select rests on validation evidence and EXP-001's train-only kNN/F diagnosis, not on the synthetic proxy.

## Alternatives considered
| Option | Why not |
|---|---|
| Keep all features | In-domain CV is higher (≈ 98 %) but transfer collapses for RBF/kNN models |
| Hard-code "drop 40–239" | Not principled; the F filter finds the same block automatically and keeps the informative ones inside it (e.g. gravity means 40–44) |
| Select by marginal shift (KS train vs test) | Tested in S002 exploration: removes informative frequency features, worse accuracy |
| Mutual information top-336 | **74.4 %** validation vs 96.7 % (EXP-003): MI also scores the noisy block well (non-linear dependence survives the noise) — 53 of its 336 picks are noise-block features that ANOVA drops |
| PCA (95 % variance) after selection | 95.3 % vs 96.7 % (E3): discards low-variance discriminative directions |
| LDA projection (5-D) | 83.6 % (E3): fragile under shift (one device 74 %) |

## Consequences
- 40 % fewer features → faster per-sample inference.
- If a new dataset has no noisy block, the filter keeps almost everything (safe default).

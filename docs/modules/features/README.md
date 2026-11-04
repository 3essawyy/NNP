# Module: features

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/features/` (`selection.py`, `reduction.py`)
- **Tasks:** T-006
- **ADRs:** ADR-006

## Responsibility
Choose which standardized features reach the classifier (selection) and optionally project them (reduction). Fit on **training data only**.

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `FeatureSelector(SelectionConfig).fit(Z, y).transform(Z)`; `support_`, `scores_` | methods `anova` (threshold or top-k), `mutual_info` (top-k), `none` |
| Provides | `Reducer(ReductionConfig).fit/transform` | `none`, `pca`, `lda` |

## Internal design
ANOVA F = between-class variance / within-class variance per feature (Fisher criterion, univariate). The released noisy block has F ≈ 3–9 vs ≈ 300–16 000 elsewhere, so a threshold of 50 separates them with a wide margin (E1 figure `fscore`). F is computed after per-device standardization + clipping, so it is calibration-invariant.

## Config
| Key | Default | Meaning |
|---|---|---|
| `method` | `anova` | selection method |
| `threshold` | 50 | keep F > threshold (tuned in E4; flat 25–200) |
| `k` | None | top-k instead of threshold |
| `min_variance` | 1e-8 | always drop constant features |

## Tests
`tests/test_features.py` (noise block dropped on synthetic data; top-k; error when nothing survives; reducer shapes).

## Known limitations
Univariate: a feature useless alone but useful jointly would be dropped (not observed to matter: E3 MI selection gives the same accuracy).

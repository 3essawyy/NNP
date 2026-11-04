# Module: preprocessing

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/preprocessing/` (`domain.py`, `clustering.py`)
- **Tasks:** T-005
- **ADRs:** ADR-004

## Responsibility
Make features from different phones comparable: per-domain (device) standardization, outlier clipping, missing-value neutralization, and unsupervised discovery of which test records share a device. Does **not** select features or classify.

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `DomainStats.fit(X, scaling) / .transform(X, clip) / .subset(cols)` | `scaling ∈ {zscore, robust}`; non-finite → location (0 after transform); zero scale → 1 |
| Provides | `standardize_by_domain(X, domains)` | per-label stats |
| Provides | `DeviceClusterer(ClusterConfig).fit(X_train, y_train).partition(X_test) -> DeviceAssignment` | `labels`, `k`, `silhouette`, `assign(x)` for one row |
| Consumes | scikit-learn (`f_classif`, `PCA`, `KMeans`, `silhouette_score`) | — |

## Internal design
**Why per-device standardization works.** A calibration offset `b` and gain `a` per feature turn `x` into `a·x + b`. Mean/std of the transformed device are `a·μ + b` and `|a|·σ`, so `(a·x + b − (a·μ + b)) / (|a|·σ) = (x − μ)/σ` (for `a > 0`): the device distortion cancels exactly. Test `test_per_domain_standardization_cancels_affine_calibration` proves it numerically.

**Device discovery** (`DeviceClusterer`):
1. `fit`: ANOVA F per raw feature on training data; *probe features* = lowest 25 % F (activity-independent by construction); store their training median/IQR.
2. `partition(X_test)`: robust-z the probes with training stats → PCA(2) fit on the test set → k-means for k = 2…4.
3. Accept a k only if silhouette ≥ 0.8 **and** min cluster ≥ max(30, 5 % N). Best accepted silhouette wins; else k = 1.
4. `assign(x)`: same projection, nearest centroid (used per sample in the timed loop; identical to batch labels).

On validation: k = 2, silhouette 0.994, split exactly at file 1546 (ARI = 1 vs the observed offset split). On each half alone, or on training data: k = 1 (the ~39-record outlier groups are rejected by the size rule).

**Clipping:** standardized values clipped to ±3 (winsorizing; tuned in E4 — ±3 beat ±8 on validation, 97.3 vs 97.1 %, and on outlier robustness, EXP-005).

## Config (`ClusterConfig`, pipeline)
| Key | Default | Meaning |
|---|---|---|
| `scaling` | `zscore` | `robust` (median/IQR·0.741) compared in E3 |
| `clip` | 3.0 (shipped, `final.py`); dataclass default 8.0 | ±clip after standardization; `None` disables |
| `probe_quantile` | 0.25 | share of lowest-F features used for clustering |
| `k_max` | 4 | max devices considered |
| `min_silhouette` | 0.8 | acceptance threshold |
| `min_fraction` / `min_size` | 0.05 / 30 | minimum cluster size |

## Validation and failure modes
| Failure | Detection | Behavior |
|---|---|---|
| Activities mistaken for devices | probe features are activity-independent | tested: single device → k = 1 |
| Outlier group mistaken for device | size rule | tested |
| Too few test samples (< 60) | `n < 2·min_size` | one device |
| Device cluster < 30 | pipeline | training reference stats (`Adaptation.fallback`) |
| NaN / inf | `np.isfinite` | ignored in stats, mapped to 0 |

## Tests
`tests/test_preprocessing.py`; end-to-end in `tests/test_pipeline.py`.

## Known limitations
Assumes similar class mixes across devices; a device whose offset is tiny relative to probe spread is merged with another (then its offset is small anyway).

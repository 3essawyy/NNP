# PLAN-005: Per-device standardization + device discovery (for T-005)

- **Task:** [T-005](../tasks/T-005-preprocessing-adaptation.md)
- **Status:** Executed
- **Date:** 2026-11-04

## Approach
Evidence from EXP-001 says the device shift is a per-feature affine map. Standardizing each device with its own statistics removes it. The open problem is *which test records belong together*: cluster on features that do not depend on the activity.

## Steps
1. `DomainStats` (zscore/robust, NaN-safe, clip) + unit test proving affine invariance.
2. `DeviceClusterer`: probe features = lowest-25 % training F; robust-z with training stats; PCA(2) on test; k-means k = 2…4; accept by silhouette ≥ 0.8 + size rule.
3. Verify on real data: mixed validation → k = 2 exact split; each half alone → k = 1; training → k = 1.
4. Fallback to training reference stats for clusters < 30.

## Tests to add
| Test | Proves |
|---|---|
| `test_per_domain_standardization_cancels_affine_calibration` | the core math |
| `test_clusterer_finds_two_devices` | discovery + single-row `assign` consistency |
| `test_clusterer_does_not_split_one_device_by_activity_or_outliers` | no false devices |
| `test_tiny_cluster_falls_back_to_reference` | small-N behaviour |

## Risks / open questions
| Item | Mitigation / decision |
|---|---|
| Activities clustered instead of devices | probe features are activity-independent; tested |
| Outlier records form a "device" | min-size rule (S002 exploration: ~39 outliers gave silhouette 0.77 on one device) |

## Deviations (filled after execution)
None.

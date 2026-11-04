# T-005: Per-device standardization, clipping, device discovery

- **Status:** Done
- **Phase:** M1
- **Modules:** [preprocessing](../modules/preprocessing/README.md)
- **Requirements:** REQ-M1, REQ-M5, REQ-M6, REQ-M7
- **Depends on:** T-004
- **Plan:** [PLAN-005](../implementationPlans/PLAN-005-preprocessing-adaptation.md)
- **ADRs:** [ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md)
- **TCs:** TC-005
- **Experiments:** EXP-001, EXP-003

## Acceptance criteria
- [x] Affine calibration cancels exactly (unit test)
- [x] Validation devices discovered without labels (k = 2, silhouette ≈ 0.99, exact split)
- [x] Single device / outlier groups / tiny sets → k = 1
- [x] NaN/inf safe; outlier clipping

## Explain it (code-defense prep)
- **What:** the preprocessing that makes one phone's features look like another's.
- **Why:** offsets/gains are per-feature affine; standardizing *each device with its own* mean/std removes them (proof in the module doc). A global scaler fitted on Device_B leaves the −10 offset in place (≈ 55 % accuracy).
- **How:** training: stats per `Device_*` folder. Test: `DeviceClusterer` projects the activity-independent *probe* features (lowest ANOVA F) to 2-D PCA, k-means k = 2…4, accepts a split only if silhouette ≥ 0.8 and clusters ≥ 5 % / 30 samples; each discovered device gets its own stats. Clip at ±3σ (tuned).
- **Likely questions:** *Why not cluster on all features?* → you'd cluster activities (static vs moving) and erase them. *What if the test set has one sample?* → one device, training reference stats (limitation). *Is using test statistics cheating?* → no labels used; it is standard unsupervised test-time adaptation (AdaBN analogue); allowed by our policy ADR-004 and charged in `time.txt`. *What about outliers?* → outlier *records* contaminate the per-device mean/std; our robustness study (EXP-005) found this weakness, and clipping at ±3 (tuned) recovers most of it (76.6 → 85.3 % at 20 % outliers). Robust median/IQR stats are even more outlier-proof (91 %) but cost 0.5–0.8 pp on the real validation devices.

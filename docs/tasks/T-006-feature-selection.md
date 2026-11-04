# T-006: Feature selection / dimensionality reduction

- **Status:** Done
- **Phase:** M1
- **Modules:** [features](../modules/features/README.md)
- **Requirements:** REQ-M2, REQ-M6
- **Depends on:** T-004
- **ADRs:** [ADR-006](../decisions/ADR-006-feature-selection.md)
- **TCs:** TC-006
- **Experiments:** EXP-003, EXP-004

## Acceptance criteria
- [x] Train-only selector (ANOVA threshold / top-k, MI, none)
- [x] Reduction options (PCA, LDA) for comparison
- [x] Noise block removed on synthetic data (unit test) and on real data (336 kept)

## Explain it (code-defense prep)
- **What:** keep the 336 features whose class separability (ANOVA F) on Device_B exceeds 50.
- **Why:** the noisy block dominates Euclidean/RBF distances with information that does not transfer; removing it lifts the RBF SVM from ≈ 36 % to ≈ 97 % under per-device scaling.
- **How:** `f_classif` on standardized, clipped training features → threshold; threshold tuned with C (flat between 25 and 200).
- **Likely questions:** *Didn't you lose accuracy in-domain?* → yes, ≈ 98 → 96 %; we optimize transfer (ADR-005). *Why not PCA?* → 95.3 % vs 96.7 % (EXP-003): it keeps high-variance directions, not discriminative ones. *Why not mutual information?* → 74.4 %: MI still ranks 53 noise-block features in its top 336 (non-linear dependence survives heavy noise), and those break the RBF kernel across devices. *Can your synthetic-shift CV justify selection?* → no — without selection it reports 97.3 % while validation is 35.0 %; that is a limitation of the proxy (it models offsets/gains/noise, not the non-transferable feature block). *Did you use test labels to pick features?* → no; F is computed on training labels only; the threshold value was tuned on validation like any hyperparameter.

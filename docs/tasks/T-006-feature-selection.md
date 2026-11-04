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
- [x] Noise block removed on synthetic data (unit test) and on real data (315 kept at F > 100)

## Explain it (code-defense prep)
- **What:** keep the 315 features whose class separability (ANOVA F) on Device_B exceeds 100 (threshold tuned; plateau 25–400).
- **Why:** the noisy block dominates Euclidean/RBF distances with information that does not transfer; removing it lifts the RBF SVM from 30.1 % to 97.3 % under per-device scaling (EXP-003).
- **How:** `f_classif` on standardized, clipped training features → threshold; threshold tuned with C (flat between 25 and 200).
- **Likely questions:** *Didn't you lose accuracy in-domain?* → yes: the conventional pipeline (all features, global scaler) scores 99.1 % in-domain CV but 24.7 % on validation; ours ≈ 96 % in-domain and 97.3 % on validation. We optimize transfer (ADR-005). *Why not PCA?* → 95.6 % vs 97.3 % (EXP-003): it keeps high-variance directions, not discriminative ones. *Why not mutual information?* → 81.0 %: MI still ranks 32 noise-block features in its top 315 (non-linear dependence survives heavy noise), and those break the RBF kernel across devices. *Can your synthetic-shift CV justify selection?* → no — without selection it reports 98.2 % while validation is 30.1 %; that is a limitation of the proxy (it models offsets/gains/noise, not the non-transferable feature block). *Did you use test labels to pick features?* → no; F is computed on training labels only; the threshold value was tuned on validation like any hyperparameter.

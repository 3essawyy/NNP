# EXP-001: Shift diagnosis (Device_B vs validation)

- **Date:** 2026-11-04 (S001 exploration, formalized as E1 in S002)
- **Code:** `src/nnp/experiments/eda.py`; results `experiments/results/eda.json`
- **Decisions:** ADR-004, ADR-006

## Question
Why does a pipeline with ~98 % in-domain CV accuracy score ~55 % on validation?

## Observations
1. **Ranges.** 310 of 561 training features lie in [−1, 1] (the UCI HAR normalization); 251 don't, mostly shifted by ≈ +10 (e.g. feature 0: mean 10.28, std 0.05, identical across classes).
2. **Two validation devices.** Feature 1 is bimodal on validation (1544 values ≈ −0.7, 1544 ≈ 10.6; files 1–1545 vs 1546–3090).
3. **Affine per feature.** Fitting per-feature class-conditional means validation = a·train + b: files 1546–3090 have median slope ≈ 0.99 and intercept ≈ −10 on features 0–99; ≈ 0 intercept elsewhere. Files 1–1545: slope ≈ 1, intercept ≈ 0 on most features — yet a global-scaler SVM still scores 41 % there, so smaller per-feature gains/offsets matter too.
4. **Per-half standardization works.** Logistic regression trained on z-scored Device_B, applied to each validation half z-scored with its own stats: 93.4 % / 93.8 % (vs 64 % with the training scaler).
5. **But kernel/distance models still fail** (RBF SVM 36 %, kNN 49 %) → something else.
6. **Noise block.** kNN on 40-feature blocks *within* Device_B: blocks 40–239 at 17–25 % (chance 16.7 %), others 44–77 %. ANOVA F median ≈ 3–9 on 40–239 vs 300–1300 elsewhere; identical kurtosis (2.2) across the whole block → injected heavy noise. Logistic regression on that block alone: 72 % (weak linear signal remains).
7. **Remove it.** Dropping features with F ≤ 50 (336 kept): RBF SVM 97.0 % / 96.2 %, kNN 76 %, LDA 95 %.
8. **Device discovery.** k-means on 2-D PCA of low-F probe features: mixed validation → k = 2, silhouette 0.994, ARI 1.0 with the observed split; each half alone and training alone → k = 2 candidates only split off ~39–83 outlier records (rejected by the 5 % size rule) → k = 1.
9. **Not a permutation.** After per-domain z-scoring, every validation feature's class-mean profile is closest to the *same* training feature (no axis swaps).

## Decision
- Per-device standardization with unsupervised device discovery (ADR-004).
- Train-only ANOVA-F feature selection (ADR-006).

## Caveats
- Diagnosis used validation labels (allowed: ADR-003). The shipped *methods* themselves (clustering, F-filter) never use validation labels.

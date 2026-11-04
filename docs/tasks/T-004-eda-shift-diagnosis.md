# T-004: EDA and shift diagnosis (E1)

- **Status:** Done
- **Phase:** M1
- **Modules:** [experiments](../modules/experiments/README.md)
- **Requirements:** REQ-M4, REQ-M5, REQ-M6, REQ-P2
- **Depends on:** T-003
- **Experiments:** [EXP-001](../experiments/EXP-001-shift-diagnosis.md)

## Acceptance criteria
- [x] Dataset facts recorded (sizes, classes, missing values, ranges)
- [x] Shift characterized (offsets, which features, two devices in validation)
- [x] Noise block identified with a train-only statistic
- [x] Figures + table generated for the report

## Explain it (code-defense prep)
- **What:** the analysis that decided the whole architecture.
- **Why:** a standard pipeline scored ~55 % on validation vs ~98 % in-domain; we needed to know *why* before choosing models.
- **How:** (1) feature ranges: 251 training features leave the UCI [−1, 1] range (offset ≈ +10); (2) per-feature affine fit of class means validation vs train: slope ≈ 1, intercept ≈ −10 on features 0–250 for files 1546–3090 → second device; (3) ANOVA F per feature: a block 40–239 with F ≈ 3–9; (4) kNN per 40-feature block: chance level on that block even within Device_B.
- **Likely questions:** *How do you know it's a device and not activities?* → the offset is identical for all classes (class-mean intercepts), and the split falls exactly at file 1546. *Is the noise block useless?* → not entirely (logistic regression on it alone ≈ 72 % in-domain) but it hurts distance/kernel models under shift.

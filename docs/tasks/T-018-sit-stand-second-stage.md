# T-018: Sitting/Standing second stage (optional)

- **Status:** Todo (optional; only after T-016/T-017)
- **Modules:** [models](../modules/models/README.md)
- **Requirements:** REQ-G1
- **Source:** S002 handoff risk R5; [EXP-006](../experiments/EXP-006-final.md)

## Goal
72 of the 83 validation errors are Sitting ↔ Standing. Test a second-stage binary classifier, trained only on classes 3/4 with its own ANOVA selection, applied when the main model predicts 3 or 4. Expected gain ≤ 1 pp; must not cost more than ~0.1 ms/sample.

## Plan
1. Experiment in a new stage. Report the 3/4 confusion before and after; check that 3/4 F-scores pick different features (gravity angle features 554–560 are the likely carriers).
2. Adopt only with a validation gain ≥ 0.3 pp *and* no simulated-shift CV loss (ADR).

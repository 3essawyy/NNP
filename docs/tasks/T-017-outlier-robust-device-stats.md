# T-017: Outlier-robust per-device statistics

- **Status:** Todo (next session)
- **Modules:** [preprocessing](../modules/preprocessing/README.md)
- **Requirements:** REQ-M7
- **Source:** S002 handoff risk R2; [EXP-005](../experiments/EXP-005-robustness.md)

## Goal
Close the gap at 20 % outlier records (adapted 85.3 % vs global scaler 95.4 %) without losing validation accuracy (z-score beats median/IQR by 0.5–0.8 pp).

## Plan
1. Add `scaling="trimmed"` to `DomainStats`: mean/std after discarding records whose median |probe z| exceeds a threshold (or the top/bottom q % per feature).
2. Extend E4's clip × scaling sweep and E5's outlier curve with `trimmed`.
3. Adopt only if validation ≥ 97.2 % and the outlier curve improves (ADR + EXP-008).

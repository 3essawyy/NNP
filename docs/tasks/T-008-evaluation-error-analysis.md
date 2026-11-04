# T-008: Metrics, error analysis, robustness curves (E5–E6)

- **Status:** Done
- **Phase:** M1
- **Modules:** [evaluation](../modules/evaluation/README.md), [experiments](../modules/experiments/README.md)
- **Requirements:** REQ-M4, REQ-S1 (Results)
- **Depends on:** T-007
- **TCs:** TC-008
- **Experiments:** EXP-005, EXP-006

## Acceptance criteria
- [x] Accuracy, macro-F1, per-class P/R/F1, confusion matrix, top confusions
- [x] Synthetic shift generator (offset, gain, noise, outliers) and robustness sweeps
- [x] Final evaluation through the real inference path (timing included)

## Explain it (code-defense prep)
- **What:** how we measure and explain errors.
- **Why:** the brief asks for error analysis and failure modes, not just a score.
- **How:** `evaluate()` → confusion matrix; E6 runs `nnp.inference.predictor.run` (the exact `infer.py` code). E5 applies controlled shifts to a held-out 20 % of Device_B and compares adapted vs global scaler.
- **Likely questions:** *Main failure mode?* → Sitting ↔ Standing (static postures, nearly identical inertial signatures — the classic UCI HAR confusion). *When does the method fail?* → non-affine distortions, strong noise (E5 noise curve), tiny device groups (fallback), class-mix differences between devices.

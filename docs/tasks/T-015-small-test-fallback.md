# T-015: Small test sets — fallback model and lower minimum sizes

- **Status:** Done
- **Phase:** M1
- **Modules:** [models](../modules/models/README.md), [preprocessing](../modules/preprocessing/README.md), [experiments](../modules/experiments/README.md)
- **Requirements:** REQ-I1, REQ-G1
- **Depends on:** T-009
- **ADRs:** [ADR-013](../decisions/ADR-013-small-test-set-fallback.md)
- **Experiments:** [EXP-007](../experiments/EXP-007-small-test-sets.md)
- **TCs:** TC-015
- **Session:** S003

## Goal
Keep accuracy high when `infer.py` receives few files, where per-device adaptation can't work.

## Acceptance criteria
- [x] `PipelineConfig.fallback_model` / `min_stats_size`; fallback used *only* where adaptation falls back (unit test)
- [x] Exact `FastLinear` predictor (unit test)
- [x] E7 experiment: N = 1…200, mixed and single-device draws, full validation unchanged
- [x] Shipped config updated; `infer.py` on an 8-file folder → 7/8 correct

## Explain it (code-defense prep)
- **What:** if a test folder is too small to estimate a device's statistics, the record is standardized with the training statistics and classified by a logistic regression trained for exactly that situation.
- **Why LR and not the SVM?** Without adaptation, residual calibration offsets remain. The RBF kernel is very sensitive to them (75.9 % per file); a linear model degrades more gracefully (87.7 %). The ±3 clipping also limits how far an offset can push a feature.
- **Why lower minimum sizes?** A 40-file folder from two phones couldn't be split with a 30-per-cluster minimum, so one set of statistics was computed over both phones (≈ 81 %). With a minimum of 5 it splits (≈ 93 %). The 5 %-of-N rule keeps large sets unchanged.
- **Likely questions:** *Does this change your main score?* → no, 97.31 % is asserted unchanged. *What is the weakest case now?* → ≈ 10 files from two phones (EXP-007).

# T-007: Model zoo, pipeline, CV protocols, comparison and tuning (E2–E4)

- **Status:** Done
- **Phase:** M1
- **Modules:** [models](../modules/models/README.md), [experiments](../modules/experiments/README.md)
- **Requirements:** REQ-M3, REQ-G1
- **Depends on:** T-005, T-006
- **Plan:** [PLAN-007](../implementationPlans/PLAN-007-models-cv-tuning.md)
- **ADRs:** [ADR-005](../decisions/ADR-005-validation-protocol.md), [ADR-007](../decisions/ADR-007-final-model.md)
- **TCs:** TC-007
- **Experiments:** EXP-002, EXP-003, EXP-004

## Acceptance criteria
- [x] 12 model families behind one interface
- [x] In-domain CV, simulated-shift CV, validation (per device)
- [x] Comparison, ablation and C × F grid + γ sweep generated as tables/figures
- [x] Shipped config recorded in `models/final.py` + ADR-007

## Notes / log
- 2026-11-04: first full run stalled on random forest — `fit` verified the generic fallback predictor on 500 single rows (pointless; it *is* `predict`). Fixed: only re-implementations are verified. RF fit 0.9 s after fix.
- QDA (sklearn 1.9) needs > n_features samples per class with `solver="svd"`; switched to the shrinkage solver.

## Explain it (code-defense prep)
- **What:** the model-selection machinery and its result (RBF SVM, C/F from E4).
- **Why:** we need evidence that the chosen model transfers, not only fits Device_B.
- **How:** `HARPipeline` wraps everything so each protocol (`in_domain_cv`, `shifted_cv`, `validate`) evaluates the *whole* pipeline, adaptation included, for any `PipelineConfig`.
- **Likely questions:** *Why RBF SVM over LightGBM/MLP?* → highest validation and simulated-shift accuracy, ~0.3 ms/sample; trees are poor at extrapolating across offsets and slower per sample. *How did you tune without overfitting validation?* → small grid, flat optimum, tie-break on simulated-shift CV, report all three estimates. *Why no deep learning?* → features are already hand-crafted; deep nets help on raw signals (Wang 2019); small MLP did not beat the SVM.

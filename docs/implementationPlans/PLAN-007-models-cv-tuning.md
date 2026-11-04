# PLAN-007: Model zoo, CV protocols, tuning (for T-007)

- **Task:** [T-007](../tasks/T-007-models-cv-tuning.md)
- **Status:** Executed
- **Date:** 2026-11-04

## Approach
Wrap preprocessing + selection + classifier in `HARPipeline` so that every protocol evaluates the *whole* pipeline, adaptation included. Compare families under one preprocessing (E2), ablate preprocessing with the best family (E3), tune the best family (E4).

## Steps
1. `zoo.make_model` for 12 families; `PipelineConfig` dataclass.
2. `cv.in_domain_cv`, `cv.shifted_cv` (synthetic device per held-out fold), `cv.validate` (per discovered device).
3. Experiments E2–E4 writing JSON, figures, LaTeX.
4. Choose config by ADR-005 rule; record in `models/final.py` + ADR-007; retrain `models/model.pkl`.

## Risks / open questions
| Item | Mitigation / decision |
|---|---|
| Overfitting the validation set while tuning | small grid; tie-break by simulated-shift CV; report all estimates |
| Slow families (RF, MLP, LightGBM) | acceptable for research; per-sample time reported |

## Deviations (filled after execution)
- Bug found by running E2: the fast-path verification ran for generic predictors too (RF: 500 parallel single-row predictions per fit). Fixed.
- QDA `reg_param=0.5` kept (shrinkage solver variants scored 73–84 % on validation vs 91 %); unit test uses top-40 features so QDA has more samples than features.

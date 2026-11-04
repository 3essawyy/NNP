# ADR-007: Final model — RBF SVM (C = 100, γ = scale) on ANOVA-selected (F > 100), per-device-standardized features

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S002
- **Related:** REQ-M3, REQ-G1, T-007, EXP-002, EXP-003, EXP-004; config in `src/nnp/models/final.py`

## Context
- **EXP-002:** with identical preprocessing, the RBF SVM is the best family on validation (96.7 %) and simulated-shift CV, and among the fastest per sample.
- **EXP-004:** tuning used the C × F-threshold grid, then a γ sweep, with the ADR-005 rule: highest validation; ties within 0.2 pp go to the higher simulated-shift CV, then the simpler model.

## Decision
```python
PipelineConfig(scaling="zscore", clip=8.0, adapt=True,
               selection=SelectionConfig("anova", threshold=100.0),
               model="svc_rbf", model_params={"C": 100.0, "gamma": "scale"})
```
- **Grid result:** the best validation cell is C = 30, F > 50 (97.12 %). C = 100, F > 100 (97.06 %) lies within 0.2 pp and has a higher simulated-shift CV (95.53 % vs 95.14 %), so the rule selects it.
- **γ:** the default `scale` multiplier is the peak of the sweep (×0.25…×4: 96.6, 96.8, **97.1**, 96.1, 93.5 %).
- **Size:** 313 features; model file 2.8 MB; about 0.46 ms charged per sample on the validation run.

## Alternatives considered
| Option | Why not |
|---|---|
| C = 10, F > 50 (pre-tuning default) | 96.7 % vs 97.1 %; same speed |
| C = 30, F > 50 (single best validation cell) | +0.06 pp is within noise (about 2 files); lower simulated-shift CV |
| Vote ensemble (RBF + linear SVM + LR) | 96.4 % at 4× latency (EXP-002) |
| LightGBM / MLP | 95.9 / 96.1 %; LightGBM 4× slower per sample |
| Logistic regression | 96.2 %, similar speed; kept as the documented runner-up (more interpretable) |

## Consequences
- The grid is a broad plateau (96–97 % for F ∈ [25, 400], C ≥ 3), so the choice is not fragile. The +0.36 pp over the default is about 11 validation files, roughly the size of the selection bias, as stated in the report.
- Any change to `final.py` requires `poe train`, `poe experiments` and an update to this ADR.

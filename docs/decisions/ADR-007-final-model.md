# ADR-007: Final model — RBF SVM (C = 100, γ = scale) on ANOVA-selected (F > 100), per-device-standardized, ±3-clipped features

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S002
- **Related:** REQ-M3, REQ-G1, T-007, EXP-002, EXP-003, EXP-004; config in `src/nnp/models/final.py`

## Context
- **EXP-002:** with identical preprocessing, the RBF SVM is the best family on validation (96.7 %) and simulated-shift CV, and among the fastest per sample.
- **EXP-004:** tuning used the C × F-threshold grid, then a γ sweep, with the ADR-005 rule: highest validation; ties within 0.2 pp go to the higher simulated-shift CV, then the simpler model.

## Decision
```python
PipelineConfig(scaling="zscore", clip=3.0, adapt=True,
               selection=SelectionConfig("anova", threshold=100.0),
               model="svc_rbf", model_params={"C": 100.0, "gamma": "scale"})
```
- **Grid result:** the best validation cell is C = 30, F > 50 (97.12 %). C = 100, F > 100 (97.06 %) lies within 0.2 pp and has a higher simulated-shift CV (95.53 % vs 95.14 %), so the rule selects it.
- **γ:** the default `scale` multiplier is the peak of the sweep (×0.25…×4: 96.6, 96.8, **97.1**, 96.1, 93.5 %).
- **Clipping (amendment, same session):** E5 showed the adapted pipeline is *more* fragile than a global scaler to outlier records (they contaminate per-device mean/std: 76.6 % vs 83.8 % at 20 % outliers). A clip × scaling sweep at C = 100, F > 100 (now part of E4): z-score ±2 97.38 %, **±3 97.31 % (best sim.-shift CV 96.20 %)**, ±4/±5 97.25 %, ±8 97.06 %; robust median/IQR scaling 96.6–96.8 % at every clip. Rule → **z-score, clip ±3**. Outlier robustness at 20 % outliers: 76.6 % → 85.3 %.
- **Size:** 315 features; model file 2.75 MB; ≈ 0.49 ms charged per sample; **97.31 %** validation through `infer.py` (macro-F1 97.42 %).

## Alternatives considered
| Option | Why not |
|---|---|
| C = 10, F > 50, clip ±8 (pre-tuning default) | 96.7 % vs 97.3 %; same speed |
| Robust (median/IQR) scaling | more outlier-robust (91 % at 20 % outliers) but 0.5–0.8 pp lower on validation at every clip |
| C = 30, F > 50 (single best validation cell) | +0.06 pp is within noise (about 2 files); lower simulated-shift CV |
| Vote ensemble (RBF + linear SVM + LR) | 96.4 % at 4× latency (EXP-002) |
| LightGBM / MLP | 95.9 / 96.1 %; LightGBM 4× slower per sample |
| Logistic regression | 96.2 %, similar speed; kept as the documented runner-up (more interpretable) |

## Consequences
- The grid is a broad plateau (95–97 % for F ∈ [25, 400], C ≥ 3), so the choice is not fragile. The +0.6 pp over the default is ≈ 19 validation files — of the order of the selection bias; stated in the report.
- C/F were tuned at clip ±8, then clip was tuned at the chosen C/F (coordinate-wise, not a joint grid).
- Any change to `final.py` requires `poe train`, `poe experiments` and an update to this ADR.

# EXP-006: Shipped model through the real infer.py path (E6)

- **Date:** 2026-11-04 (S002)
- **Code:** `src/nnp/experiments/final.py` (calls `nnp.inference.predictor.run`); results `experiments/results/final.json`; report `tab:per_class`, `fig:confusion`, `fig:timing`
- **Cross-checks:** `python infer.py …` + `poe score` (97.31 %), Docker run (TC-010), `tests/test_dataset.py`

## Result
- **Accuracy 97.31 %**, macro-F1 97.42 %. Discovered devices: 97.2 % (files 1–1545) and 97.4 % (1546–3090).
- **Timing** (charged per sample, amortized adaptation included): mean 0.49 ms, p95 0.62 ms, max 4.8 ms. Adaptation took 0.28 s for 3090 files; the timed total is 1.50 s (1.25 s after rounding each line to 3 decimals).

| Activity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 Walking | 99.2 | 99.2 | 99.2 | 526 |
| 1 Walking Upstairs | 99.6 | 99.0 | 99.3 | 482 |
| 2 Walking Downstairs | 98.8 | 99.5 | 99.1 | 402 |
| 3 Sitting | 91.9 | 94.4 | 93.1 | 517 |
| 4 Standing | 94.9 | 92.6 | 93.7 | 582 |
| 5 Laying | 100.0 | 100.0 | 100.0 | 581 |

Confusion matrix (rows = true): `[[522,2,2,0,0,0],[2,477,3,0,0,0],[2,0,400,0,0,0],[0,0,0,488,29,0],[0,0,0,43,539,0],[0,0,0,0,0,581]]`

## Error analysis
- **83 errors in total.** 72 of them (87 %) are Sitting ↔ Standing (29 Sitting → Standing, 43 Standing → Sitting). Both are static upright postures with nearly identical acceleration statistics; only a subtle trunk/thigh angle difference reaches the phone. This is the classic UCI-HAR confusion.
- **Zero dynamic ↔ static confusions**; the other 11 errors are among the three walking classes.
- **Laying is perfect.** Its gravity orientation is unique.
- **The failure mode is shared across devices** (97.2 vs 97.4 %), so it is a feature-set limitation, not a device effect.

## Rounding note
At 3 decimals, sub-0.5 ms samples are written as `0.000`. This is faithful to the brief's rounding rule. The unrounded total is in the logs and the report.

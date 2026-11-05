# EXP-007: Small test folders (E7)

- **Date:** 2026-11-05 (S003)
- **Code:** `src/nnp/experiments/smalln.py`; results `experiments/results/smalln.json`; report `fig:smalln`
- **Decision:** ADR-013

## Question
How accurate is `infer.py` when the test folder has few files, and how do we fix it?

## Setup
- Random subsets of the validation set with N ∈ {1, 5, 10, 20, 30, 45, 60, 100, 200} files.
- **Mixed** draws come from both devices; **single** draws come from one device.
- 100 draws per N below 100, 20 draws otherwise. Seed 0.
- The full-validation accuracy of every variant is asserted unchanged (97.31 %).

## Result (accuracy %)

| N | 1 | 5 | 10 | 20 | 30 | 45 | 60 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| **Mixed**, S002 (no fallback, min 30) | 77.0 | 76.4 | 74.0 | 75.4 | 80.5 | 81.9 | 84.3 | 96.2 | 96.7 |
| Mixed, + LR fallback, min 30 | 91.0 | 87.2 | 87.3 | 87.8 | 80.5 | 81.9 | 84.3 | 96.2 | 96.7 |
| **Mixed, shipped (fallback, min 5/12)** | **91.0** | **87.2** | **87.3** | 87.5 | **91.4** | **94.1** | **94.7** | 96.2 | 96.7 |
| **Single**, S002 | 72.0 | 74.8 | 73.6 | 77.4 | 94.6 | 95.6 | 96.0 | 96.7 | 97.1 |
| Single, + LR fallback, min 30 | 88.0 | 88.2 | 85.3 | 89.0 | 94.6 | 95.6 | 96.0 | 96.7 | 97.1 |
| **Single, shipped** | **88.0** | **88.2** | 85.3 | **93.4** | 94.6 | 95.6 | 96.0 | 96.7 | 97.1 |

Supporting sweep, `min_stats_size` with clusters from 5 records (S003, 100 draws, N = 5…45):

| min stats | 10 | 12 | 15 | 20 |
|---|---|---|---|---|
| Mixed, N = 10 | 78.0 | **87.0** | 87.0 | 87.0 |
| Single, N = 10 | 87.8 | 85.6 | 85.6 | 85.6 |
| Mixed, N = 30 | 91.7 | **91.8** | 91.2 | 88.1 |

## Interpretation
- **Fallback (N < 30).** With S002 settings, small folders used training statistics with the RBF SVM: about 75 %. Logistic regression on the same features scores about 88 %, because a linear model tolerates residual calibration offsets far better than the RBF kernel.
- **Lower minimums (N 20–60, mixed).** With a 30-per-cluster minimum a mixed folder of 30–60 files could not be split, so one set of statistics was computed over two phones (80–84 %). Splitting from 5 records per cluster recovers 91–95 %.
- **Statistics minimum = 12, not 10.** At 10, a mixed 10-file folder (unsplittable) used its own mixed statistics (78 %). At 12 it uses the fallback (87 %). The cost is −2.2 pp on single-device 10-file folders.
- **N ≥ 100 and the full set are unchanged**: the 5 %-of-N rule dominates the minimum there.
- **Weakest remaining band:** 10–20 files (85–88 %).

## Decision
Ship clusters ≥ 5, statistics ≥ 12, LR fallback (ADR-013).

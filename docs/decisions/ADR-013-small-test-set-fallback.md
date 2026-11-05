# ADR-013: Small test sets — lower split/statistics minimums (5 / 12) + logistic-regression fallback

- **Status:** Accepted
- **Date:** 2026-11-05
- **Session:** S003
- **Related:** REQ-I1, REQ-G1, T-015, EXP-007; amends ADR-004 (minimum sizes)

## Context
Adaptation (ADR-004) needs a batch. Under the S002 settings:
- a device split needed ≥ 30 records per cluster;
- a device's own statistics needed ≥ 30 records;
- smaller groups fell back to the **training** statistics with the main RBF SVM, which scores only **75.9 %** per file on the validation devices.

Measured on random validation subsets (EXP-007):
- **mixed-device folders of 30–60 files:** 81–83 %, because they could not be split and one set of statistics was computed over two phones;
- **folders under 30 files:** 69–77 %.

The hidden-test size is unknown (staff question).

## Decision
1. Clusters may be as small as **5 records** (`ClusterConfig.min_size = 5`). The **5 % of N** rule still applies, so large sets behave exactly as before.
2. A discovered device uses its own statistics from **12 records** (`min_stats_size = 12`). Sweep 10/12/15/20 (S003): 12 lifts mixed 10-file folders from 78.0 to 87.0 % for −2.2 pp on single-device 10-file folders; other sizes within noise.
3. Records whose device falls back to training statistics are classified by a **fallback model**: logistic regression trained on training-reference-standardized features (`fallback_model = "logreg"`). Per file, with training statistics, it scores **87.7 %** vs 75.9 % for the SVM (EXP-002's "global scaler" column).
4. The fallback has an exact numpy fast path (`FastLinear`).

## Alternatives considered
| Option | Why not |
|---|---|
| Keep S002 settings | 69–83 % for N < 100 |
| Fallback only (min sizes 30) | fixes N < 30 (85–92 %) but not mixed sets of 30–60 (81–83 %) |
| Fallback + min 10/10 or 15/15 | worse at N = 10–20 for mixed sets (EXP-007 exploration) |
| Fallback + min 5/10 | mixed 10-file folders 78.0 % (one set of stats over two phones) |
| Fallback + min 5/15 or 5/20 | no better than 5/12 at N = 10; 5/20 loses 3.7 pp at mixed N = 30 |
| MLP or kNN as fallback | 86.7 / 85.4 % per file, below LR's 87.7 %, and slower |
| Per-sample offset correction (no batch) | future work (T-017 / S002 R3); needs its own experiment |

## Consequences
- Full validation is unchanged: **97.31 %** (asserted in EXP-007 for every variant).
- **Remaining weak cells:** single-device folders of 10–11 files use the fallback (≈ 86 %) instead of their own statistics; 12–20-file mixed folders are about 87–88 % (EXP-007).
- The model bundle now contains two classifiers (2.76 MB).

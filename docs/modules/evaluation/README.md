# Module: evaluation

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/evaluation/` (`metrics.py`, `shift.py`, `score.py`)
- **Tasks:** T-008
- **ADRs:** ADR-005

## Responsibility
Measure and explain performance: accuracy, macro-F1, per-class P/R/F1, confusion matrix and top confusions; generate synthetic device shifts for robustness testing; score a `results.txt`/`time.txt` pair.

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `evaluate(y_true, y_pred) -> Evaluation` | rows of `confusion` = true class |
| Provides | `apply_shift(X, ShiftSpec, rng)` | offset (×σ, on a random fraction of features), log-normal gain, per-record heteroscedastic noise, outlier records |
| Provides | `python -m nnp.evaluation.score results.txt ground_truth.txt [time.txt]` / `poe score` | JSON summary; exit 1 on length mismatch |

## Internal design
`ShiftSpec` defaults model a "new phone": offsets 3σ on 50 % of features, gain σ 0.2. The robustness experiment (E5) sweeps each factor separately.

## Tests
`tests/test_evaluation.py`.

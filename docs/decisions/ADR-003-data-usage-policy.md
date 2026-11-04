# ADR-003: Data usage policy (training, validation, missing Device_A)

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-D1…D3, REQ-G5, T-003, T-004

## Context
Observed in the released archive (not assumed):
- `student_realease/Device_A/` is **empty**; `Device_B/train_data.csv` has 3604 labelled records.
- `eval_data/data/1.csv … 3090.csv` comes **with labels** (`eval_data/ground_truth.txt`). Its folder is named `data`, like the hidden test folder.
- Files 1–1545 look like Device_B's calibration; files 1546–3090 have a ≈ −10 offset on features 0–250 (a second device).

The brief calls for a "validation set: reserved split for local cross-validation and hyperparameter tuning". Training on the eval labels could be seen as leakage if the hidden test overlaps them.

## Decision (user, S001)
1. **Train on `Device_*` folders only.** The pipeline handles any number of device folders; empty ones are skipped with a warning.
2. **Validation labels are used for model selection and tuning only** — never fitted on. Every report table states this, and simulated-shift CV (ADR-005) is reported alongside as a selection-free estimate.
3. **Device_A missing:** proceed as single-source domain generalization; log an open item for the team to ask course staff. If Device_A data appears, re-run `poe train` + `poe experiments` (no code change; leave-one-device-out CV becomes possible).
4. **The dataset is never committed** (`data/` gitignored). `uv run poe data` downloads it from the brief's link.

## Alternatives considered
| Option | Why not |
|---|---|
| Also train on eval labels | Highest score if hidden test = eval, but an academic-integrity risk (REQ-G5) and no robustness gain for genuinely new devices |
| Strict holdout (never look at eval labels until the end) | We would tune blind to the real shift; the brief explicitly offers a validation set for tuning |
| Wait for staff about Device_A | Blocks all work; the design does not depend on the answer |

## Consequences
- Validation accuracy is mildly optimistic (selection bias) — stated in the report.
- Open item for the team: ask staff (a) whether Device_A data is coming, (b) whether the hidden test overlaps `eval_data`.

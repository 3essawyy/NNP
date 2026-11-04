# ADR-005: Validation protocol (three estimates side by side)

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S002
- **Related:** REQ-M3, REQ-M4, REQ-D2, T-007, T-008

## Context
Only one labelled training device exists; there are no subject IDs, so subject-wise CV is impossible. In-domain CV says nothing about device shift, and the validation set is also used for selection (ADR-003).

## Decision
Every candidate is scored three ways (`nnp.models.cv`):
1. **In-domain CV** — stratified 5-fold on Device_B (upper bound).
2. **Simulated-shift CV** — same folds; each held-out fold is turned into a synthetic new phone (`nnp.evaluation.shift`: offsets 3σ on 50 % of features, log-gain σ = 0.2, per-record noise 0.1σ, 1 % outlier records) before prediction. No validation labels involved.
3. **Validation** — released eval_data, overall and per discovered device.

Model/hyperparameter selection rule: highest validation accuracy; ties within 0.2 pp → higher simulated-shift CV → simpler model.

## Alternatives considered
| Option | Why not |
|---|---|
| In-domain CV only | Picked the wrong preprocessing in early tests (selection hurts in-domain CV but wins under shift) |
| Validation only | Single dataset → selection bias; no robustness evidence beyond the two validation devices |
| Leave-one-device-out | Needs ≥ 2 labelled devices (Device_A empty) — enabled automatically if it arrives |

## Consequences
- Reports show all three numbers; disagreements are discussed.
- **Known blind spot (EXP-003):** simulated-shift CV rates the pipeline *without* feature selection at 97.3 % while validation gives 35.0 % — the synthetic phone (offsets, gains, noise) doesn't reproduce why the noise block fails to transfer. It is evidence about calibration robustness only; feature decisions rely on EXP-001 and validation.

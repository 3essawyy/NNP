# T-009: infer.py — ordering, formats, amortized timing, fast SVM path

- **Status:** Done
- **Phase:** M1
- **Modules:** [inference](../modules/inference/README.md), [models](../modules/models/README.md)
- **Requirements:** REQ-I1…I4, REQ-S2, REQ-S3, REQ-G1
- **Depends on:** T-007
- **ADRs:** [ADR-004](../decisions/ADR-004-unlabeled-test-time-adaptation.md), [ADR-008](../decisions/ADR-008-model-artifact.md)
- **TCs:** TC-009

## Acceptance criteria
- [x] `python infer.py <dir>` writes `results.txt` / `time.txt` in CWD, formats per §9
- [x] Numeric order guaranteed and tested
- [x] Timer around per-sample compute only; adaptation amortized
- [x] Exact numpy RBF-SVM decision (~0.3 ms vs ~1 ms)
- [x] Validation: 97.3 % (shipped config, EXP-006), ≈ 0.5 ms charged per sample

## Notes / log
- 2026-11-04: per-sample 1.07 ms with `SVC.predict` → 0.52 ms with `FastRBFSVC` (incl. amortized adaptation); results byte-identical.

## Explain it (code-defense prep)
- **What:** the graded entry point.
- **Why:** 70 % of the grade flows through this script (accuracy + time).
- **How:** read all files (untimed) → `adapt` once (timed, divided by N) → loop: `perf_counter` → `predict_one` → `perf_counter` → write lines with `:.3f`.
- **Likely questions:** *Where exactly is the timer?* → `nnp/inference/predictor.py::run`, around `pipeline.predict_one`. *Isn't reading all files first against the rules?* → the brief excludes I/O from the timer; we load, then time compute; the folder-level adaptation compute is charged to every line. *Why is time 0.000 for some lines?* → sub-0.5 ms compute rounds to 0.000 at 3 decimals; totals are still reported in the logs/report. *What does FastRBFSVC do?* → same math as libsvm's one-vs-one vote, in numpy, verified identical.

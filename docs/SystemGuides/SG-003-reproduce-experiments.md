# SG-003: Reproduce every report number and retrain

- **Audience:** developer | examiner
- **Last verified:** 2026-11-04 on Linux (4 vCPU): full suite ≈ 30–40 min

## Steps
1. Data + deps: [SG-001](SG-001-setup-and-run.md) steps 1–2.
2. Retrain the shipped model (config in `src/nnp/models/final.py`):
   ```bash
   uv run poe train
   ```
3. Run all experiments (E1–E6):
   ```bash
   uv run poe experiments
   ```
   Subsets: `uv run nnp-experiments --only eda final`; smoke run: `--quick`.
4. Rebuild the report (`report/`, see [report module](../modules/report/README.md)).

## Verify it worked
`git diff experiments/results` shows only timing fields changing (accuracies are deterministic: fixed seeds, deterministic solvers).

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| compare stage very slow | tree models' per-row predict | expected (minutes); use `--quick` |
| numbers differ slightly | different BLAS/CPU | timings only; accuracies should match |

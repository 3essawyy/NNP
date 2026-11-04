# Module: experiments

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/experiments/` (`eda`, `compare`, `ablation`, `tuning`, `robustness`, `final`, `common`, `run`)
- **Tasks:** T-004, T-007, T-008
- **ADRs:** ADR-005, ADR-007

## Responsibility
Produce every number, table and figure of the report, reproducibly: `uv run poe experiments [--only …] [--quick]`.

| Stage | Question | Outputs |
|---|---|---|
| `eda` (E1) | What is in the data, what shifts, which features are noise? | `eda.json`; `fscore`, `block_knn`, `device_clusters`, `class_balance` figures; `dataset` table |
| `compare` (E2) | Which model family transfers best under one preprocessing? | `compare.json`; `compare` figure + table |
| `ablation` (E3) | What does each preprocessing step buy? | `ablation.json`; figure + table |
| `tuning` (E4) | C × F-threshold grid, γ sweep | `tuning.json`; heatmap |
| `robustness` (E5) | Accuracy vs controlled offset / gain / noise / outliers, adapted vs global scaler | `robustness.json`; 4 curves |
| `final` (E6) | Shipped model through the real `infer.py` path | `final.json`; confusion, timing histogram, per-class table |

Outputs: `experiments/results/*.json`, `report/figures/*.{pdf,png}`, `report/generated/*.tex` (+ `macros.tex`). The import-linter contract keeps this module out of the inference image.

## Interpretation log
Conclusions are recorded as `EXP-NNN` in [docs/experiments](../../experiments/README.md).

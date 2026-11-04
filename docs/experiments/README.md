# Experiment records

One record per question that changed (or confirmed) a design decision. Raw outputs: `experiments/results/*.json` (written by `uv run poe experiments`); figures/tables: `report/`.

| ID | Question | Stage | Outcome | Decision |
|---|---|---|---|---|
| [EXP-001](EXP-001-shift-diagnosis.md) | What is the shift between Device_B and the validation devices? | E1 + S001/S002 exploration | per-feature affine offsets; 2 devices; noise block 40–239 | ADR-004, ADR-006 |
| [EXP-002](EXP-002-model-comparison.md) | Which model family transfers best? | E2 | see record | ADR-007 |
| [EXP-003](EXP-003-ablation.md) | What does each preprocessing step buy? | E3 | see record | ADR-004, ADR-006 |
| [EXP-004](EXP-004-tuning.md) | Best C / F-threshold / γ? | E4 | see record | ADR-007 |
| [EXP-005](EXP-005-robustness.md) | How does accuracy degrade under controlled shifts? | E5 | see record | limitations |
| [EXP-006](EXP-006-final.md) | Shipped model through infer.py: accuracy, errors, timing | E6 | see record | — |

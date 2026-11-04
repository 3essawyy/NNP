# Modules (low-level design)

One folder per module; `README.md` inside is its LLD (template: [_TEMPLATE.md](_TEMPLATE.md)). The LLD describes the module as built on `main`; update it in the same PR as the code (Definition of Done).

| Module | Responsibility | Code | Owner | Status | Tasks |
|---|---|---|---|---|---|
| [data](data/README.md) | Read device folders, sample folders (numeric order), labels; download dataset | `src/nnp/data/` | TBD | Implemented | T-003 |
| [preprocessing](preprocessing/README.md) | Per-device standardization, clipping, NaN handling, device discovery | `src/nnp/preprocessing/` | TBD | Implemented | T-005 |
| [features](features/README.md) | ANOVA/MI selection, PCA/LDA reduction | `src/nnp/features/` | TBD | Implemented | T-006 |
| [models](models/README.md) | Model zoo, pipeline, CV protocols, tuning, training, bundle, fast predictor | `src/nnp/models/` | TBD | Implemented | T-007, T-009 |
| [evaluation](evaluation/README.md) | Metrics, confusion/error analysis, synthetic device shifts, `score` CLI | `src/nnp/evaluation/` | TBD | Implemented | T-008 |
| [inference](inference/README.md) | Folder → `results.txt` + `time.txt` (`infer.py`) | `src/nnp/inference/`, `infer.py` | TBD | Implemented | T-009 |
| [experiments](experiments/README.md) | E1–E6 runs → JSON, figures, LaTeX tables/macros | `src/nnp/experiments/` | TBD | Implemented | T-004, T-007, T-008 |
| [deploy](deploy/README.md) | Dockerfile, HF Space, submission zip | `Dockerfile`, `app/`, `scripts/` | TBD | Implemented (HF deploy pending) | T-010, T-011, T-013 |
| [report](report/README.md) | LaTeX report wired to generated numbers | `report/` | TBD | Draft | T-012, T-014 |

> **Owners (TODO(team)):** assign one member per group and replace `TBD`. Suggested grouping for 4 members, each covering one oral-exam phase (REQ-G2):
> ① **Problem & system design** — data + inference + deploy ·
> ② **Preprocessing & feature engineering** — preprocessing + features ·
> ③ **Model selection & training** — models ·
> ④ **Evaluation & analysis** — evaluation + experiments + report.
>
> Team number: `TEAM_NN` · Members: Member 1–4 (fill in here, in `report/main.tex`, and `NNP_TEAM` for `poe package`).

## Dependency direction
`experiments → inference → models → {features, preprocessing, evaluation} → data`. Lower layers never import upper layers; `inference` never imports `experiments` or `evaluation`. Enforced by import-linter contracts in `pyproject.toml`, checked by `tests/test_architecture.py`.

# ADR-002: Tech stack and tooling

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-S2, REQ-S4, REQ-G1, REQ-G3, T-002

## Context
The brief recommends Python (scikit-learn, XGBoost, PyTorch, NumPy, pandas). Ranking = accuracy first, then running time; the grading machine is unknown, so assume CPU-only. Deliverables need `requirements.txt` and a Dockerfile.

## Decision
| Concern | Choice |
|---|---|
| Language | Python **3.12** (`.python-version`; `requires-python >=3.12,<3.14`) |
| Env / lock | **uv** (`uv.lock`); `requirements.txt` is **exported** from the lock (`uv run poe requirements`), CI fails if stale |
| Runtime deps | numpy, pandas, scipy, scikit-learn, joblib only (what `infer.py` needs) |
| Research extras | `lightgbm`, `matplotlib`, `py7zr` (`uv sync --extra research`); Gradio under `--extra app` |
| Compute | **CPU only**; no deep-learning framework in the runtime image |
| Quality gates | ruff (lint+format), mypy `--strict`, pytest, import-linter contracts; `uv run poe check` = CI |
| Task runner | poethepoet (`uv run poe <task>`), works on Windows/macOS/Linux |
| CI | GitHub Actions: checks job + Docker build-and-run job |
| Layout | single package `src/nnp/` with one sub-package per module; `infer.py` at repo root (brief) |

## Alternatives considered
| Option | Why not |
|---|---|
| PyTorch MLP / 1D-CNN as main model | Features are already hand-crafted aggregates; small MLP scored below the RBF SVM in E2 and costs a 700 MB+ dependency; kept as a scikit-learn MLP baseline |
| XGBoost | LightGBM covers gradient boosting; both lose to linear/kernel models under shift (E2) |
| pip + hand-written requirements.txt | Unpinned transitive deps → "works on my machine" in the Docker build |

## Consequences
- The Docker image installs exactly the pinned runtime set.
- Adding a runtime dependency = `uv add` + `uv run poe requirements` + commit both.

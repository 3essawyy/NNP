# SG-002: Developer workflow

- **Audience:** developer
- **Last verified:** 2026-11-04 on Linux

Process rules are in [CLAUDE.md](../../CLAUDE.md); tooling rationale in [ADR-002](../decisions/ADR-002-tech-stack-and-tooling.md).

## Repository layout
```
infer.py                 submission entry point (brief §9)
pyproject.toml, uv.lock  project, tools, poe tasks; requirements.txt is exported from uv.lock
src/nnp/                 one package, one sub-package per module
  data/ preprocessing/ features/ models/ evaluation/ inference/ experiments/
  config.py              constants + default paths
tests/                   pytest (synthetic data; dataset tests skip without data/)
models/model.pkl         shipped model (committed)
experiments/results/     JSON outputs of `poe experiments`
report/                  LaTeX; figures/ and generated/ are written by experiments
app/                     Gradio app (HF Space)
scripts/                 package_submission.py, build_space.py
docs/                    all documentation
```

## Everyday commands
| Goal | Command |
|---|---|
| Everything CI runs | `uv run poe check` |
| Auto-fix lint + format | `uv run poe fmt` |
| One test | `uv run pytest tests/test_inference.py -k order` |
| Dataset-backed tests | `uv run pytest -m dataset` |
| Module boundaries only | `uv run lint-imports` |
| Add a runtime dependency | `uv add <pkg>` then `uv run poe requirements`; commit `pyproject.toml`, `uv.lock`, `requirements.txt` |
| Add a research-only dependency | `uv add --optional research <pkg>` |

## The feedback loop (user rule)
After any change to preprocessing/features/models: `uv run poe check`, then `uv run poe train && uv run poe infer && uv run poe score` and compare with `experiments/results/final.json`. A drop is a finding — record it in the task log, don't paper over it.

## Module boundaries
`experiments → inference → models → {features, preprocessing, evaluation} → data`; inference must not import experiments/evaluation/matplotlib/lightgbm. A broken contract fails `tests/test_architecture.py`. Fix the design (or write an ADR), don't add ignores.

## CI
`.github/workflows/ci.yml`: **checks** (uv sync --locked, requirements freshness, ruff, mypy, pytest) and **docker** (build, generate 150 synthetic samples, run the container, validate both files).

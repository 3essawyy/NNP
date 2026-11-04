# PLAN-002: Repo skeleton, tooling, CI (for T-002)

- **Task:** [T-002](../tasks/T-002-repo-skeleton-ci.md)
- **Status:** Executed
- **Date:** 2026-11-04

## Steps
1. `pyproject.toml` (hatchling, `src/nnp`), Python 3.12 via uv, extras `research` and `app`, dev group.
2. Tool config: ruff, mypy strict, pytest, import-linter layers.
3. poe tasks: data, train, infer, score, experiments, requirements, app, package, lint, fmt, types, test, check.
4. CI: checks job + Docker job.

## Files to create/change
| Path | Change |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | new |
| `.github/workflows/ci.yml` | new |
| `src/nnp/**/__init__.py`, `src/nnp/config.py` | new |
| `tests/test_architecture.py` | import contracts + requirements sync |

## Risks / open questions
| Item | Mitigation / decision |
|---|---|
| sklearn/numpy untyped for mypy | `ignore_missing_imports` overrides |
| Dataset absent in CI | synthetic fixtures in `tests/conftest.py`; dataset tests skip |

## Deviations (filled after execution)
- import-linter needs `include_external_packages = true` for the "inference never imports matplotlib/lightgbm" contract.

# T-002: Repo skeleton, tooling, CI

- **Status:** Done
- **Phase:** M1
- **Modules:** all
- **Requirements:** REQ-S2, REQ-S4
- **Depends on:** T-001
- **Plan:** [PLAN-002](../implementationPlans/PLAN-002-repo-skeleton-ci.md)
- **ADRs:** [ADR-002](../decisions/ADR-002-tech-stack-and-tooling.md), [ADR-011](../decisions/ADR-011-git-flow.md)
- **TCs:** TC-002, TC-003
- **Guides:** [SG-001](../SystemGuides/SG-001-setup-and-run.md), [SG-002](../SystemGuides/SG-002-developer-workflow.md)

## Acceptance criteria
- [x] `uv sync` + `uv run poe check` work from a fresh clone
- [x] One package `src/nnp/` with a sub-package per module; import-linter contracts
- [x] `requirements.txt` exported from `uv.lock`; CI fails when stale
- [x] CI: ruff, mypy --strict, pytest; Docker build-and-run job

## Explain it (code-defense prep)
- **What:** project layout, dependency management, quality gates.
- **Why:** reproducibility is graded implicitly (staff must run our code). A lockfile + exported pins means the Docker image installs exactly what we tested. Strict typing/tests replace human code review because Claude writes most code.
- **How:** `pyproject.toml` holds deps, tool config and `poe` tasks; `uv.lock` pins; `poe requirements` exports runtime pins; CI runs `poe lint/types/test` and builds the Docker image, then runs it on synthetic data.
- **Likely questions:** *Why is requirements.txt generated?* → single source of truth (uv.lock), no drift. *What do the import contracts guarantee?* → inference can't pull research-only code (e.g. matplotlib) into the runtime image; lower layers never import upper ones.

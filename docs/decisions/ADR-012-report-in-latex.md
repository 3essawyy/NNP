# ADR-012: Report written in LaTeX in the repo, numbers generated

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-S1, T-012

## Decision
- `report/main.tex` + `report/sections/*.tex` + `report/refs.bib`, IEEE-conference-style placeholder layout until the team supplies the course Overleaf template (then only the preamble changes).
- All results enter via `report/generated/macros.tex` and generated tables/figures from `uv run poe experiments` — no hand-typed numbers.
- Team-specific parts (names, team number, workload table) are marked `TODO(team)`.
- Build: Overleaf (upload the `report/` folder) or `latexmk -pdf main.tex` locally.

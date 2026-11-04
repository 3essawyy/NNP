# ADR-001: Team workflow and documentation structure (mirrors GATP)

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-G2, REQ-G4, all tasks

## Context
Four team members must each be able to defend *every* phase in the oral discussion (REQ-G2), while Claude writes most of the code. The team already uses a docs system in the GATP repository and asked for the same hierarchy here, "plus anything needed".

## Decision
- Same structure and ID conventions as GATP: `CLAUDE.md` + `docs/{architecture, modules, decisions, tasks, implementationPlans, TCs, SystemGuides, RefrenceDocs, sessions}` with a `README.md` index in each and templates.
- **Added for ML work:** `docs/experiments/` (`EXP-NNN` records: question → setup → result → decision), and `report/` (LaTeX, numbers generated from experiment outputs).
- Modules are the pipeline stages: data, preprocessing, features, models, evaluation, inference, experiments, deploy, report (9 — same count as GATP).
- Every task has an **Explain it** section (what, why, how, likely examiner questions).
- Claude plans, implements, tests and self-merges PRs (no human review). User rule (S001): **test every step on real data and let results steer the next step.**

## Alternatives considered
| Option | Why not |
|---|---|
| README only | Doesn't prepare four people for an oral exam on every phase |
| Notebook-driven project | Hard to test, review and reproduce; numbers drift from the report |

## Consequences
- Doc upkeep is part of the Definition of Done.
- Report numbers are generated (`report/generated/*.tex`), so they can't go stale silently.

# T-001: Docs scaffold, REQs, HLD, ADRs, task breakdown

- **Status:** Done
- **Phase:** M1
- **Modules:** docs
- **Requirements:** REQ-G2, all REQs (extraction)
- **Depends on:** —
- **Plan:** [PLAN-001](../implementationPlans/PLAN-001-docs-scaffold.md)
- **TCs:** TC-001
- **Session:** [S001](../sessions/2026-11-04-S001-kickoff-design-interview.md)

## Goal
Give the team the same working system as GATP: requirements traceable to the brief, a design everyone can explain, decisions with reasons, tasks with exam prep.

## Acceptance criteria
- [x] Brief extracted into REQ IDs ([REQ-requirements](../RefrenceDocs/REQ-requirements.md))
- [x] HLD with pipeline diagram ([HLD](../architecture/HLD.md))
- [x] ADR-001…012 for every kickoff decision
- [x] Every docs folder has a README index + template

## Explain it (code-defense prep)
- **What:** the documentation system (CLAUDE.md + docs/ tree).
- **Why:** 30 % of the grade is the report + oral discussion where *every* member must explain every phase (REQ-G2). Writing the "why" next to each decision makes that possible.
- **How:** REQ → task → plan → code → TC; ADRs never deleted, only superseded; numbers in the report are generated, not typed.
- **Likely questions:** *Where is decision X justified?* → ADR index. *How do you know the report numbers match the code?* → `report/generated/` is written by `poe experiments`.

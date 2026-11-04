# T-012: LaTeX report wired to generated results

- **Status:** Draft — team info + course template pending
- **Phase:** M1
- **Modules:** [report](../modules/report/README.md)
- **Requirements:** REQ-S1, REQ-G2
- **Depends on:** T-008
- **ADRs:** [ADR-012](../decisions/ADR-012-report-in-latex.md)
- **TCs:** TC-012

## Acceptance criteria
- [x] All brief sections present (Abstract … Workload Distribution)
- [x] Every number from `report/generated/`; figures from `report/figures/`
- [ ] Team names, number, workload table (TODO(team))
- [ ] Moved into the course Overleaf template (TODO(team): share template)
- [ ] Citations verified (T-014)

## Explain it (code-defense prep)
- **How are numbers kept correct?** `uv run poe experiments` rewrites `report/generated/macros.tex` and tables; the report only references macros like `\FinalAcc`.

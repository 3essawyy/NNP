# SG-005: Submission checklist

- **Audience:** team
- **Last verified:** not yet run

1. [ ] Team number + member names in `docs/modules/README.md`, `report/main.tex`, `report/sections/workload.tex`.
2. [ ] `uv run poe check` green; CI green on `main`.
3. [ ] `uv run poe train && uv run poe experiments` (if anything changed); commit results.
4. [ ] Report: in the course Overleaf template; citations verified (T-014); PDF exported.
5. [ ] HF Space link in README + report (bonus, optional).
6. [ ] `NNP_TEAM=<n> uv run poe package` → `dist/code_<n>.zip`; run TC-013 on the unzipped copy.
7. [ ] Upload zip + PDF to the course Drive before the deadline.
8. [ ] `git tag submission && git push origin submission`.
9. [ ] Every member reads the Explain-it sections of all tasks before the discussion.

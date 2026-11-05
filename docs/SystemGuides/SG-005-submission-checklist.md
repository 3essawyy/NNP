# SG-005: Submission checklist

- **Audience:** team
- **Last verified:** not yet run

0. [ ] **Keep the GitHub repo private until grading ends.** The brief penalizes *all involved teams* in a plagiarism case, so a copied public repo hurts us too.
0. [ ] Staff questions sent and answers recorded ([staff-questions.md](../RefrenceDocs/staff-questions.md)).
1. [ ] Team number + member names in `docs/modules/README.md`, `report/main.tex`, `report/sections/workload.tex`, `README.md`.
2. [ ] `uv run poe check` green; CI green on `main`.
3. [ ] `uv run poe train && uv run poe experiments` (if anything changed); commit results.
4. [ ] Report: in the course Overleaf template; citations verified (T-014); the data-usage statement is visible in the abstract/setup; PDF exported. If the template has a page limit, cut in this order: robustness figure → clip table → small-N figure → class-balance table (the ablation and confusion matrix carry the argument).
4b. [ ] Timing check on the slowest team machine (T-019).
5. [ ] HF Space link in README + report (bonus, optional).
6. [ ] `NNP_TEAM=<n> uv run poe package` → `dist/code_<n>.zip`; run TC-013 on the unzipped copy.
7. [ ] Upload zip + PDF to the course Drive before the deadline.
8. [ ] `git tag submission && git push origin submission`.
9. [ ] Every member works through [SG-006](SG-006-oral-exam-study-guide.md) (T-020).

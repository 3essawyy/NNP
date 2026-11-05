# Session S003 — 2026-11-05 — Small-test fallback + handoff additions

- **Tasks touched:** T-015 (Done), T-016…T-020 (created)
- **Branches / PRs:** `claude/modest-allen-w7yhnf` → PR (squash-merged to `main`)

## Done
- **T-015 small test folders** (ADR-013, EXP-007):
  - LR fallback model for records whose device can't be adapted
  - exact `FastLinear` predictor
  - clusters may be as small as 5 records; a device uses its own statistics from 12 records
  - new experiment stage E7 (`smalln`)
  - result: N < 100 improves from 74–84 % to 85–95 %; full validation **unchanged at 97.31 %**
- **New tasks from the S002 handoff risks:**
  - T-016 Python-version compatibility (most important)
  - T-017 outlier-robust device stats
  - T-018 Sitting/Standing second stage (optional)
  - T-019 timing on a slow machine
  - T-020 oral-exam prep
- **New docs:** [SG-006](../SystemGuides/SG-006-oral-exam-study-guide.md) oral-exam study guide (reading list, five core questions, traps); [staff-questions.md](../RefrenceDocs/staff-questions.md) (7 questions with consequences); SG-005 additions (keep repo private, page-limit cut order, timing check).
- **Data-usage statement** added to the README and the report abstract.
- **Tests:** 59 passing (3 new: fallback routing, fallback pickle, FastLinear exactness).

## Feedback loop
| Check | Finding | Change |
|---|---|---|
| Small-N baseline | mixed 30–60 files: 81–84 %; < 30: 69–77 % | fallback + lower minimums |
| E7 with min stats 10 | mixed 10-file folders 78 % (statistics over two phones) | sweep → 12 (87 %) |
| Existing test asserted "non-SVM ⇒ generic predictor" | linear models now intentionally use `FastLinear` | assertion updated to the new exact expectation (not weakened) |
| Task-table edit | a blank line split the Markdown table | fixed |

## Open items / next steps
Unchanged from the [S002 handoff](2026-11-04-S002-end-to-end-build.md) §7–§9, now tracked as tasks:
1. **T-016** Python compatibility (first, next session).
2. **T-017** trimmed device statistics.
3. **Team:** names/number, Overleaf template, citations (T-014), staff questions, timing check (T-019), oral prep (T-020).
4. **T-013** submission.

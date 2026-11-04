# Working agreement for Claude on NNP

Read this first in every session. It is the operating manual for whoever (Claude or a human) continues the work. Mirrors the GATP repo's system, adapted to an ML project.

## Roles
- Team of 4 (names/team number: see [docs/modules/README.md](docs/modules/README.md) — placeholders until filled). Claude plans, implements, experiments and tests.
- No PR review. Claude opens and squash-merges its own PRs.
- Each module has a component owner (recorded in [docs/modules/README.md](docs/modules/README.md)). Owners prepare for the oral discussion through the task docs' **Explain it** sections.

## Session protocol
**Start of session**
1. Read the latest file in `docs/sessions/` (handoff + next steps).
2. Read `docs/tasks/README.md` (status = source of truth).
3. Read the task file, its PLAN, and the module docs it touches.

**During**
- **Test every step and let the result steer the next one** (user instruction, S001). After each change: run the relevant tests, and — whenever the dataset is available — re-run the affected experiment or `infer.py` on the validation folder and compare with the last recorded numbers in `experiments/results/`. Report regressions instead of hiding them.
- Cloud sessions push to their designated `claude/*` branch and squash-merge PRs from it (ADR-011); local work uses `<type>/T-NNN-short-slug` branches.
- Write or update `docs/implementationPlans/PLAN-NNN-*.md` before coding non-trivial tasks.
- Record any new decision as an ADR in `docs/decisions/`. Record every experiment run that changes a conclusion in `docs/experiments/`.

**End of session**
1. Update task status in `docs/tasks/README.md` and the task file.
2. Write `docs/sessions/YYYY-MM-DD-SNNN-slug.md` using the template (done, decisions, open items, next steps).
3. Push, open the PR, squash-merge to `main`.

## Definition of Done (every task)
- [ ] Code implemented; tests written and passing (`uv run poe check`, CI green)
- [ ] Verified on the real dataset when it is relevant (numbers recorded in `experiments/results/` or the task log)
- [ ] Module design doc (`docs/modules/<name>/README.md`) updated
- [ ] Task doc complete, including the **Explain it** section (what, why, how it works, likely examiner questions)
- [ ] Acceptance TCs in `docs/TCs/` updated (IDs + status)
- [ ] ADR added if a decision was made
- [ ] SystemGuides updated if setup or usage changed
- [ ] Session log written
- [ ] PR merged (squash) to `main`

## Conventions
- IDs: `T-NNN` task, `PLAN-NNN` plan (same number as its task), `ADR-NNN` decision, `TC-NNN` acceptance test case, `REQ-NN` requirement, `SG-NNN` system guide, `EXP-NNN` experiment record, `SNNN` session.
- Every docs folder has a `README.md` index. Add a row whenever you add a file.
- Cross-link by ID and relative link. Never duplicate content; link to it.
- Report numbers come from `report/generated/*.tex`, written by `uv run poe experiments`. Never type a result into the report by hand.
- Release: tag `submission` on `main` when the zip is handed in.
- Commit messages: `T-NNN: imperative summary` (a PR closing several tasks: `T-NNN..T-MMM: …`).

## Non-negotiable project rules (from the brief + kickoff decisions)
- **Validation labels (`eval_data/ground_truth.txt`) are never used for training** — only for model selection/tuning and reporting (ADR-003). Training uses the `Device_*` folders only.
- Test-time adaptation may use the **unlabeled** test features only (ADR-004); its cost is amortized into every line of `time.txt`, never hidden.
- The dataset is never committed (`data/` is gitignored). The model bundle `models/model.pkl` is committed (ADR-008).
- `infer.py` must keep the brief's contract (§9): numeric file order, one integer 0–5 per line in `results.txt`, one 3-decimal time per line in `time.txt`, both in the working directory. `tests/test_inference.py` guards this.
- Accuracy dominates the ranking, then runtime: don't trade accuracy for speed without an ADR.
- Never weaken a test or an assertion to make it pass; never tune on the hidden test set (we don't have it).
- No paid services or secrets in the repo.

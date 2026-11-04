# Tasks — source of truth for status

Update this table in the same PR that changes a task's status. Template: [_TEMPLATE.md](_TEMPLATE.md). Weeks are relative (W+0 = week of 2026-11-04); the submission deadline is **TODO(team)**.

Milestone: **M1 = submission** (report PDF, `code_<team>.zip`, model, Dockerfile, optional HF link) · **M2 = oral discussion**

| ID | Title | Modules | Depends on | Target | Status |
|---|---|---|---|---|---|
| [T-001](T-001-docs-scaffold.md) | Docs scaffold, REQs, HLD, ADRs, task breakdown | docs | — | W+0 | Done |
| [T-002](T-002-repo-skeleton-ci.md) | Repo skeleton, tooling, CI | all | T-001 | W+0 | Done |
| [T-003](T-003-data-module.md) | Data module: loaders, numeric ordering, download | data | T-002 | W+0 | Done |
| [T-004](T-004-eda-shift-diagnosis.md) | EDA + shift diagnosis (E1) | experiments | T-003 | W+0 | Done |
| [T-005](T-005-preprocessing-adaptation.md) | Per-device standardization, clipping, device discovery | preprocessing | T-004 | W+0 | Done |
| [T-006](T-006-feature-selection.md) | Feature selection / reduction | features | T-004 | W+0 | Done |
| [T-007](T-007-models-cv-tuning.md) | Model zoo, pipeline, CV protocols, comparison + tuning (E2–E4) | models, experiments | T-005, T-006 | W+0 | Done |
| [T-008](T-008-evaluation-error-analysis.md) | Metrics, error analysis, robustness curves (E5–E6) | evaluation, experiments | T-007 | W+0 | Done |
| [T-009](T-009-inference.md) | `infer.py`: ordering, formats, amortized timing, fast SVM path | inference, models | T-007 | W+0 | Done |
| [T-010](T-010-docker.md) | Dockerfile + CI build-and-run | deploy | T-009 | W+0 | Done |
| [T-011](T-011-hf-space.md) | Bonus: Gradio app + HF Space bundle | deploy | T-009 | W+0 | Done (deploy by team: Todo) |
| [T-012](T-012-report.md) | LaTeX report wired to generated results | report | T-008 | W+0 | Draft — team info + template pending |
| [T-013](T-013-submission-pack.md) | Submission zip, README, final checks, tag | deploy, docs | T-010, T-012 | before deadline | Todo (script done) |
| [T-014](T-014-literature-verification.md) | Literature survey: verify every citation, extend related work | report | T-012 | before deadline | Todo (team) |

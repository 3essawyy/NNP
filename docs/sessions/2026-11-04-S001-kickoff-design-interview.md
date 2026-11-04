# Session S001 — 2026-11-04 — Kickoff: brief, data inspection, design interview

- **Tasks touched:** T-001
- **Branches / PRs:** `claude/modest-allen-w7yhnf` (cloud session)

## Done
- Read the brief and GATP's repo hierarchy (template for this repo).
- Downloaded and inspected the dataset (after the user allowed `drive.usercontent.google.com` in the environment's network policy).
- Design interview (grill-me) in four rounds; decisions below.

## Observations (facts, not assumptions)
- `Device_A/` is empty; `Device_B/train_data.csv` = 3604 × (561 + target); no missing values.
- `eval_data/` = 3090 single-row CSVs + `ground_truth.txt`.
- Files 1–1545 share Device_B's offsets; 1546–3090 have ≈ −10 on features 0–250.
- Global-scaler RBF SVM: 41 % / 19 % on the two halves; logistic regression with per-half standardization ≈ 93.5 %.

## Decisions
| Topic | Decision | ADR |
|---|---|---|
| Track | Standard leaderboard | — |
| Repo/docs | Mirror GATP + experiments/ + report/ | ADR-001 |
| Git | Bootstrap `main`, PRs from the designated branch, squash-merge | ADR-011 |
| Tooling | Python 3.12, uv, ruff, mypy, pytest, poe, CI; requirements.txt exported | ADR-002 |
| Compute/models | CPU-only, broad classical + MLP search | ADR-002, ADR-007 |
| Dataset access | allow host in env network policy; data never committed | ADR-003 |
| Eval labels | validation/tuning only, never training | ADR-003 |
| Adaptation | unlabeled transductive adaptation allowed | ADR-004 |
| Device_A | proceed single-source + ask staff | ADR-003 |
| Timing | amortize one-off adaptation into every time.txt line | ADR-004 |
| Model file | commit models/model.pkl while small | ADR-008 |
| Report | LaTeX in repo, generated numbers | ADR-012 |
| HF bonus | Claude builds, team deploys | ADR-010 |
| Scope | everything end-to-end this session | — |
| Process | **test every step and let feedback steer** (user, mid-session) | CLAUDE.md |

## Open items / blockers
- Team number, member names, deadline → placeholders (`TEAM_NN`, Member 1–4).
- Ask staff: Device_A data? Hidden test overlaps eval_data?
- Course Overleaf template.

## Next steps
See S002.

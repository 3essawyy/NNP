# Session S002 — 2026-11-04/05 — End-to-end build and handoff

- **Tasks touched:** T-002 … T-013 (T-001 in S001)
- **Branches / PRs:** `claude/modest-allen-w7yhnf` → [3essawyy/NNP#1](https://github.com/3essawyy/NNP/pull/1), squash-merged to `main` as `98a4022`. The branch was then reset onto `main`.
- **State at handoff:** `main` is green (CI checks + Docker job). The working tree is clean. Every task except T-013/T-014 is Done; those two and the items under "Open items" need the team.

---

## 1. TL;DR
NNP is a complete, tested HAR system for the course brief:
- **Model:** per-device standardization with unsupervised device discovery, a train-only ANOVA feature filter, and an RBF SVM.
- **Score:** **97.31 %** on the released validation set (3090 files from two phones), **0.49 ms** charged per sample.
- **Baseline:** a conventional pipeline scores **24.7 %** on the same data, despite 99.1 % in-domain CV.

Code, Docker, CI, Gradio app, LaTeX report (`report/main.pdf`, 6 pages) and the full docs system are on `main`. What remains is team input (names, template, citations, HF deploy) and two questions for course staff.

## 2. What we did, in order

| # | Step | Outcome |
|---|---|---|
| 1 | Read the brief + GATP's repo hierarchy | NNP mirrors GATP (CLAUDE.md, docs tree, IDs, DoD, Explain-it) |
| 2 | Design interview (grill-me, S001) | 15 decisions; see §3 |
| 3 | Got the data | the Drive connector caps at 10 MB; the user allowed `drive.usercontent.google.com` in the env network policy |
| 4 | **Inspected the data (facts)** | Device_A **empty**; Device_B 3604 × 561 + target; `eval_data` = 3090 single-row CSVs **with labels**, two devices (files 1–1545 / 1546–3090) |
| 5 | **Diagnosed the shift** (EXP-001) | per-feature affine calibration offsets (≈ −10 on features 0–250) + a noise-dominated block (~40–239; kNN at chance even in-domain) |
| 6 | Built `src/nnp/` + tests + CI | per-device z-scoring, probe-feature device clustering, ANOVA filter, 12-model zoo, 3 validation protocols, fast SVM path |
| 7 | `infer.py` + Docker + Gradio app | brief §9 contract; amortized timing; Docker verified offline |
| 8 | Experiments E1–E6 (EXP-001…006) | RBF SVM best; ablation proves both core steps are necessary |
| 9 | Tuning → C = 100, F > 100 → robustness exposed an outlier weakness → clip ±3 tuned in | 96.7 → **97.31 %**; config is a fixed point of the tuning procedure |
| 10 | Re-ran *everything* with the final config | report numbers regenerated; docs rewritten against JSON |
| 11 | Verified 4 ways | `infer.py`, Docker `--network none`, unzipped submission in plain pip, pytest dataset test → all 97.31 % |
| 12 | Report compiled (TeX Live) | 6 pages, no undefined refs/overflow; `report/main.pdf` committed |
| 13 | PR #1 squash-merged | CI green on the final commit |

## 3. Decisions (all recorded as ADRs unless marked)

| Topic | Decision | Who | Where |
|---|---|---|---|
| Track | Standard leaderboard (not research) | user | S001 |
| Repo/docs | Mirror GATP; add `docs/experiments/` (EXP) + `report/` | user | ADR-001 |
| Git | Bootstrap `main`; PRs from the designated `claude/*` branch; squash-merge; reset branch after merge | user | ADR-011 |
| Stack | Python 3.12, uv, ruff, mypy --strict, pytest, poe, GitHub Actions; `requirements.txt` exported from `uv.lock` | user | ADR-002 |
| Compute | CPU-only, broad classical + MLP search | user | ADR-002 |
| Data access | Allow the host in the network policy; **dataset never committed** | user | ADR-003 |
| Eval labels | **Validation/tuning only, never training** | user | ADR-003 |
| Device_A empty | Proceed single-source; ask staff | user | ADR-003 |
| Test-time adaptation | Unlabeled transductive adaptation allowed | user | ADR-004 |
| Timing | One-off adaptation cost **amortized** into every `time.txt` line; I/O untimed | user | ADR-004 |
| Model file | Commit `models/model.pkl` while small (2.75 MB) | user | ADR-008 |
| Report | LaTeX in repo; all numbers generated (`report/generated/`) | user | ADR-012 |
| HF bonus | Claude builds; team deploys | user | ADR-010 |
| Process | **Test every step; let results steer** | user (mid-S001) | CLAUDE.md |
| Method | Per-device standardization + device discovery on low-F probe features | Claude, evidence EXP-001 | ADR-004 |
| Validation protocol | In-domain CV + simulated-shift CV + validation; rule: best val, ties ≤ 0.2 pp → sim-shift → simpler | Claude | ADR-005 |
| Features | Train-only ANOVA F > 100 (315 kept); MI/PCA/LDA rejected | Claude, EXP-003/004 | ADR-006 |
| Final model | RBF SVM C = 100, γ = scale, z-score, clip ±3 | Claude, EXP-002/004 | ADR-007 |
| Docker | `python:3.12-slim`, pinned pip, single-threaded BLAS, non-root, `/data` → `/out` | Claude | ADR-009 |

## 4. Key results (validation unless stated; source `experiments/results/*.json`)

| Metric | Value |
|---|---|
| Shipped accuracy / macro-F1 | **97.31 % / 97.42 %** (device 0: 97.2 %, device 1: 97.4 %) |
| Conventional pipeline | 24.7 % (in-domain CV 99.1 %) |
| No adaptation / no feature selection | 75.9 % / 30.1 % |
| Errors | 83 in total: 72 Sitting↔Standing, 11 among walking classes, **0** dynamic↔static; Laying 100 % |
| Time charged per sample | mean 0.49 ms, p95 0.62 ms; adaptation 0.28 s for 3090 files |
| Best other families (defaults) | vote 96.4 %, LR 96.2 %, LightGBM 96.1 %, MLP 94.2 %; trees ≈ 88 ms/sample |
| Robustness | invariant to any offset/gain; **weakness: 20 % outlier records → 85.3 %** (global scaler 95.4 %) |

## 5. Feedback loop: what testing changed

| Check | Finding | Change |
|---|---|---|
| Full validation run | `pandas.read_csv` 23 ms/file → 72 s total | plain parser; identical predictions; 5 s |
| Per-sample timing | `SVC.predict` ≈ 1 ms/row | exact numpy one-vs-one (`FastRBFSVC`), ≈ 0.3 ms |
| First experiment run stalled | fast-path verification ran for generic predictors (RF: 500 parallel single-row predicts per fit) | verify only re-implementations |
| QDA on real data | svd solver needs n > d per class; the shrinkage variant scored 73–84 % | keep reg 0.5; unit test uses top-40 features |
| `.gitignore` check | `data/` pattern also ignored `src/nnp/data/` (source code!) | anchored to `/data` |
| Ablation (E3) | MI selection far worse (81 %), not "similar" as first written; sim-shift CV rates "no selection" 98 % vs 30 % real | ADR-005/006 corrected; blind spot documented |
| Tuning (E4) | C = 100, F > 100 beats the default | adopted (ADR-007) |
| Robustness (E5) | adapted pipeline fragile to outlier records | clip × scaling sweep → clip ±3 (97.31 %; 76.6 → 85.3 % at 20 % outliers) |
| Report build | `siunitx` missing; wide tables overflowed; 1-decimal clip table hid the ±2/±3 difference | removed package; `table*`; 2 decimals |
| App test | first row 28 ms (lazy fast-path build) | warm-up charged to amortized adaptation |
| Dependency check (handoff) | `uv.lock` pins wheels for **Python 3.12–3.13 only** | open item (§7, R1) |

## 6. Where things are
| Need | Go to |
|---|---|
| Run / reproduce | `README.md`, [SG-001](../SystemGuides/SG-001-setup-and-run.md), [SG-003](../SystemGuides/SG-003-reproduce-experiments.md) |
| Why each choice | [decisions/](../decisions/README.md) (ADR-001…012) |
| Evidence | [experiments/](../experiments/README.md) (EXP-001…006), `experiments/results/*.json` |
| Oral-exam prep | each task's **Explain it** ([tasks/](../tasks/README.md)); reading order in [docs/README](../README.md) |
| Shipped config | `src/nnp/models/final.py` |
| Graded entry point | `infer.py` → `src/nnp/inference/predictor.py` (timer around `predict_one`) |
| Report | `report/main.tex`, `report/sections/*.tex`, `report/main.pdf` |
| Submission checklist | [SG-005](../SystemGuides/SG-005-submission-checklist.md) |

## 7. Open items, risks and recommendations

### Needs the team (blocking submission)
1. **Team number + member names + deadline.** Replace `TEAM_NN` / Member 1–4 in `docs/modules/README.md`, `report/main.tex`, `report/sections/workload.tex` and `README.md`; assign module owners.
2. **Course Overleaf template.** Only the report preamble changes (ADR-012).
3. **Verify every citation (T-014).** All were listed from memory; open each one before submitting.
4. **Workload table:** replace the suggested split in `report/sections/workload.tex` with what really happened.

### Ask course staff
5. Is Device_A data coming? If yes: `poe data → poe train → poe experiments`, and add leave-one-device-out CV.
6. Does the hidden test overlap `eval_data`, and how many files/devices does it have? Adaptation needs ≥ 60 files to split devices and ≥ 30 per device.
7. Is `0.000` acceptable in `time.txt`? About 70 % of lines (2153 of 3090, final Docker run) round to it, which follows the brief's 3-decimal rule literally.
8. Which Python version will grade `infer.py` if not via Docker? (see R1)

### Optional (bonus)
9. Deploy the HF Space ([SG-004](../SystemGuides/SG-004-hf-space.md)); put the link in README + `\HFLink`.
10. Enable branch protection on `main` (GitHub settings).

### Risks Claude would address next (ranked)
- **R1: Python version.** `requirements.txt` pins numpy 2.5 / pandas 3.1 / sklearn 1.9, available for Python ≥ 3.12 here. A grader using Python 3.10/3.11 without Docker may fail to install. *Fix:* test `infer.py` on 3.10/3.11 with relaxed lower-bound pins; the model pickle needs the same scikit-learn version, so either pin a version that supports 3.10+ and retrain, or state "Python 3.12 or Docker" prominently in the README.
- **R2: Outlier-heavy device.** Adaptation degrades with many corrupted records (EXP-005). *Fix:* trimmed statistics: drop records with extreme probe-z before estimating device stats. That should keep 97 % while approaching the global scaler's robustness. Needs EXP-007 + ADR.
- **R3: Small or unbalanced hidden test.** With < 60 files there is no device split, and offset devices would be classified with training stats. *Fix:* a per-sample offset correction for tiny sets (e.g. shift a lone record's probe features to the training median). Evaluate on subsampled validation.
- **R4: Selection bias.** The 97.3 % used validation for tuning; the true hidden-test score may be ≈ 0.5 pp lower. Already stated in the report; no action.
- **R5: Sitting/Standing (72 of 83 errors).** A posture-specific second stage (features + classifier trained only on classes 3/4) is the only lever left for accuracy. Expected gain ≤ 1 pp; try only after R1–R3.

## 8. Next steps (in order)
1. **Team:** items 1–4; ask staff 5–8.
2. **Claude, next session:** R1 (Python-version compatibility test + README note), then R2 (trimmed stats, EXP-007), then R3, each through the feedback loop: test → experiment → ADR → regenerate report → PR.
3. **Submission (T-013):** `NNP_TEAM=<n> uv run poe package`, re-run TC-013 on the unzipped copy, upload zip + PDF, `git tag submission`.
4. **Oral prep:** every member reads HLD → EXP-001 → ADR-003/004/006/007 → all Explain-it sections; each owner rehearses their phase.

## 9. Notes for the next Claude session (environment)
- **Dataset:** `data/radar_project_data` is a *symlink into this session's scratchpad* and will be broken in a new container. Run `uv run poe data` (cloud: the environment must allow `drive.usercontent.google.com`).
- **Env:** `uv sync --extra research`. Gradio is tested in a separate venv (adding `--extra app` reinstalls `nnp`; don't do it during a running experiment).
- **Docker in a cloud session:** start `dockerd` manually. Build from a scratch copy of the context with the proxy CA injected (`/root/.ccr/ca-bundle.crt`, `--network host`, proxy build args). Never commit the CA. See T-010 log.
- **Report build:** `apt-get install texlive-latex-recommended texlive-latex-extra texlive-publishers texlive-fonts-recommended latexmk`, then `latexmk -pdf main.tex` in a copy of `report/`.
- **Runtimes (4 vCPU):** full `poe experiments` ≈ 17 min (compare ≈ 9 min, tuning ≈ 4–5 min); `--only final` ≈ 3 s; `poe check` ≈ 1 min.
- Any change to `models/final.py` → `poe train` → `poe experiments` → update ADR-007 + EXP records → rebuild PDF. Never hand-edit numbers in the report.

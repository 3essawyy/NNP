# Session S002 — 2026-11-04 — End-to-end build: pipeline, experiments, inference, Docker, report draft

- **Tasks touched:** T-002 … T-013
- **Branches / PRs:** `claude/modest-allen-w7yhnf` → [3essawyy/NNP#1](https://github.com/3essawyy/NNP/pull/1) (squash-merged to `main`)

## Done
- **Code:** `src/nnp/` (data, preprocessing, features, models, evaluation, inference, experiments), `infer.py`, Dockerfile, CI (checks + docker job), Gradio app, packaging scripts.
- **Tests:** 56 tests on synthetic data plus a dataset-gated test. CI was green from the first push.
- **Experiments E1–E6** with results in `experiments/results/`, figures in `report/figures/` and tables/macros in `report/generated/`. Interpretations are in EXP-001…006.
- **Shipped model** (ADR-007): RBF SVM, C = 100, γ = scale, ANOVA F > 100 (315 features), per-device z-score, clip ±3. Through `infer.py`: **97.3 %** validation, ≈ 0.5 ms charged per sample, 2.75 MB.
- **Docker:** built and run offline in the session (daemon started manually; proxy CA injected into a scratch build context only).
- **HF app:** tested end-to-end through `gradio_client` in a scratch venv. 80 uploaded files → 2 devices discovered.
- **Report:** LaTeX draft that compiles (TeX Live installed in the session), 5–6 pages, IEEE layout.

## Feedback loop: what testing changed (user rule)
| Check | Finding | Change |
|---|---|---|
| Full validation run | `pandas.read_csv` 23 ms/file → 72 s total | plain parser; identical predictions; 5 s |
| Per-sample timing | `SVC.predict` ≈ 1 ms/row | exact numpy one-vs-one (`FastRBFSVC`), ≈ 0.3 ms |
| First experiment run stalled | fast-path verification ran for generic predictors (RF: 500 parallel single-row predicts per fit) | verify only re-implementations |
| QDA on real data | svd solver needs n > d per class; the shrinkage variant scored 73–84 % | keep reg 0.5; unit test uses top-40 features |
| `.gitignore` check | `data/` pattern also ignored `src/nnp/data/` (source code!) | anchored to `/data` |
| Ablation (E3) | MI selection 74–83 %, not "similar" as ADR-006 claimed; sim-shift CV rates "no selection" 97 % vs 35 % real | ADR-005/006 corrected; blind spot documented |
| Tuning (E4) | C = 100, F > 100 beat the default (97.06 vs 96.7 %) | adopted (ADR-007) |
| Robustness (E5) | adapted pipeline *more* fragile to outlier records than a global scaler | clip × scaling sweep → clip ±3 (97.31 %; 20 % outliers 76.6 → 85.3 %) |
| Report build | `siunitx` missing; wide tables overflowed the column | removed package; `table*` for wide tables |
| App test | first row 28 ms (lazy fast-path build) | warm-up charged to the amortized adaptation |

## Decisions
ADR-002…012 written. ADR-007 final model; ADR-004/005/006 amended with EXP-003/005 findings.

## Open items / blockers (need the team)
1. **Team number, member names, deadline**: `TEAM_NN` / Member 1–4 placeholders in `docs/modules/README.md`, `report/main.tex`, `report/sections/workload.tex`, README.
2. **Ask course staff:** is Device_A data coming? Does the hidden test overlap `eval_data`? (ADR-003)
3. **Course Overleaf template**: move `report/sections/*` into it (ADR-012).
4. **Verify citations** (T-014): all entries were listed from memory.
5. **Deploy the HF Space** (SG-004) and put the link in README + `\HFLink`.
6. **Branch protection** on `main` is not configured (GitHub settings).

## Next steps (in order)
1. Team fills items 1, 3 and 4; Claude regenerates the report if anything changes.
2. If Device_A arrives: `poe data` → `poe train` → `poe experiments`; leave-one-device-out CV becomes possible (extend `models/cv.py`).
3. T-013: `NNP_TEAM=<n> uv run poe package`, run TC-013 on the unzipped copy, upload, tag `submission`.
4. Before the discussion, every member reads all **Explain it** sections plus EXP-001…006.

## Notes for next session
- The dataset lives in the gitignored `data/` (fetch with `uv run poe data`; in cloud sessions the environment must allow `drive.usercontent.google.com`).
- Full experiment suite ≈ 15 min on 4 vCPU (compare is the slow stage: tree models' per-row predict).
- Docker in a cloud session: start `dockerd`, then build with `--network host` and the proxy CA in a scratch copy of the context (see T-010 log).

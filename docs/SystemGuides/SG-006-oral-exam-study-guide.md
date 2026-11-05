# SG-006: Oral discussion study guide

- **Audience:** every team member
- **Last verified:** 2026-11-05 (S003) against `main`

The discussion grades **every member on every phase** (REQ-G2): problem & design, preprocessing & features, model selection & training, evaluation & analysis, conclusion.

## Reading list (in order, ≈ 2 h)
1. [HLD](../architecture/HLD.md): the pipeline and "why this shape" in four lines.
2. [EXP-001](../experiments/EXP-001-shift-diagnosis.md): how we found the shift.
3. ADR-003 (data policy), ADR-004 (adaptation + timing), ADR-006 (features), ADR-007 (final model), ADR-013 (small test sets).
4. EXP-002…007: one table + interpretation each.
5. Every task's **Explain it** section ([tasks/](../tasks/README.md)).
6. The report (`report/main.pdf`).

## The five core questions (everyone, no notes)
| # | Question | Short answer (expand from the linked doc) |
|---|---|---|
| 1 | Why is per-device standardization *exact* for calibration offsets and gains? | `(a·x + b − (a·μ + b)) / (a·σ) = (x − μ)/σ`: the device's own mean/std absorb `a` and `b`. A scaler fitted on another phone can't ([report §II-B](../../report/sections/system.tex)). |
| 2 | How do you know which file came from which phone, without labels? Why not cluster on all features? | k-means on *probe* features, i.e. the lowest training ANOVA F, which are activity-independent; accepted only if silhouette ≥ 0.8 and clusters are large enough. Clustering on all features would find *activities* (static vs dynamic) and erase them (ADR-004). |
| 3 | Why drop ~246 features, and why not mutual information? | Features ~40–239 are noise-dominated: kNN is at chance on them even within Device_B. Kept, they dominate RBF distances after z-scoring (30.1 % vs 97.3 %). MI keeps 32 of them (81.0 %) because it detects non-linear dependence that doesn't transfer (EXP-003). |
| 4 | Where is the timer, and why is adaptation spread over every line? | Around `pipeline.predict_one` in `src/nnp/inference/predictor.py::run`, after the CSV is in memory. The one-off device discovery + statistics is real compute, so it is divided by N and added to every line (ADR-004). File I/O is untimed, per the brief. |
| 5 | What would you do with Device_A data / what are the limitations? | `poe data → poe train → poe experiments`, plus leave-one-device-out CV. Limitations: first-order (affine) correction only; class-mix (prior) shift; outlier-heavy devices (EXP-005); ≈ 10-file mixed folders (EXP-007); validation used for selection. |

## Phase deep-dives (owner + backup)
| Phase | Must explain | Docs | Code to walk through |
|---|---|---|---|
| Problem & design | requirements, data facts, pipeline, timing contract | REQ, HLD, ADR-003/004, T-003, T-009 | `infer.py`, `inference/predictor.py`, `data/io.py::list_sample_files` |
| Preprocessing & features | affine proof, device discovery, clipping, ANOVA vs MI/PCA | ADR-004/006, EXP-001/003, T-005/T-006 | `preprocessing/domain.py`, `preprocessing/clustering.py`, `features/selection.py` |
| Model selection & training | 12 families, 3 protocols, tuning rule, fast SVM, fallback | ADR-005/007/013, EXP-002/004/007, T-007/T-015 | `models/pipeline.py`, `models/cv.py`, `models/fast.py`, `models/final.py` |
| Evaluation & analysis | ablation, robustness, Sit/Stand errors, sim-shift blind spot, selection bias | EXP-003/005/006, T-008 | `evaluation/metrics.py`, `evaluation/shift.py`, `experiments/*` |

## Traps examiners like
- *"You used the validation labels: isn't that cheating?"* → Selection/tuning only, never training (ADR-003); stated in the report; simulated-shift CV reported alongside; we estimate ≈ 0.5 pp optimism.
- *"Is using test-set statistics allowed?"* → No labels are used; it is unsupervised test-time adaptation (cf. AdaBN), and its cost is in `time.txt`.
- *"Your in-domain CV is lower than a plain SVM's (96 vs 99 %): why accept that?"* → The plain SVM scores 24.7 % on new phones. We optimize transfer.
- *"Why are many time.txt lines 0.000?"* → Sub-0.5 ms compute rounded to 3 decimals as the brief requires; the unrounded total is in the logs/report.
- *"Why not deep learning?"* → Features are already hand-crafted aggregates; the MLP scored 94.2 % (below the SVM) and would add a large dependency; deep models pay off on raw signals.

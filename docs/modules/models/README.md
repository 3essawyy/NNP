# Module: models

- **Owner:** TBD
- **Status:** Implemented
- **Code:** `src/nnp/models/` (`pipeline.py`, `zoo.py`, `cv.py`, `final.py`, `train.py`, `bundle.py`, `fast.py`)
- **Tasks:** T-007, T-009
- **ADRs:** ADR-005, ADR-007, ADR-008

## Responsibility
Compose preprocessing → selection → reduction → classifier into one object (`HARPipeline`), provide the model zoo, the three validation protocols, the shipped configuration, training, persistence, and the fast single-sample predictor.

## Interfaces
| Direction | Interface | Contract |
|---|---|---|
| Provides | `HARPipeline(PipelineConfig).fit(X, y, devices)` | per-device standardize (if `adapt`), clip, select, reduce, fit classifier, fit clusterer |
| Provides | `.adapt(X_test) -> Adaptation` | one-off, label-free: discovered devices + their stats on the selected columns |
| Provides | `.predict_one(x, adaptation) -> int` | per-sample step timed by `infer.py` |
| Provides | `.predict(X, adaptation=None) -> labels` | batch; identical to `predict_one` row by row (tested) |
| Provides | `make_model(name, params, seed)`, `MODEL_NAMES` | 12 families |
| Provides | `in_domain_cv`, `shifted_cv`, `validate` | ADR-005 protocols |
| Provides | `FINAL_CONFIG`, `nnp-train` / `poe train`, `save_bundle`/`load_bundle` | ADR-007, ADR-008 |
| Consumes | features, preprocessing, evaluation.shift | — |

## Internal design
- `PipelineConfig` is a frozen dataclass; `describe()` goes into the bundle metadata and the report.
- With `adapt=False` the pipeline is the classic baseline: one scaler fitted on all training data, applied to everything.
- **Fast path** (`fast.py`): for RBF SVMs `FastRBFSVC` evaluates `exp(−γ‖sv − z‖²)` for all support vectors with one mat-vec, the 15 one-vs-one decisions with a precomputed weight matrix (libsvm coefficient layout), and a majority vote (ties → lowest class, like libsvm). ≈ 0.3 ms vs ≈ 1 ms for `SVC.predict` on one row. Verified exact at fit time on 500 training rows (falls back to `predict` on any mismatch) and in `tests/test_pipeline.py`; byte-identical `results.txt` on the 3090 validation files.
- The fast predictor is dropped from the pickle (`__getstate__`) and rebuilt on first use.

## Model zoo (defaults)
`gaussian_nb`, `lda` (Ledoit–Wolf shrinkage), `qda` (reg_param 0.5), `logreg` (C=1), `linear_svc` (C=0.05), `svc_rbf` (C=10, γ=scale; the shipped model uses C=100 — `final.py`), `knn` (k=10, distance), `random_forest`/`extra_trees` (300 trees), `lightgbm` (300, lr 0.05), `mlp` (256 hidden, α=0.01, early stopping), `ensemble` (hard vote RBF SVM + linear SVM + LR).

## Tests
`tests/test_pipeline.py` (adaptation beats global scaler under shift; batch = single; fast SVC exact; pickle round-trip; fallback; every family trains; CV protocols).

## Known limitations
Hyperparameters were tuned on the validation set (ADR-003/005) — reported as such.

# ADR-008: Model artifact — committed pickle bundle

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-S3, T-009

## Decision
- `models/model.pkl` is a pickle of `{"format": 1, "pipeline": HARPipeline, "meta": {...versions, config, n_train...}}`, written by `uv run poe train`, **committed to git** while it stays small (≈ 3.4 MB; switch to a GitHub release asset above ~20 MB).
- `infer.py` loads it from a path relative to its own location, so it works from any working directory.
- The fast single-row predictor is **not** pickled (rebuilt lazily), keeping the file small.
- The Docker build loads the bundle once, failing the build on version drift.

## Alternatives considered
| Option | Why not |
|---|---|
| Git LFS | Extra setup for every member and for graders |
| Release asset only | The zip/Docker must contain the model anyway; extra download step |
| ONNX / skops | Safer than pickle, but the brief prefers .pkl and we only load our own files |

## Consequences
- Retraining changes a binary in git: always retrain via `poe train`, then `poe experiments --only final`, and commit both.
- Pickle is version-sensitive: scikit-learn is pinned in `requirements.txt` and recorded in the bundle metadata.

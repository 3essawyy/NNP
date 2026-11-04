# ADR-009: Docker image for inference

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S002
- **Related:** REQ-S4, T-010, TC-010

## Decision
- `python:3.12-slim`, `pip install -r requirements.txt` (pinned), copy only `infer.py`, `src/` (without experiments) and `models/model.pkl` (`.dockerignore` allow-list).
- Single-threaded BLAS (`OMP_NUM_THREADS=1` etc.): per-sample work is tiny and thread start-up only adds latency/jitter.
- Non-root user; working directory `/out` (results land there); `ENTRYPOINT python /app/infer.py`, default argument `/data`.
- The build fails if the model bundle can't be loaded.
- Usage: `docker run --rm -v <data>:/data:ro -v "$PWD/out":/out nnp-har`. Verified with `--network none`.

## Alternatives considered
| Option | Why not |
|---|---|
| Install via uv inside the image | One more tool in the image; pip + exported pins is what staff expect |
| Full repo in the image | Larger, includes research deps |
| Multi-stage/distroless | Marginal gain for ~650 MB image; harder to debug for staff |

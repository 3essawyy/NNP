# ADR-010: Bonus — Hugging Face Space (Gradio), deployed by a team member

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** REQ-S5, T-011, SG-004

## Decision
- `app/app.py`: Gradio UI — upload CSVs, get predictions with the same pipeline (batch adaptation when ≥ 60 files).
- `scripts/build_space.py` assembles `dist/hf_space/` (app, `nnp` package, model, pinned requirements, Space README front-matter).
- Claude cannot hold Hugging Face credentials, so a team member creates the Space and pushes the folder (SG-004); the link then goes into README and report.

## Consequences
- No HF token ever enters the repo.
- Single-file uploads use training reference statistics (no adaptation) — stated in the UI.

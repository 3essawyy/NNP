---
title: NNP Robust HAR
emoji: 🏃
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# NNP — Robust Human Activity Recognition (Hugging Face Space)

This folder is the source of the bonus Hugging Face Space (ADR-010). The deployable Space is
**assembled** by `uv run python scripts/build_space.py`, which copies this app, the `nnp`
package, the trained `models/model.pkl` and a pinned `requirements.txt` into `dist/hf_space/`.

Deployment steps (a team member with a Hugging Face account): [SG-004](../docs/SystemGuides/SG-004-hf-space.md).

# T-011: Bonus — Gradio app + Hugging Face Space bundle

- **Status:** Done (code) · Deploy: Todo (team account)
- **Phase:** M1
- **Modules:** [deploy](../modules/deploy/README.md)
- **Requirements:** REQ-S5
- **Depends on:** T-009
- **ADRs:** [ADR-010](../decisions/ADR-010-hf-space.md)
- **Guides:** [SG-004](../SystemGuides/SG-004-hf-space.md)
- **TCs:** TC-011

## Acceptance criteria
- [x] `uv run poe app` serves the UI locally
- [x] `scripts/build_space.py` assembles a self-contained Space folder
- [ ] A team member deploys it and adds the link to README + report (TODO(team))

## Explain it (code-defense prep)
- **What:** a web demo of the same pipeline.
- **Likely questions:** *Does it adapt?* → yes when ≥ 60 files are uploaded together; otherwise training reference statistics (shown in the UI).

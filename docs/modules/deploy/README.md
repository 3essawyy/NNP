# Module: deploy

- **Owner:** TBD
- **Status:** Implemented (HF Space deployment pending a team account)
- **Code:** `Dockerfile`, `.dockerignore`, `app/`, `scripts/build_space.py`, `scripts/package_submission.py`
- **Tasks:** T-010, T-011, T-013
- **ADRs:** ADR-009, ADR-010

## Responsibility
Everything that leaves the repository: Docker image, Hugging Face Space, the submission zip.

## Interfaces
| Artefact | Command | Contract |
|---|---|---|
| Docker image | `docker build -t nnp-har .` / `docker run --rm -v <data>:/data:ro -v "$PWD/out":/out nnp-har` | results in `/out`; offline-capable; non-root |
| HF Space folder | `uv run python scripts/build_space.py` → `dist/hf_space/` | Gradio app + `nnp` + model + pins |
| Submission zip | `NNP_TEAM=<n> uv run poe package` → `dist/code_<n>.zip` | `readme.md`, `requirements.txt`, `infer.py`, `models/model.pkl`, `Dockerfile`, source, docs, report |

## Verified
Docker: built and run in S002 (`--network none`) with the final model: 97.31 % on validation, 0.49 ms/sample (unrounded mean), image 654 MB. CI job builds the image and runs it on synthetic data every PR.

## Known limitations
In this cloud sandbox the Docker build needs the proxy CA injected (scratch context only; the shipped Dockerfile is unchanged) — see SG-001 troubleshooting.

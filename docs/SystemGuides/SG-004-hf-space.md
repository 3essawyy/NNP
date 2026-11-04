# SG-004: Deploy the Hugging Face Space (bonus)

- **Audience:** team member with a Hugging Face account
- **Last verified:** 2026-11-04 — Space folder assembled; public deploy not yet done

## Steps
1. Build the folder:
   ```bash
   uv run python scripts/build_space.py
   ```
2. On huggingface.co: **New Space** → SDK **Gradio** → name e.g. `nnp-har` → public.
3. Push the folder (needs a write token — keep it out of the repo):
   ```bash
   cd dist/hf_space && git init && git remote add origin https://huggingface.co/spaces/<user>/nnp-har && git add . && git commit -m "Deploy" && git push -u origin main --force
   ```
   (or upload the files through the web UI).
4. Wait for the build; upload ≥ 60 CSVs from `eval_data/data` to test.
5. Put the link in `README.md` ("Live demo") and `report/main.tex` (`\HFLink`).

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| Build fails on numpy/sklearn versions | Space Python differs | keep the pins; set `python_version: "3.12"` in README front-matter |
| Model load error | pickle/sklearn mismatch | same scikit-learn version as `requirements.txt` |

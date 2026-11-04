# Acceptance test case definitions

Short TCs live here; long ones get their own file from [_TEMPLATE.md](_TEMPLATE.md).

### TC-001 — Traceability
**Steps:** for each row of [REQ-requirements](../RefrenceDocs/REQ-requirements.md), find the task(s) in [tasks/README](../tasks/README.md) and the module doc. **Expected:** none missing.

### TC-002 — Fresh clone
**Steps:** `git clone …; cd NNP; uv sync --extra research; uv run poe check`. **Expected:** lint, mypy, pytest green (dataset tests skipped).

### TC-009 — Inference contract
**Steps:** `uv run poe data && uv run poe infer && uv run poe score`. **Expected:** `results.txt`/`time.txt` with 3090 lines each; only `[0-5]` / `\d+\.\d{3}`; accuracy printed. Automated part: `tests/test_inference.py`.

### TC-010 — Docker
**Steps:** `docker build -t nnp-har .`; `docker run --rm --network none -v $PWD/data/radar_project_data/eval_data/data:/data:ro -v $PWD/out:/out nnp-har`; `uv run python -m nnp.evaluation.score out/results.txt data/radar_project_data/eval_data/ground_truth.txt out/time.txt`. **Expected:** both files, accuracy as in E6.
**Evidence (2026-11-04, final model):** 3090 lines each, 97.31 %, 0.49 ms/sample unrounded, `--network none`, image 654 MB.

### TC-011 — HF Space
**Steps:** [SG-004](../SystemGuides/SG-004-hf-space.md); upload ≥ 60 validation CSVs. **Expected:** table of predictions, "2 device(s) discovered" when both halves are included.

### TC-012 — Report
**Steps:** build `report/`; check sections Abstract, Introduction, System Overview, Experimental Setup, Results, Conclusion, Workload Distribution; grep `report/sections` for hard-coded percentages (should be macros). **Expected:** all present; no `TODO(team)` left at submission.

### TC-013 — Submission zip
**Steps:** `NNP_TEAM=<n> uv run poe package`; unzip in an empty folder; `pip install -r requirements.txt`; `python infer.py <data>`. **Expected:** `readme.md`, `requirements.txt`, `infer.py`, `models/model.pkl`, `Dockerfile` at the zip root; inference works without uv.

### TC-014 — Score
**Steps:** `uv run pytest -m dataset` (with data). **Expected:** ≥ 95 % validation accuracy, two discovered devices.

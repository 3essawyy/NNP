# Reference Docs

| File | What | Notes |
|---|---|---|
| [Pattern_Recognition_Project_Fall_2026.pdf](Pattern_Recognition_Project_Fall_2026.pdf) | Official project brief (8 pages) | Authoritative source |
| [REQ-requirements.md](REQ-requirements.md) | Brief extracted into traceable `REQ-*` IDs | Used by tasks and TCs |
| [literature.md](literature.md) | Literature survey notes: HAR on inertial features, domain generalization/adaptation, robust scaling | Feeds report §Related work (T-014) |

## Dataset
- `radar_project_data.7z` (18.7 MB) from the brief's Drive link — fetched by `uv run poe data` ([SG-001](../SystemGuides/SG-001-setup-and-run.md)); never committed (ADR-003).
- Layout as released: `student_realease/Device_A/` (empty), `student_realease/Device_B/train_data.csv` (3604 × 562), `eval_data/data/1.csv … 3090.csv`, `eval_data/ground_truth.txt`.

## External references (add as used)

| Topic | Link | Used by |
|---|---|---|
| scikit-learn user guide | https://scikit-learn.org/stable/user_guide.html | models, features, preprocessing |
| UCI HAR dataset (the feature definitions this data is derived from) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones | data, report |
| uv | https://docs.astral.sh/uv/ | tooling |
| Hugging Face Spaces (Gradio SDK) | https://huggingface.co/docs/hub/spaces-sdks-gradio | deploy |

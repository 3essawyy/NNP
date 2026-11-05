# NNP — Robust Human Activity Recognition under device shift

Course project for **Pattern Recognition and Neural Networks** (Cairo University, Computer Engineering). Team `TEAM_NN` — TODO(team): names.

Classifies smartphone inertial telemetry (561 features per record) into six activities — 0 Walking, 1 Upstairs, 2 Downstairs, 3 Sitting, 4 Standing, 5 Laying — on phones **not seen during training**.

| | Validation (released `eval_data`, 3090 files, 2 devices) |
|---|---|
| Accuracy | **97.3 %** (macro-F1 97.4 %) |
| Time charged per sample (`time.txt`) | 0.49 ms mean (p95 0.62 ms), incl. amortized adaptation |
| Conventional pipeline (global scaler, all features, RBF SVM) | 24.7 % (despite 99.1 % in-domain CV) |

Live demo (bonus): TODO(team) — see [SG-004](docs/SystemGuides/SG-004-hf-space.md).

## How it works (one paragraph)
The training phone (Device_B) and the test phones differ by per-feature calibration offsets and gains. We standardize **each device with its own statistics**, which cancels any per-feature affine distortion. At test time we don't know which file came from which phone, so `infer.py` first **discovers the devices** without labels: it clusters the test set on activity-independent "probe" features (those with the lowest training ANOVA F-score). A train-only ANOVA filter drops ~246 noise-dominated features, and an RBF SVM classifies. The SVM decision is re-implemented in numpy for sub-millisecond per-sample latency. Details: [HLD](docs/architecture/HLD.md) · decisions: [ADRs](docs/decisions/README.md).

## Quick start
Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.12). Full guide incl. Windows: [SG-001](docs/SystemGuides/SG-001-setup-and-run.md).

```bash
uv sync --extra research
```
```bash
uv run poe data
```
```bash
uv run poe infer
```
```bash
uv run poe score
```

### Run inference exactly as the brief specifies
```bash
pip install -r requirements.txt
```
```bash
python infer.py /path/to/data
```
Reads `1.csv … N.csv` in numeric order, loads `models/model.pkl` automatically, and writes `results.txt` (one label 0–5 per line) and `time.txt` (seconds per sample, 3 decimals) to the **current working directory**. Options: `--model PATH`, `--out-dir DIR`, `-q`.

**Timing rule:** files are read first (disk I/O is untimed, per the brief). The one-off unlabeled adaptation (device clustering + statistics) is timed and its cost is divided equally over all lines. Each line = that sample's preprocessing + prediction time + its share of the adaptation ([ADR-004](docs/decisions/ADR-004-unlabeled-test-time-adaptation.md)).

### Docker
```bash
docker build -t nnp-har .
```
```bash
docker run --rm -v /path/to/data:/data:ro -v "$PWD/out":/out nnp-har
```
`results.txt` and `time.txt` appear in `./out/`. The image needs no network at run time.

## Dependencies
Runtime (`requirements.txt`, pinned): numpy, pandas, scipy, scikit-learn, joblib. Research extras (`uv sync --extra research`): lightgbm, matplotlib, py7zr. Bonus app: `--extra app` (gradio).

## Reproduce everything
```bash
uv run poe train
```
```bash
uv run poe experiments
```
Rewrites `experiments/results/*.json`, `report/figures/*` and `report/generated/*.tex`. The report pulls every number from there ([SG-003](docs/SystemGuides/SG-003-reproduce-experiments.md)).

## Development
`uv run poe check` runs lint, strict types and tests, like CI. See [SG-002](docs/SystemGuides/SG-002-developer-workflow.md) and [CLAUDE.md](CLAUDE.md).

## Repository layout
```
infer.py               submission entry point
models/model.pkl       trained pipeline (loaded automatically)
src/nnp/               data · preprocessing · features · models · evaluation · inference · experiments
tests/                 pytest suite (synthetic data; dataset tests skip without data/)
experiments/results/   experiment outputs (JSON)
report/                LaTeX report (figures + numbers generated)
app/                   Gradio app for the Hugging Face Space (bonus)
scripts/               submission zip + Space builder
docs/                  requirements, design, decisions, tasks, guides, session logs
Dockerfile             inference image
```

## Small test folders
`infer.py` adapts per device when the folder is large enough. Groups under 12 records fall back to the training statistics and a logistic-regression fallback model (≈ 88 % per file; [ADR-013](docs/decisions/ADR-013-small-test-set-fallback.md)). The full validation score is unaffected.

## Data
**Data usage statement:** the model is trained on `Device_B` only. Labels of the released `eval_data` were used for model selection and tuning, never for training ([ADR-003](docs/decisions/ADR-003-data-usage-policy.md)).

The dataset (`radar_project_data.7z`) is **not** in the repository. `uv run poe data` downloads it from the course link into `data/`. Labels of the released `eval_data` are used for model selection only, never for training ([ADR-003](docs/decisions/ADR-003-data-usage-policy.md)).

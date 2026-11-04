# SG-001: Setup and run

- **Audience:** developer | examiner
- **Last verified:** 2026-11-04 on Linux (4 vCPU) — uv 0.11, Python 3.12.13, Docker 29.9

## Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (installs Python 3.12 itself). Windows: PowerShell `irm https://astral.sh/uv/install.ps1 | iex`.
- Optional: Docker.

## Steps
1. Install:
   ```bash
   uv sync --extra research
   ```
2. Get the dataset into `data/` (gitignored):
   ```bash
   uv run poe data
   ```
   If the download is blocked, download `radar_project_data.7z` from the brief's Drive link and run `uv run nnp-download --archive path/to/radar_project_data.7z`.
3. (Optional) retrain — `models/model.pkl` is already committed:
   ```bash
   uv run poe train
   ```
4. Predict the validation folder (writes `results.txt`, `time.txt` in the current directory):
   ```bash
   uv run poe infer
   ```
   or for any folder: `uv run python infer.py path/to/data`.
5. Score:
   ```bash
   uv run poe score
   ```

### Without uv (what graders do with the zip)
```bash
pip install -r requirements.txt
python infer.py /path/to/data
```

### Docker
```bash
docker build -t nnp-har .
docker run --rm -v /path/to/data:/data:ro -v "$PWD/out":/out nnp-har
```
Outputs land in `./out/`.

## Verify it worked
`uv run poe score` prints `"accuracy": 0.967` (validation) and per-sample time ≈ 0.4–0.6 ms.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `training root not found` | dataset missing | step 2 |
| `model bundle not found` | `models/model.pkl` missing | `uv run poe train` |
| `non-numeric sample file names` | a file like `a.csv` in the data folder | remove it (we refuse to guess the order) |
| `py7zr missing` | research extra not installed | `uv sync --extra research` |
| Docker build can't reach PyPI behind a corporate/TLS proxy | container doesn't trust the proxy CA | build with `--network host` and proxy build args; add the CA in a local copy of the build context (don't commit it) |
| `Cannot connect to the Docker daemon` | daemon not running | start Docker Desktop / `sudo systemctl start docker` |

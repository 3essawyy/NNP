# T-019: Timing check on the slowest team machine

- **Status:** Todo (team, before submission)
- **Modules:** [inference](../modules/inference/README.md)
- **Requirements:** REQ-G1, REQ-I3

## Goal
Our latency (≈ 0.5 ms/sample) was measured on a 4-vCPU cloud container; graders' hardware is unknown.

## Steps
1. On the slowest laptop: `uv sync`, `uv run poe data`, `uv run poe infer`, `uv run poe score`.
2. Record `time_mean_ms` and the `adapt` seconds printed by `infer.py` in this file.
3. If mean > 2 ms: profile `predict_one` (support-vector count, BLAS threads) before touching the model; any accuracy-for-speed trade needs an ADR (CLAUDE.md rule).

## Results
| Machine | OS / CPU | mean ms | adapt s | accuracy |
|---|---|---|---|---|
| Cloud container (S002) | Linux, 4 vCPU | 0.47–0.49 | 0.3–0.4 | 97.31 % |

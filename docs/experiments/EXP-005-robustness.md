# EXP-005: Robustness under controlled synthetic shifts (E5)

- **Date:** 2026-11-04 (S002; final run with the shipped config)
- **Code:** `src/nnp/experiments/robustness.py`; results `experiments/results/robustness.json`; report `fig:robust`
- **Decisions:** ADR-007 (clip ±3 adopted because of this experiment), limitations in the report

## Question
How does accuracy degrade when one shift factor grows, with per-device adaptation vs a global scaler?

## Setup
- Stratified 80/20 split of Device_B; both pipelines are trained on the 80 %.
- The held-out 20 % goes through `apply_shift` with one factor varied at a time (seed 1).
- Offsets are applied to 50 % of features.

## Result (accuracy %)

| Factor | Values | Adapted (shipped) | Global scaler |
|---|---|---|---|
| Offset (× std) | 0, 0.5, 1, 2, 5, 10, 20 | 96.1 at every value | 95.7, 95.6, 91.1, **38.3**, 20.9, 13.7, 13.7 |
| Log-gain σ | 0, 0.1, 0.2, 0.4, 0.8 | 96.1 at every value | 95.7, 96.0, 95.8, 95.7, 92.9 |
| Additive noise (× std) | 0, 0.1, 0.25, 0.5, 1.0 | 96.1, 95.7, 94.5, 90.4, 77.9 | 95.7, 95.6, 94.0, 90.7, 67.1 |
| Outlier-record rate | 0, 0.01, 0.05, 0.1, 0.2 | 96.1, 95.3, 94.3, 92.4, **85.3** | 95.7, 95.7, 95.7, 95.7, 95.4 |

## Interpretation
- **Offsets and gains:** adaptation makes accuracy exactly invariant, as the algebra predicts. The global scaler collapses from offsets of 2σ upward, below chance by 10σ.
- **Noise** degrades both. Adaptation removes no noise but is somewhat more tolerant at high noise (77.9 vs 67.1 %).
- **Outlier records: a genuine weakness of adaptation.** Outliers contaminate the device's own mean/std, which shifts and squashes every record. The global scaler's statistics come from clean training data. Clip ±3 reduced the damage (76.6 → 85.3 % at 20 % outliers, compared with ±8) but did not close the gap. Robust median/IQR statistics reach about 91 % (S002 check) at a 0.6 pp validation cost (EXP-004).

## Consequences
- Report limitation: devices with many corrupted records degrade adapted accuracy. Mitigations, in order: robust statistics; trimmed statistics (drop records with extreme probe z before estimating); per-record outlier detection.

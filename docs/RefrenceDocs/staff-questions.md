# Questions for course staff

Send before the deadline; record answers here with the date and who answered, then act on them.

| # | Question | Why it matters | If yes / if no | Answer |
|---|---|---|---|---|
| 1 | The released `Device_A` folder is empty. Will Device_A training data be released? | Second labelled device → leave-one-device-out CV, likely better generalization | yes → `poe data`, `poe train`, `poe experiments`, add LODO CV | |
| 2 | Is the hidden test set the released `eval_data/data` (or overlapping it)? | If identical, our validation score ≈ final score | — | |
| 3 | Roughly how many files will the hidden `data` folder contain, and from how many phones? | Adaptation needs batches; < 10 mixed-device files is our weakest case (EXP-007) | small → prioritize per-sample offset correction | |
| 4 | Will `infer.py` be run once on the whole folder (as §9 describes) or once per file? | Per-file runs disable adaptation → fallback model (≈ 88 %) | per-file → prioritize per-sample correction | |
| 5 | Is `0.000` acceptable in `time.txt` (sub-millisecond samples rounded to 3 decimals)? | ≈ 70 % of our lines are `0.000` | no → report with more decimals only if staff allow | |
| 6 | Which Python version / environment runs `infer.py` if not via our Dockerfile? | Our pins target Python 3.12–3.13 (T-016) | older → T-016 first | |
| 7 | Is reading all CSVs before timing (I/O untimed), then charging the one-off adaptation evenly to every sample, acceptable under §9.2.2? | Our timing protocol (ADR-004) | no → ask which alternative they prefer | |

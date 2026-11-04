# T-003: Data module — loaders, numeric ordering, download

- **Status:** Done
- **Phase:** M1
- **Modules:** [data](../modules/data/README.md)
- **Requirements:** REQ-D1, REQ-D2, REQ-I1
- **Depends on:** T-002
- **ADRs:** [ADR-003](../decisions/ADR-003-data-usage-policy.md)
- **TCs:** TC-004

## Acceptance criteria
- [x] Load every `Device_*` folder; skip empty ones with a warning (Device_A)
- [x] Sample files sorted numerically (`10.csv` after `9.csv`); non-numeric names rejected
- [x] Robust single-sample parser (header order, `target` column, empty cells)
- [x] `poe data` downloads + extracts the archive into gitignored `data/`

## Notes / log
- 2026-11-04: `pandas.read_csv` took ~23 ms per 561-column file (72 s for validation). Replaced with a plain parser (~1 ms); predictions byte-identical. Feedback loop: measured → changed → re-measured.

## Explain it (code-defense prep)
- **What:** file → array conversion with the brief's ordering guarantee.
- **Why:** an ordering bug would shift every prediction by one line and score ~chance even with a perfect model. A string sort puts `10.csv` before `2.csv`.
- **How:** `list_sample_files` filters `.csv`, rejects non-integer stems, sorts by `int(stem)`; `read_sample` splits header/row, maps columns by name, drops `target`, empty → NaN.
- **Likely questions:** *What if a file is corrupt?* → inference substitutes an all-NaN row (imputed to neutral values) so line *i* still matches `i.csv`. *Why not pandas?* → 20× slower for 1-row files; measured.

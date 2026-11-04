# Module: report

- **Owner:** TBD
- **Status:** Draft (team info + course template pending)
- **Code:** `report/main.tex`, `report/sections/*.tex`, `report/refs.bib`, `report/generated/*` (generated), `report/figures/*` (generated)
- **Tasks:** T-012, T-014
- **ADRs:** ADR-012

## Responsibility
The formal PDF report (REQ-S1) with every brief-mandated section, wired to generated numbers.

## Structure
| Brief section | File |
|---|---|
| Abstract | `main.tex` |
| Introduction | `sections/introduction.tex` |
| System Overview | `sections/system.tex` |
| Experimental Setup | `sections/setup.tex` |
| Results (comparative, ablation, robustness, error analysis) | `sections/results.tex` |
| Conclusion (capabilities, limitations, future work) | `sections/conclusion.tex` |
| Workload Distribution | `sections/workload.tex` (TODO(team)) |

## Build
Overleaf: upload `report/` (figures are PDFs). Locally: `cd report && latexmk -pdf main.tex`. Regenerate numbers first: `uv run poe experiments`.

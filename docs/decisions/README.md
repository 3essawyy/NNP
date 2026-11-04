# Decisions (ADRs)

Never delete or rewrite an accepted ADR. To change a decision, add a new ADR and mark the old one `Superseded by ADR-NNN`. Template: [_TEMPLATE.md](_TEMPLATE.md).

| ID | Title | Status | Date |
|---|---|---|---|
| [ADR-001](ADR-001-team-workflow-and-docs.md) | Team workflow and documentation structure (mirrors GATP) | Accepted | 2026-11-04 |
| [ADR-002](ADR-002-tech-stack-and-tooling.md) | Tech stack and tooling (Python 3.12, uv, CPU-only, scikit-learn) | Accepted | 2026-11-04 |
| [ADR-003](ADR-003-data-usage-policy.md) | Data usage: train on Device_* only; validation labels for selection only; Device_A missing | Accepted | 2026-11-04 |
| [ADR-004](ADR-004-unlabeled-test-time-adaptation.md) | Unlabeled test-time adaptation (per-device standardization + device discovery); amortized timing | Accepted | 2026-11-04 |
| [ADR-005](ADR-005-validation-protocol.md) | Validation protocol: in-domain CV, simulated-shift CV, validation | Accepted | 2026-11-04 |
| [ADR-006](ADR-006-feature-selection.md) | Feature selection: train-only ANOVA F filter | Accepted | 2026-11-04 |
| [ADR-007](ADR-007-final-model.md) | Final model: RBF SVM on per-device-standardized, ANOVA-selected features | Accepted | 2026-11-04 |
| [ADR-008](ADR-008-model-artifact.md) | Model artifact: committed pickle bundle | Accepted | 2026-11-04 |
| [ADR-009](ADR-009-docker-image.md) | Docker image for inference | Accepted | 2026-11-04 |
| [ADR-010](ADR-010-hf-space.md) | Bonus: Hugging Face Space, deployed by a team member | Accepted | 2026-11-04 |
| [ADR-011](ADR-011-git-flow.md) | Git flow in cloud sessions | Accepted | 2026-11-04 |
| [ADR-012](ADR-012-report-in-latex.md) | Report in LaTeX, numbers generated | Accepted | 2026-11-04 |

## Open decisions
| Topic | Decide when |
|---|---|
| Retrain with Device_A (leave-one-device-out CV) | if staff release Device_A data |
| Second-order alignment (CORAL) or pseudo-label class-balanced stats | only if a new device shows prior/covariance shift |

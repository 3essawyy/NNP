# T-013: Submission pack

- **Status:** Todo (script done)
- **Phase:** M1
- **Modules:** [deploy](../modules/deploy/README.md), docs
- **Requirements:** REQ-S2…S5, REQ-G6
- **Depends on:** T-010, T-012
- **Guides:** [SG-005](../SystemGuides/SG-005-submission-checklist.md)
- **TCs:** TC-013

## Acceptance criteria
- [x] `NNP_TEAM=<n> uv run poe package` builds `dist/code_<n>.zip` with `readme.md` (TC-013 passed with a test team number: unzipped `infer.py` → 97.31 % in a plain-pip env)
- [ ] Team number filled; zip + report PDF uploaded to the course Drive (TODO(team))
- [ ] `submission` tag on `main`

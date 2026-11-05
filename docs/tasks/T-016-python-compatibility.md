# T-016: Python-version compatibility of the non-Docker path

- **Status:** Todo (next session)
- **Phase:** M1
- **Modules:** [deploy](../modules/deploy/README.md), [inference](../modules/inference/README.md)
- **Requirements:** REQ-S2, REQ-S3
- **Source:** S002 handoff risk R1

## Goal
`pip install -r requirements.txt && python infer.py data` must work for graders who don't use Docker. Today the pins (numpy 2.5, pandas 3.1, scikit-learn 1.9) target Python 3.12–3.13 only.

## Plan
1. Create venvs for Python 3.10, 3.11, 3.12 and 3.13; try installing `requirements.txt`; record which fail.
2. If 3.10/3.11 fail: find the newest scikit-learn/numpy versions supporting them. Lower `requires-python`, re-lock, **retrain** (pickles are version-bound), re-run `poe experiments --only final`, and confirm 97.31 % ± noise.
3. Either way: README "Requirements" line naming the supported Python versions, with Docker as the guaranteed path.
4. Optional: CI matrix job running `infer.py` on synthetic data for each supported version.

## Acceptance criteria
- [ ] Documented supported Python versions, each verified
- [ ] TC-016 added (fresh venv per version → infer.py runs)

"""Module boundaries (docs/modules/README.md) are enforced by import-linter contracts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from nnp.config import REPO_ROOT


def test_import_contracts_hold() -> None:
    # Run in a subprocess: import-linter reconfigures logging when run in-process.
    exe = shutil.which("lint-imports") or str(Path(sys.executable).parent / "lint-imports")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [exe], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_requirements_txt_lists_every_runtime_dependency() -> None:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    names = {
        d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in project["dependencies"]
    }
    req = (REPO_ROOT / "requirements.txt").read_text().lower()
    pinned = {line.split("==")[0].strip() for line in req.splitlines() if "==" in line}
    missing = names - pinned
    assert not missing, f"run `uv run poe requirements`; missing: {missing}"


def test_model_bundle_is_committed() -> None:
    assert (REPO_ROOT / "models" / "model.pkl").is_file()

"""Assemble the Hugging Face Space folder in ``dist/hf_space/`` (bonus, ADR-010)."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf_space"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    shutil.copy2(ROOT / "app" / "app.py", OUT / "app.py")
    shutil.copy2(ROOT / "app" / "README.md", OUT / "README.md")
    shutil.copy2(ROOT / "models" / "model.pkl", OUT / "models" / "model.pkl")
    shutil.copytree(
        ROOT / "src" / "nnp",
        OUT / "src" / "nnp",
        ignore=shutil.ignore_patterns("__pycache__", "experiments"),
    )
    # The Space installs gradio itself (sdk); we add the pinned runtime deps.
    req = [
        ln
        for ln in (ROOT / "requirements.txt").read_text().splitlines()
        if ln and not ln.startswith(("#", " "))
    ]
    (OUT / "requirements.txt").write_text("\n".join(req) + "\n")
    print(f"Space assembled in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

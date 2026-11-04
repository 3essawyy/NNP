"""Build ``dist/code_<team>.zip`` exactly as brief §5.2 asks.

``uv run poe package`` (team number from ``NNP_TEAM`` env var or ``--team``). The zip holds
the source, ``readme.md`` (copy of README.md, lower-case as the brief names it),
``requirements.txt``, ``infer.py``, ``models/model.pkl``, ``Dockerfile`` and the report PDF if
built. It never includes ``data/``.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_FILES = [
    "infer.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "models/model.pkl",
]
INCLUDE_DIRS = ["src", "tests", "scripts", "app", "docs", "experiments", "report"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def collect() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = [(ROOT / "README.md", "readme.md")]
    items += [(ROOT / f, f) for f in INCLUDE_FILES if (ROOT / f).exists()]
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not EXCLUDE_PARTS & set(p.parts):
                items.append((p, p.relative_to(ROOT).as_posix()))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default=os.environ.get("NNP_TEAM", "TEAM_NN"))
    args = parser.parse_args(argv)
    out = ROOT / "dist" / f"code_{args.team}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    items = collect()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arc in items:
            z.write(src, arc)
    print(f"wrote {out} ({len(items)} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

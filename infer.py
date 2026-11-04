#!/usr/bin/env python
"""Submission entry point (brief §5.2, §9).

    python infer.py <path/to/data>

Reads ``1.csv … N.csv`` in numeric order, loads ``models/model.pkl`` (relative to this
file), and writes ``results.txt`` and ``time.txt`` to the current working directory.
Works from a source checkout without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nnp.inference.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

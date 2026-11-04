"""Download and extract the course dataset into ``data/`` (never committed — ADR-003).

Usage: ``uv run poe data`` (needs the ``research`` extra for ``py7zr``), or download
``radar_project_data.7z`` by hand from the brief's Drive link and run
``uv run nnp-download --archive path/to/radar_project_data.7z``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

from nnp.config import DATA_DIR, DATASET_DIR, DATASET_URL

log = logging.getLogger(__name__)


def download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (overwrites)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not url.startswith("https://"):
        raise ValueError("only https URLs are allowed")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310 (https only)
        shutil.copyfileobj(resp, fh)
    return dest


def extract(archive: Path, out_dir: Path) -> None:
    """Extract a .7z archive (requires the optional ``py7zr`` dependency)."""
    try:
        import py7zr
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise SystemExit("py7zr missing: run `uv sync --extra research`") from exc
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use a local .7z instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-extract even if data exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if DATASET_DIR.exists() and not args.force:
        log.info("dataset already present at %s (use --force to re-extract)", DATASET_DIR)
        return 0
    archive = args.archive or DATA_DIR / "radar_project_data.7z"
    if args.archive is None:
        log.info("downloading %s", DATASET_URL)
        download(DATASET_URL, archive)
    log.info("extracting %s → %s", archive, DATA_DIR)
    extract(archive, DATA_DIR)
    log.info("done: %s", DATASET_DIR)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

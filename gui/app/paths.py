"""Filesystem path helpers shared across views.

Currently just `unique_path`, used by Process (collision-free output
directories) and Exports (collision-free output filenames). Lifted out of
views/exports.py so both call sites share one definition.
"""
from __future__ import annotations

from pathlib import Path


def unique_path(parent: Path, stem: str, ext: str = "") -> Path:
    """Return a path under `parent` whose `<stem><ext>` name is currently free.

    On first collision append `_2`, then `_3`, … until a free slot is found.
    `ext` is empty for directories and `.ply` / `.mp4` / etc. for files; in
    both cases "free" means `Path.exists()` returns False.
    """
    target = parent / f"{stem}{ext}"
    if not target.exists():
        return target
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

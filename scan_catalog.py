#!/usr/bin/env python3
"""Scan the data/ tree and (re)build catalog.db.

The catalog is a thin index over the filesystem: paths + relationships, no
file contents. Files stay in data/{processed,outputs,renders,exports}/; the
DB just tells downstream tools (e.g. vizer) which artifacts belong together.

Stored paths are always relative to the data root (the directory that
contains catalog.db), so the same DB is usable from the host and from any
container that mounts the data dir.

Usage:
    python scan_catalog.py                       # scan ./data, write ./data/catalog.db
    python scan_catalog.py --data-root /path     # scan a different root
    python scan_catalog.py --quiet               # suppress per-table summaries
    python scan_catalog.py --dry-run             # parse + report, don't write
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# Five tables. All path columns are RELATIVE to the data root.
# Foreign keys are nullable: an artifact can outlive its parent on disk
# (e.g. a .ply whose training run has been pruned to save space).
#
# A meta table records when the catalog was last built and the schema
# version, so consumers can detect a stale or incompatible DB.

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    -- Filename stem. NOT unique: two videos can share a stem
    -- (e.g. full_living_room.MOV + full_living_room.mp4).
    name        TEXT    NOT NULL,
    path        TEXT    NOT NULL UNIQUE,   -- real ground truth: "arc.MP4"
    size_bytes  INTEGER NOT NULL,
    mtime       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(name);

CREATE TABLE IF NOT EXISTS processed_runs (
    id              INTEGER PRIMARY KEY,
    source_id       INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    name            TEXT    NOT NULL UNIQUE,   -- "arc_splat_exhaustive"
    path            TEXT    NOT NULL UNIQUE,   -- "processed/arc_splat_exhaustive"
    colmap_sparse   TEXT,                       -- "processed/.../colmap/sparse/0" or NULL
    images_dir      TEXT,                       -- "processed/.../images" or NULL
    mtime           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS train_runs (
    id                INTEGER PRIMARY KEY,
    processed_run_id  INTEGER REFERENCES processed_runs(id) ON DELETE SET NULL,
    exp_name          TEXT    NOT NULL,         -- "arc_splat_exhaustive" (= outputs/<exp_name>)
    method            TEXT    NOT NULL,         -- "splatfacto", "splatfacto-mcmc", ...
    timestamp_dir     TEXT    NOT NULL,         -- "2026-04-22_025802"
    path              TEXT    NOT NULL UNIQUE,  -- "outputs/arc_splat_exhaustive/splatfacto/2026-04-22_025802"
    config_yml        TEXT,                     -- relpath to config.yml or NULL
    transforms_json   TEXT,                     -- relpath to dataparser_transforms.json or NULL
    latest_ckpt       TEXT,                     -- relpath to highest-step ckpt or NULL
    latest_step       INTEGER,                  -- step number of latest_ckpt
    mtime             INTEGER NOT NULL,
    UNIQUE (exp_name, method, timestamp_dir)
);

CREATE TABLE IF NOT EXISTS exports (
    id            INTEGER PRIMARY KEY,
    train_run_id  INTEGER REFERENCES train_runs(id) ON DELETE SET NULL,
    kind          TEXT    NOT NULL,             -- "splat" | "mesh" | "pcd"
    path          TEXT    NOT NULL UNIQUE,      -- "exports/splat/<file>.ply"
    filename      TEXT    NOT NULL,             -- basename, parsed for vizer regex
    exp_name      TEXT,                          -- parsed from filename, or NULL
    method        TEXT,                          -- parsed from filename, or NULL
    timestamp_dir TEXT,                          -- parsed from filename, or NULL
    size_bytes    INTEGER NOT NULL,
    mtime         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS renders (
    id                INTEGER PRIMARY KEY,
    processed_run_id  INTEGER REFERENCES processed_runs(id) ON DELETE SET NULL,
    project_name      TEXT    NOT NULL,         -- "bomber_birds_eye_splat_exhaustive"
    path              TEXT    NOT NULL UNIQUE,
    filename          TEXT    NOT NULL,
    size_bytes        INTEGER NOT NULL,
    mtime             INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_processed_source  ON processed_runs(source_id);
CREATE INDEX IF NOT EXISTS idx_train_processed   ON train_runs(processed_run_id);
CREATE INDEX IF NOT EXISTS idx_train_exp_method  ON train_runs(exp_name, method);
CREATE INDEX IF NOT EXISTS idx_exports_train     ON exports(train_run_id);
CREATE INDEX IF NOT EXISTS idx_exports_parsed    ON exports(exp_name, method, timestamp_dir);
CREATE INDEX IF NOT EXISTS idx_renders_processed ON renders(processed_run_id);
"""

# ---------------------------------------------------------------------------
# Filename parsing (kept compatible with vizer/splat_viewer.py:47-55)
# ---------------------------------------------------------------------------

PLY_NAME_RE = re.compile(
    r"^(?P<exp>.+?)_+(?P<method>splatfacto[A-Za-z0-9\-]*)"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{6})"
    r"(?:_\d+)?\.ply$"
)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

CKPT_STEP_RE = re.compile(r"step-(\d+)\.ckpt$")
TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def _norm(s: str) -> str:
    """Normalize a name for prefix matching: lowercase, hyphens→underscores."""
    return s.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# config.yml dataset extraction (mirrors splat_viewer.py:305-343)
# ---------------------------------------------------------------------------

def read_config_dataset_dir(config_yml: Path) -> Optional[str]:
    """Extract the processed dataset basename from a nerfstudio config.yml.

    The file starts with a tagged-PosixPath block:

        data: &id003 !!python/object/apply:pathlib.PosixPath
        - /
        - workspace
        - processed
        - <dataset>

    so we read the first chunk, find the `data:` line, collect the
    subsequent `- <part>` lines, and return the last component (the
    processed dataset folder name). This avoids a YAML safe-load that
    would choke on the python-object tags.
    """
    try:
        with open(config_yml, "r") as f:
            head = "".join(f.readline() for _ in range(60))
    except OSError:
        return None
    in_data = False
    parts: list[str] = []
    for line in head.splitlines():
        if not in_data:
            if line.lstrip().startswith("data:") and "PosixPath" in line:
                in_data = True
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            parts.append(stripped[2:].strip())
        else:
            break
    if not parts:
        return None
    return parts[-1]


# ---------------------------------------------------------------------------
# Walk + classify
# ---------------------------------------------------------------------------

@dataclass
class ProcessedRun:
    name: str
    rel_path: str
    colmap_sparse: Optional[str]
    images_dir: Optional[str]
    mtime: int


@dataclass
class TrainRun:
    exp_name: str
    method: str
    timestamp_dir: str
    rel_path: str
    config_yml: Optional[str]
    transforms_json: Optional[str]
    latest_ckpt: Optional[str]
    latest_step: Optional[int]
    mtime: int
    config_dataset: Optional[str]   # nerfstudio's view of which processed_run trained this


@dataclass
class Export:
    rel_path: str
    filename: str
    kind: str
    exp_name: Optional[str]
    method: Optional[str]
    timestamp_dir: Optional[str]
    size_bytes: int
    mtime: int


@dataclass
class Render:
    project_name: str
    rel_path: str
    filename: str
    size_bytes: int
    mtime: int


def _safe_stat(p: Path) -> Optional[os.stat_result]:
    try:
        return p.stat()
    except OSError:
        return None


def _safe_iterdir(p: Path) -> Iterable[Path]:
    try:
        return sorted(p.iterdir())
    except OSError:
        return ()


def scan_sources(data_root: Path) -> list[tuple[str, str, int, int]]:
    """Return (name, rel_path, size, mtime) per top-level video file."""
    out = []
    for child in _safe_iterdir(data_root):
        if not child.is_file():
            continue
        if child.suffix.lower() not in VIDEO_EXTS:
            continue
        st = _safe_stat(child)
        if st is None:
            continue
        out.append((child.stem, child.name, st.st_size, int(st.st_mtime)))
    return out


def scan_processed(data_root: Path) -> list[ProcessedRun]:
    proc_root = data_root / "processed"
    out: list[ProcessedRun] = []
    for child in _safe_iterdir(proc_root):
        if not child.is_dir():
            continue
        st = _safe_stat(child)
        if st is None:
            continue
        # Recognize both COLMAP layouts (vizer/splat_viewer.py:220-225).
        a = child / "sparse" / "0" / "cameras.bin"
        b = child / "colmap" / "sparse" / "0" / "cameras.bin"
        sparse_dir: Optional[Path] = None
        if a.exists():
            sparse_dir = a.parent
        elif b.exists():
            sparse_dir = b.parent
        images = child / "images"
        rel_root = child.relative_to(data_root).as_posix()
        out.append(ProcessedRun(
            name=child.name,
            rel_path=rel_root,
            colmap_sparse=sparse_dir.relative_to(data_root).as_posix() if sparse_dir else None,
            images_dir=images.relative_to(data_root).as_posix() if images.is_dir() else None,
            mtime=int(st.st_mtime),
        ))
    return out


def _find_latest_ckpt(models_dir: Path) -> tuple[Optional[Path], Optional[int]]:
    best: tuple[Optional[Path], Optional[int]] = (None, None)
    for f in _safe_iterdir(models_dir):
        if not f.is_file():
            continue
        m = CKPT_STEP_RE.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        if best[1] is None or step > best[1]:
            best = (f, step)
    return best


def scan_train_runs(data_root: Path) -> list[TrainRun]:
    out_root = data_root / "outputs"
    out: list[TrainRun] = []
    for exp_dir in _safe_iterdir(out_root):
        if not exp_dir.is_dir():
            continue
        for method_dir in _safe_iterdir(exp_dir):
            if not method_dir.is_dir():
                continue
            for ts_dir in _safe_iterdir(method_dir):
                if not ts_dir.is_dir():
                    continue
                if not TIMESTAMP_DIR_RE.match(ts_dir.name):
                    continue
                st = _safe_stat(ts_dir)
                if st is None:
                    continue
                config_yml = ts_dir / "config.yml"
                transforms_json = ts_dir / "dataparser_transforms.json"
                models_dir = ts_dir / "nerfstudio_models"
                ckpt_path, ckpt_step = _find_latest_ckpt(models_dir) if models_dir.is_dir() else (None, None)
                cfg_dataset = read_config_dataset_dir(config_yml) if config_yml.is_file() else None
                out.append(TrainRun(
                    exp_name=exp_dir.name,
                    method=method_dir.name,
                    timestamp_dir=ts_dir.name,
                    rel_path=ts_dir.relative_to(data_root).as_posix(),
                    config_yml=config_yml.relative_to(data_root).as_posix() if config_yml.is_file() else None,
                    transforms_json=transforms_json.relative_to(data_root).as_posix() if transforms_json.is_file() else None,
                    latest_ckpt=ckpt_path.relative_to(data_root).as_posix() if ckpt_path else None,
                    latest_step=ckpt_step,
                    mtime=int(st.st_mtime),
                    config_dataset=cfg_dataset,
                ))
    return out


def scan_exports(data_root: Path) -> list[Export]:
    exp_root = data_root / "exports"
    out: list[Export] = []
    for kind_dir in _safe_iterdir(exp_root):
        if not kind_dir.is_dir():
            continue
        kind = kind_dir.name   # splat | mesh | pcd
        for f in _safe_iterdir(kind_dir):
            if not f.is_file():
                continue
            st = _safe_stat(f)
            if st is None:
                continue
            m = PLY_NAME_RE.match(f.name) if kind == "splat" else None
            out.append(Export(
                rel_path=f.relative_to(data_root).as_posix(),
                filename=f.name,
                kind=kind,
                exp_name=m.group("exp") if m else None,
                method=m.group("method") if m else None,
                timestamp_dir=m.group("timestamp") if m else None,
                size_bytes=st.st_size,
                mtime=int(st.st_mtime),
            ))
    return out


def scan_renders(data_root: Path) -> list[Render]:
    rend_root = data_root / "renders"
    out: list[Render] = []
    for proj_dir in _safe_iterdir(rend_root):
        if not proj_dir.is_dir():
            continue
        for f in _safe_iterdir(proj_dir):
            if not f.is_file():
                continue
            st = _safe_stat(f)
            if st is None:
                continue
            out.append(Render(
                project_name=proj_dir.name,
                rel_path=f.relative_to(data_root).as_posix(),
                filename=f.name,
                size_bytes=st.st_size,
                mtime=int(st.st_mtime),
            ))
    return out


# ---------------------------------------------------------------------------
# Linking heuristics
# ---------------------------------------------------------------------------

def link_processed_to_source(
    processed_name: str,
    sources: list[tuple[str, str, int, int]],
) -> Optional[str]:
    """Pick the source video PATH that best matches a processed-run name.

    Strategy:
      1. Among sources whose normalized stem is a prefix of the normalized
         processed name (with a token boundary), keep the longest matches.
      2. If multiple sources tie on stem length (e.g. .MOV + .mp4 with the
         same stem), prefer the largest file. Larger usually means the
         original capture; smaller usually means a derived/preview encode.

    Returns the source's `path` (unique), or None.
    """
    n = _norm(processed_name)
    candidates: list[tuple[int, int, str]] = []   # (stem_len, size, path)
    for stem, path, size, _mtime in sources:
        s = _norm(stem)
        if not s:
            continue
        if n == s or n.startswith(s + "_"):
            candidates.append((len(s), size, path))
    if not candidates:
        return None
    # longest stem wins; ties broken by largest file
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def link_train_to_processed(
    train: TrainRun,
    processed_by_name: dict[str, int],
) -> Optional[int]:
    """Resolve a train_run to a processed_run id.

    Priority:
      1. config.yml `data:` path (authoritative — what nerfstudio actually
         loaded).
      2. exp_name == processed_run.name (the convention).
      3. exp_name + "_splat" (older naming).
      4. Longest-prefix match against processed_run names.
    """
    if train.config_dataset and train.config_dataset in processed_by_name:
        return processed_by_name[train.config_dataset]
    if train.exp_name in processed_by_name:
        return processed_by_name[train.exp_name]
    suffixed = f"{train.exp_name}_splat"
    if suffixed in processed_by_name:
        return processed_by_name[suffixed]
    # Longest-prefix fallback (mirrors vizer/splat_viewer.py:255-267).
    n = _norm(train.exp_name)
    best: Optional[int] = None
    best_len = 0
    for name, pid in processed_by_name.items():
        common = 0
        for a, b in zip(n, _norm(name)):
            if a != b:
                break
            common += 1
        if common > best_len and common >= 4:
            best_len = common
            best = pid
    return best


def link_export_to_train(
    export: Export,
    train_by_key: dict[tuple[str, str, str], int],
    train_by_exp_ts: dict[tuple[str, str], int],
    train_by_method_ts: dict[tuple[str, str], list[tuple[int, str]]],
) -> Optional[int]:
    """Match an export .ply to a train_run.

    Tries in order:
      1. (exp, method, ts)         — the canonical case.
      2. (exp, ts) ignoring method — re-exports with a custom method tag
         (vizer/splat_viewer.py:382-395 documents this case).
      3. (method, ts) — when only one such run exists, the exp_name in
         the filename was likely munged (e.g. trailing-underscore stripped
         on re-export). With 1 candidate, the timestamp+method pair is
         already unique enough to be safe.
      4. (method, ts) with multiple candidates — longest-prefix exp_name
         match against the candidates' on-disk exp names.
    """
    if export.exp_name is None or export.timestamp_dir is None:
        return None
    if export.method is not None:
        tid = train_by_key.get((export.exp_name, export.method, export.timestamp_dir))
        if tid is not None:
            return tid
    tid = train_by_exp_ts.get((export.exp_name, export.timestamp_dir))
    if tid is not None:
        return tid
    if export.method is not None:
        candidates = train_by_method_ts.get((export.method, export.timestamp_dir), [])
        if len(candidates) == 1:
            return candidates[0][0]
        if len(candidates) > 1:
            n = _norm(export.exp_name)
            best_id: Optional[int] = None
            best_len = -1
            for tid_c, exp_c in candidates:
                ec = _norm(exp_c)
                common = 0
                for a, b in zip(n, ec):
                    if a != b:
                        break
                    common += 1
                if common > best_len:
                    best_len = common
                    best_id = tid_c
            return best_id
    return None


def link_render_to_processed(render: Render, processed_by_name: dict[str, int]) -> Optional[int]:
    """Renders use processed-run names verbatim as their project dir."""
    return processed_by_name.get(render.project_name)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_catalog(db_path: Path, data_root: Path, *, quiet: bool = False) -> dict[str, int]:
    """(Re)build the catalog at db_path. Returns per-table row counts."""
    sources = scan_sources(data_root)
    processed = scan_processed(data_root)
    trains = scan_train_runs(data_root)
    exports = scan_exports(data_root)
    renders = scan_renders(data_root)

    conn = sqlite3.connect(db_path)
    try:
        # If an older-schema DB exists, CREATE IF NOT EXISTS would silently
        # leave it alone and we'd hit confusing UNIQUE/missing-column errors
        # on insert. Detect a version mismatch and drop everything first.
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if existing:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None or row[0] != str(SCHEMA_VERSION):
                for t in ("renders", "exports", "train_runs",
                          "processed_runs", "sources", "meta"):
                    conn.execute(f"DROP TABLE IF EXISTS {t}")
                conn.commit()
        conn.executescript(SCHEMA_SQL)
        with conn:                          # one transaction, atomic from reader POV
            conn.execute("DELETE FROM renders")
            conn.execute("DELETE FROM exports")
            conn.execute("DELETE FROM train_runs")
            conn.execute("DELETE FROM processed_runs")
            conn.execute("DELETE FROM sources")

            # sources
            conn.executemany(
                "INSERT INTO sources(name, path, size_bytes, mtime) VALUES (?, ?, ?, ?)",
                sources,
            )
            source_id_by_path = {
                row[0]: row[1]
                for row in conn.execute("SELECT path, id FROM sources")
            }

            # processed_runs — link by source path (unique), since stems collide
            for pr in processed:
                src_path = link_processed_to_source(pr.name, sources)
                src_id = source_id_by_path.get(src_path) if src_path else None
                conn.execute(
                    "INSERT INTO processed_runs(source_id, name, path, "
                    "colmap_sparse, images_dir, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                    (src_id, pr.name, pr.rel_path, pr.colmap_sparse, pr.images_dir, pr.mtime),
                )
            processed_id_by_name = {
                row[0]: row[1]
                for row in conn.execute("SELECT name, id FROM processed_runs")
            }

            # train_runs
            for tr in trains:
                pid = link_train_to_processed(tr, processed_id_by_name)
                conn.execute(
                    "INSERT INTO train_runs(processed_run_id, exp_name, method, "
                    "timestamp_dir, path, config_yml, transforms_json, latest_ckpt, "
                    "latest_step, mtime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, tr.exp_name, tr.method, tr.timestamp_dir, tr.rel_path,
                     tr.config_yml, tr.transforms_json, tr.latest_ckpt, tr.latest_step, tr.mtime),
                )
            train_by_key: dict[tuple[str, str, str], int] = {}
            train_by_exp_ts: dict[tuple[str, str], int] = {}
            train_by_method_ts: dict[tuple[str, str], list[tuple[int, str]]] = {}
            for exp_name, method, ts, tid in conn.execute(
                "SELECT exp_name, method, timestamp_dir, id FROM train_runs"
            ):
                train_by_key[(exp_name, method, ts)] = tid
                # When two methods share an (exp, ts), later one wins. In
                # practice training runs at distinct timestamps, so this
                # collision is rare; match_export prefers the (exp, method, ts)
                # key first so the fallback only fires when the method tag
                # differs.
                train_by_exp_ts[(exp_name, ts)] = tid
                train_by_method_ts.setdefault((method, ts), []).append((tid, exp_name))

            # exports
            for ex in exports:
                tid = link_export_to_train(ex, train_by_key, train_by_exp_ts, train_by_method_ts)
                conn.execute(
                    "INSERT INTO exports(train_run_id, kind, path, filename, "
                    "exp_name, method, timestamp_dir, size_bytes, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (tid, ex.kind, ex.rel_path, ex.filename, ex.exp_name,
                     ex.method, ex.timestamp_dir, ex.size_bytes, ex.mtime),
                )

            # renders
            for r in renders:
                pid = link_render_to_processed(r, processed_id_by_name)
                conn.execute(
                    "INSERT INTO renders(processed_run_id, project_name, path, "
                    "filename, size_bytes, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                    (pid, r.project_name, r.rel_path, r.filename, r.size_bytes, r.mtime),
                )

            # meta
            now = int(time.time())
            conn.execute("DELETE FROM meta")
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("scanned_at", str(now)),
                    ("data_root", str(data_root.resolve())),
                ],
            )

        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("sources", "processed_runs", "train_runs", "exports", "renders")
        }
    finally:
        conn.close()

    if not quiet:
        # Linkage stats — surface broken links so they're not silent.
        c2 = sqlite3.connect(db_path)
        try:
            stats = {
                "processed without source":
                    c2.execute("SELECT COUNT(*) FROM processed_runs WHERE source_id IS NULL").fetchone()[0],
                "train_runs without processed":
                    c2.execute("SELECT COUNT(*) FROM train_runs WHERE processed_run_id IS NULL").fetchone()[0],
                "exports without train_run":
                    c2.execute("SELECT COUNT(*) FROM exports WHERE train_run_id IS NULL").fetchone()[0],
                "renders without processed":
                    c2.execute("SELECT COUNT(*) FROM renders WHERE processed_run_id IS NULL").fetchone()[0],
            }
        finally:
            c2.close()
        print(f"[catalog] {db_path}")
        for t, n in counts.items():
            print(f"  {t:20s} {n}")
        print("[catalog] orphans:")
        for label, n in stats.items():
            print(f"  {label:32s} {n}")

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scan data/ and (re)build catalog.db")
    p.add_argument("--data-root", type=Path,
                   default=Path(__file__).resolve().parent / "data",
                   help="Data directory (default: ./data)")
    p.add_argument("--db-path", type=Path, default=None,
                   help="Output DB path (default: <data-root>/catalog.db)")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and print counts, do not write the DB")
    args = p.parse_args(argv)

    data_root: Path = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"error: --data-root {data_root} is not a directory", file=sys.stderr)
        return 2

    db_path: Path = (args.db_path or (data_root / "catalog.db")).resolve()

    if args.dry_run:
        sources = scan_sources(data_root)
        processed = scan_processed(data_root)
        trains = scan_train_runs(data_root)
        exports = scan_exports(data_root)
        renders = scan_renders(data_root)
        print(f"[dry-run] would write {db_path}")
        print(f"  sources         {len(sources)}")
        print(f"  processed_runs  {len(processed)}")
        print(f"  train_runs      {len(trains)}")
        print(f"  exports         {len(exports)}")
        print(f"  renders         {len(renders)}")
        return 0

    build_catalog(db_path, data_root, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

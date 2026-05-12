#!/usr/bin/env python3
"""Scan the data/ + data_v2/ trees and (re)build catalog.db.

The catalog is a thin index over the filesystem: paths + relationships, no
file contents. Two parallel artifact stores are indexed in a single DB:

* Legacy tree at `data/`: tables `sources`, `processed_runs`, `train_runs`,
  `exports`, `renders`. Links derived from filename / folder conventions.
* Warehouse tree at `data_v2/`: tables `warehouse_sources`,
  `warehouse_processed`, `warehouse_train`, `warehouse_exports`,
  `warehouse_renders`. ULID-keyed; lineage links come from each
  `manifest.json`'s explicit `parent.run_id` field — no filename
  parsing, no convention matching.

Stored paths are always relative to the root the table belongs to (the
legacy tables relative to `--data-root`, the warehouse tables relative to
`--warehouse-root`), so the same DB is usable from any consumer that
mounts either tree.

Usage:
    python scan_catalog.py                          # scan ./data + ./data_v2
    python scan_catalog.py --data-root /path        # legacy tree
    python scan_catalog.py --warehouse-root /path   # warehouse tree
    python scan_catalog.py --quiet                  # suppress per-table summaries
    python scan_catalog.py --dry-run                # parse + report, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

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

SCHEMA_VERSION = 2

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

-- -------------------------------------------------------------------------
-- Warehouse tables (data_v2/). One row per manifest.json found on disk;
-- ULID is the primary key and `parent_run_id` is the manifest's declared
-- lineage pointer. No filename parsing — joins are by ULID.
-- All `*_path` columns are POSIX relpaths relative to the warehouse root.
-- -------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS warehouse_sources (
    run_id        TEXT PRIMARY KEY CHECK(length(run_id) = 26),
    display_name  TEXT,
    kind          TEXT NOT NULL,                -- "video" | "image_folder" | "imported_dataset"
    created_at    TEXT NOT NULL,                -- ISO 8601 UTC
    path          TEXT NOT NULL,                -- relpath to source artifact (or its dir)
    size_bytes    INTEGER,
    mtime         INTEGER
);

CREATE TABLE IF NOT EXISTS warehouse_processed (
    run_id            TEXT PRIMARY KEY CHECK(length(run_id) = 26),
    parent_run_id     TEXT NOT NULL,            -- soft FK -> warehouse_sources(run_id)
    display_name      TEXT,
    created_at        TEXT NOT NULL,
    transforms_json   TEXT,                     -- relpath to transforms.json or NULL
    colmap_sparse     TEXT,                     -- relpath to colmap/sparse/0 or NULL
    images_dir        TEXT,                     -- relpath to images/ or NULL
    mtime             INTEGER
);

CREATE TABLE IF NOT EXISTS warehouse_train (
    run_id              TEXT PRIMARY KEY CHECK(length(run_id) = 26),
    parent_run_id       TEXT NOT NULL,          -- soft FK -> warehouse_processed(run_id)
    display_name        TEXT,
    created_at          TEXT NOT NULL,
    method              TEXT NOT NULL,
    experiment_name     TEXT NOT NULL,
    timestamp_dir       TEXT,
    config_yml          TEXT,
    latest_checkpoint   TEXT,
    latest_step         INTEGER,
    mtime               INTEGER
);

CREATE TABLE IF NOT EXISTS warehouse_exports (
    run_id                 TEXT PRIMARY KEY CHECK(length(run_id) = 26),
    parent_run_id          TEXT NOT NULL,       -- soft FK -> warehouse_train(run_id)
    display_name           TEXT,
    created_at             TEXT NOT NULL,
    kind                   TEXT NOT NULL,       -- "splat" | "pcd" | "mesh"
    exporter               TEXT NOT NULL,       -- e.g. "ns-export gaussian-splat"
    source_checkpoint_step INTEGER NOT NULL,
    payload_path           TEXT,                -- relpath of the primary .ply
    payload_size           INTEGER,
    mtime                  INTEGER
);

CREATE TABLE IF NOT EXISTS warehouse_renders (
    run_id                  TEXT PRIMARY KEY CHECK(length(run_id) = 26),
    parent_run_id           TEXT NOT NULL,      -- soft FK -> warehouse_train(run_id)
    display_name            TEXT,
    created_at              TEXT NOT NULL,
    camera_path_filename    TEXT,
    payload_path            TEXT,
    payload_size            INTEGER,
    mtime                   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_wh_processed_parent ON warehouse_processed(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_wh_train_parent     ON warehouse_train(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_wh_exports_parent   ON warehouse_exports(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_wh_exports_kind     ON warehouse_exports(kind);
CREATE INDEX IF NOT EXISTS idx_wh_renders_parent   ON warehouse_renders(parent_run_id);
"""

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

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
# Warehouse manifest walkers (data_v2/)
# ---------------------------------------------------------------------------
#
# Each per-stage tree is shaped as:
#   data_v2/<stage_plural>/<ULID>/manifest.json + payload(s)
# except `exports/`, which has a kind sub-level:
#   data_v2/exports/<kind>/<ULID>/manifest.json + payload(s)
#
# We read each manifest with `json.load` (no pydantic dep), pick out the
# fields the catalog cares about, and emit a typed record. The manifest's
# `parent.run_id` IS the lineage link — no filename heuristics.


@dataclass
class WarehouseSource:
    run_id: str
    display_name: Optional[str]
    kind: str
    created_at: str
    path: Optional[str]
    size_bytes: Optional[int]
    mtime: int


@dataclass
class WarehouseProcessed:
    run_id: str
    parent_run_id: str
    display_name: Optional[str]
    created_at: str
    transforms_json: Optional[str]
    colmap_sparse: Optional[str]
    images_dir: Optional[str]
    mtime: int


@dataclass
class WarehouseTrain:
    run_id: str
    parent_run_id: str
    display_name: Optional[str]
    created_at: str
    method: str
    experiment_name: str
    timestamp_dir: Optional[str]
    config_yml: Optional[str]
    latest_checkpoint: Optional[str]
    latest_step: Optional[int]
    mtime: int


@dataclass
class WarehouseExport:
    run_id: str
    parent_run_id: str
    display_name: Optional[str]
    created_at: str
    kind: str
    exporter: str
    source_checkpoint_step: int
    payload_path: Optional[str]
    payload_size: Optional[int]
    mtime: int


@dataclass
class WarehouseRender:
    run_id: str
    parent_run_id: str
    display_name: Optional[str]
    created_at: str
    camera_path_filename: Optional[str]
    payload_path: Optional[str]
    payload_size: Optional[int]
    mtime: int


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    """Best-effort manifest read — returns None on missing / unreadable /
    malformed JSON rather than raising so one bad file doesn't kill the
    whole scan."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _payload_size(warehouse_root: Path, relpath: Optional[str]) -> Optional[int]:
    """Stat the file at `<warehouse_root>/<relpath>`; None on missing or
    when relpath is None."""
    if not relpath:
        return None
    try:
        return (warehouse_root / relpath).stat().st_size
    except OSError:
        return None


def _picked_export_payload(outputs: dict, kind: str) -> Optional[str]:
    """The ExportManifest convention is to record the primary payload under
    `outputs.splat_ply` / `outputs.pcd_ply` / `outputs.mesh_ply`. We accept
    any of those keys; first hit wins."""
    for key in ("splat_ply", "pcd_ply", "mesh_ply"):
        if key in outputs and isinstance(outputs[key], str):
            return outputs[key]
    return None


def scan_warehouse_sources(warehouse_root: Path) -> list[WarehouseSource]:
    out: list[WarehouseSource] = []
    root = warehouse_root / "sources"
    for sub in _safe_iterdir(root):
        if not sub.is_dir() or not ULID_RE.match(sub.name):
            continue
        m = _read_manifest(sub / "manifest.json")
        if m is None or m.get("stage") != "source":
            continue
        st = _safe_stat(sub / "manifest.json")
        params = m.get("params") or {}
        outputs = m.get("outputs") or {}
        path = outputs.get("file") or outputs.get("directory")
        out.append(WarehouseSource(
            run_id=m["run_id"],
            display_name=m.get("display_name"),
            kind=params.get("kind", "imported_dataset"),
            created_at=m.get("created_at") or "",
            path=path if isinstance(path, str) else None,
            size_bytes=int(params["size_bytes"]) if isinstance(params.get("size_bytes"), (int, float)) else None,
            mtime=int(st.st_mtime) if st else 0,
        ))
    return out


def scan_warehouse_processed(warehouse_root: Path) -> list[WarehouseProcessed]:
    out: list[WarehouseProcessed] = []
    root = warehouse_root / "processed"
    for sub in _safe_iterdir(root):
        if not sub.is_dir() or not ULID_RE.match(sub.name):
            continue
        m = _read_manifest(sub / "manifest.json")
        if m is None or m.get("stage") != "process":
            continue
        parent = m.get("parent") or {}
        if not isinstance(parent, dict) or parent.get("stage") != "source":
            continue
        outputs = m.get("outputs") or {}
        st = _safe_stat(sub / "manifest.json")
        out.append(WarehouseProcessed(
            run_id=m["run_id"],
            parent_run_id=parent["run_id"],
            display_name=m.get("display_name"),
            created_at=m.get("created_at") or "",
            transforms_json=outputs.get("transforms_json"),
            colmap_sparse=outputs.get("colmap_sparse"),
            images_dir=outputs.get("images_dir"),
            mtime=int(st.st_mtime) if st else 0,
        ))
    return out


def scan_warehouse_train(warehouse_root: Path) -> list[WarehouseTrain]:
    out: list[WarehouseTrain] = []
    root = warehouse_root / "trained"
    for sub in _safe_iterdir(root):
        if not sub.is_dir() or not ULID_RE.match(sub.name):
            continue
        m = _read_manifest(sub / "manifest.json")
        if m is None or m.get("stage") != "train":
            continue
        parent = m.get("parent") or {}
        if not isinstance(parent, dict) or parent.get("stage") != "process":
            continue
        params = m.get("params") or {}
        outputs = m.get("outputs") or {}
        metrics = m.get("metrics") or {}
        st = _safe_stat(sub / "manifest.json")
        latest_step = metrics.get("latest_step")
        out.append(WarehouseTrain(
            run_id=m["run_id"],
            parent_run_id=parent["run_id"],
            display_name=m.get("display_name"),
            created_at=m.get("created_at") or "",
            method=params.get("method", ""),
            experiment_name=params.get("experiment_name", ""),
            timestamp_dir=params.get("timestamp_dir"),
            config_yml=outputs.get("config_yml"),
            latest_checkpoint=outputs.get("latest_checkpoint"),
            latest_step=int(latest_step) if isinstance(latest_step, (int, float)) else None,
            mtime=int(st.st_mtime) if st else 0,
        ))
    return out


def scan_warehouse_exports(warehouse_root: Path) -> list[WarehouseExport]:
    out: list[WarehouseExport] = []
    root = warehouse_root / "exports"
    # Two-level layout: exports/<kind>/<ULID>/manifest.json
    for kind_dir in _safe_iterdir(root):
        if not kind_dir.is_dir():
            continue
        for sub in _safe_iterdir(kind_dir):
            if not sub.is_dir() or not ULID_RE.match(sub.name):
                continue
            m = _read_manifest(sub / "manifest.json")
            if m is None or m.get("stage") != "export":
                continue
            parent = m.get("parent") or {}
            if not isinstance(parent, dict) or parent.get("stage") != "train":
                continue
            params = m.get("params") or {}
            outputs = m.get("outputs") or {}
            st = _safe_stat(sub / "manifest.json")
            payload_path = _picked_export_payload(outputs, params.get("kind", ""))
            out.append(WarehouseExport(
                run_id=m["run_id"],
                parent_run_id=parent["run_id"],
                display_name=m.get("display_name"),
                created_at=m.get("created_at") or "",
                kind=params.get("kind", kind_dir.name),
                exporter=params.get("exporter", ""),
                source_checkpoint_step=int(params.get("source_checkpoint_step", 0)),
                payload_path=payload_path,
                payload_size=_payload_size(warehouse_root, payload_path),
                mtime=int(st.st_mtime) if st else 0,
            ))
    return out


def scan_warehouse_renders(warehouse_root: Path) -> list[WarehouseRender]:
    out: list[WarehouseRender] = []
    root = warehouse_root / "renders"
    for sub in _safe_iterdir(root):
        if not sub.is_dir() or not ULID_RE.match(sub.name):
            continue
        m = _read_manifest(sub / "manifest.json")
        if m is None or m.get("stage") != "render":
            continue
        parent = m.get("parent") or {}
        if not isinstance(parent, dict) or parent.get("stage") != "train":
            continue
        params = m.get("params") or {}
        outputs = m.get("outputs") or {}
        payload_path = outputs.get("video")
        st = _safe_stat(sub / "manifest.json")
        out.append(WarehouseRender(
            run_id=m["run_id"],
            parent_run_id=parent["run_id"],
            display_name=m.get("display_name"),
            created_at=m.get("created_at") or "",
            camera_path_filename=params.get("camera_path_filename"),
            payload_path=payload_path if isinstance(payload_path, str) else None,
            payload_size=_payload_size(warehouse_root, payload_path if isinstance(payload_path, str) else None),
            mtime=int(st.st_mtime) if st else 0,
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

def build_catalog(
    db_path: Path,
    data_root: Path,
    warehouse_root: Optional[Path] = None,
    *,
    quiet: bool = False,
) -> dict[str, int]:
    """(Re)build the catalog at db_path. Returns per-table row counts.

    `warehouse_root` is optional — when omitted the warehouse_* tables are
    populated empty. This keeps the CLI usable in deployments that don't
    have a data_v2/ tree."""
    sources = scan_sources(data_root)
    processed = scan_processed(data_root)
    trains = scan_train_runs(data_root)
    exports = scan_exports(data_root)
    renders = scan_renders(data_root)

    if warehouse_root is not None and warehouse_root.is_dir():
        wh_sources = scan_warehouse_sources(warehouse_root)
        wh_processed = scan_warehouse_processed(warehouse_root)
        wh_train = scan_warehouse_train(warehouse_root)
        wh_exports = scan_warehouse_exports(warehouse_root)
        wh_renders = scan_warehouse_renders(warehouse_root)
    else:
        wh_sources = []
        wh_processed = []
        wh_train = []
        wh_exports = []
        wh_renders = []

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
                for t in ("warehouse_renders", "warehouse_exports",
                          "warehouse_train", "warehouse_processed",
                          "warehouse_sources",
                          "renders", "exports", "train_runs",
                          "processed_runs", "sources", "meta"):
                    conn.execute(f"DROP TABLE IF EXISTS {t}")
                conn.commit()
        conn.executescript(SCHEMA_SQL)
        with conn:                          # one transaction, atomic from reader POV
            conn.execute("DELETE FROM warehouse_renders")
            conn.execute("DELETE FROM warehouse_exports")
            conn.execute("DELETE FROM warehouse_train")
            conn.execute("DELETE FROM warehouse_processed")
            conn.execute("DELETE FROM warehouse_sources")
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

            # warehouse_*: insert by manifest ULID. The parent_run_id links
            # are stored as plain strings rather than FKs because we ingest
            # in topological order but a partial warehouse (e.g. a train run
            # whose parent processed manifest was hand-deleted) shouldn't
            # cause the whole row to be rejected.
            for s in wh_sources:
                conn.execute(
                    "INSERT OR REPLACE INTO warehouse_sources(run_id, "
                    "display_name, kind, created_at, path, size_bytes, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (s.run_id, s.display_name, s.kind, s.created_at,
                     s.path, s.size_bytes, s.mtime),
                )
            for pr in wh_processed:
                conn.execute(
                    "INSERT OR REPLACE INTO warehouse_processed(run_id, "
                    "parent_run_id, display_name, created_at, "
                    "transforms_json, colmap_sparse, images_dir, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (pr.run_id, pr.parent_run_id, pr.display_name,
                     pr.created_at, pr.transforms_json, pr.colmap_sparse,
                     pr.images_dir, pr.mtime),
                )
            for tr in wh_train:
                conn.execute(
                    "INSERT OR REPLACE INTO warehouse_train(run_id, "
                    "parent_run_id, display_name, created_at, method, "
                    "experiment_name, timestamp_dir, config_yml, "
                    "latest_checkpoint, latest_step, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (tr.run_id, tr.parent_run_id, tr.display_name,
                     tr.created_at, tr.method, tr.experiment_name,
                     tr.timestamp_dir, tr.config_yml, tr.latest_checkpoint,
                     tr.latest_step, tr.mtime),
                )
            for ex in wh_exports:
                conn.execute(
                    "INSERT OR REPLACE INTO warehouse_exports(run_id, "
                    "parent_run_id, display_name, created_at, kind, "
                    "exporter, source_checkpoint_step, payload_path, "
                    "payload_size, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ex.run_id, ex.parent_run_id, ex.display_name,
                     ex.created_at, ex.kind, ex.exporter,
                     ex.source_checkpoint_step, ex.payload_path,
                     ex.payload_size, ex.mtime),
                )
            for r in wh_renders:
                conn.execute(
                    "INSERT OR REPLACE INTO warehouse_renders(run_id, "
                    "parent_run_id, display_name, created_at, "
                    "camera_path_filename, payload_path, payload_size, mtime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (r.run_id, r.parent_run_id, r.display_name,
                     r.created_at, r.camera_path_filename, r.payload_path,
                     r.payload_size, r.mtime),
                )

            # meta
            now = int(time.time())
            conn.execute("DELETE FROM meta")
            meta_rows = [
                ("schema_version", str(SCHEMA_VERSION)),
                ("scanned_at", str(now)),
                ("data_root", str(data_root.resolve())),
            ]
            if warehouse_root is not None:
                meta_rows.append(
                    ("warehouse_root", str(warehouse_root.resolve()))
                )
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)", meta_rows,
            )

        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in (
                "sources", "processed_runs", "train_runs", "exports", "renders",
                "warehouse_sources", "warehouse_processed", "warehouse_train",
                "warehouse_exports", "warehouse_renders",
            )
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
                "wh_processed without wh_source":
                    c2.execute(
                        "SELECT COUNT(*) FROM warehouse_processed wp "
                        "LEFT JOIN warehouse_sources ws ON ws.run_id = wp.parent_run_id "
                        "WHERE ws.run_id IS NULL"
                    ).fetchone()[0],
                "wh_train without wh_processed":
                    c2.execute(
                        "SELECT COUNT(*) FROM warehouse_train wt "
                        "LEFT JOIN warehouse_processed wp ON wp.run_id = wt.parent_run_id "
                        "WHERE wp.run_id IS NULL"
                    ).fetchone()[0],
                "wh_exports without wh_train":
                    c2.execute(
                        "SELECT COUNT(*) FROM warehouse_exports we "
                        "LEFT JOIN warehouse_train wt ON wt.run_id = we.parent_run_id "
                        "WHERE wt.run_id IS NULL"
                    ).fetchone()[0],
                "wh_renders without wh_train":
                    c2.execute(
                        "SELECT COUNT(*) FROM warehouse_renders wr "
                        "LEFT JOIN warehouse_train wt ON wt.run_id = wr.parent_run_id "
                        "WHERE wt.run_id IS NULL"
                    ).fetchone()[0],
            }
        finally:
            c2.close()
        print(f"[catalog] {db_path}")
        for t, n in counts.items():
            print(f"  {t:24s} {n}")
        print("[catalog] orphans:")
        for label, n in stats.items():
            print(f"  {label:32s} {n}")

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Scan data/ + data_v2/ and (re)build catalog.db"
    )
    p.add_argument("--data-root", type=Path,
                   default=Path(__file__).resolve().parent / "data",
                   help="Legacy data directory (default: ./data)")
    p.add_argument("--warehouse-root", type=Path,
                   default=Path(__file__).resolve().parent / "data_v2",
                   help="Manifest-driven warehouse directory (default: ./data_v2). "
                   "Pass --no-warehouse to skip the warehouse_* tables.")
    p.add_argument("--no-warehouse", action="store_true",
                   help="Skip scanning the warehouse root; the warehouse_* "
                   "tables will be created but empty.")
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

    warehouse_root: Optional[Path] = None
    if not args.no_warehouse:
        candidate = args.warehouse_root.resolve()
        if candidate.is_dir():
            warehouse_root = candidate
        elif args.warehouse_root != p.get_default("warehouse_root"):
            # Only error out when the user explicitly passed --warehouse-root.
            # The default is allowed to silently miss so the script still works
            # on hosts without a data_v2/ tree.
            print(
                f"error: --warehouse-root {candidate} is not a directory",
                file=sys.stderr,
            )
            return 2

    db_path: Path = (args.db_path or (data_root / "catalog.db")).resolve()

    if args.dry_run:
        sources = scan_sources(data_root)
        processed = scan_processed(data_root)
        trains = scan_train_runs(data_root)
        exports = scan_exports(data_root)
        renders = scan_renders(data_root)
        if warehouse_root is not None:
            wh_sources = scan_warehouse_sources(warehouse_root)
            wh_processed = scan_warehouse_processed(warehouse_root)
            wh_train = scan_warehouse_train(warehouse_root)
            wh_exports = scan_warehouse_exports(warehouse_root)
            wh_renders = scan_warehouse_renders(warehouse_root)
        else:
            wh_sources = wh_processed = wh_train = wh_exports = wh_renders = []
        print(f"[dry-run] would write {db_path}")
        print(f"  legacy data_root      {data_root}")
        print(f"    sources             {len(sources)}")
        print(f"    processed_runs      {len(processed)}")
        print(f"    train_runs          {len(trains)}")
        print(f"    exports             {len(exports)}")
        print(f"    renders             {len(renders)}")
        if warehouse_root is not None:
            print(f"  warehouse_root        {warehouse_root}")
            print(f"    warehouse_sources   {len(wh_sources)}")
            print(f"    warehouse_processed {len(wh_processed)}")
            print(f"    warehouse_train     {len(wh_train)}")
            print(f"    warehouse_exports   {len(wh_exports)}")
            print(f"    warehouse_renders   {len(wh_renders)}")
        else:
            print("  warehouse_root        (skipped)")
        return 0

    build_catalog(db_path, data_root, warehouse_root, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

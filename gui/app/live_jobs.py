"""
Live-processing handoff between the desktop GUI and the vizer live viewer.

When the user picks "Run with vizer" in the GUI's Process tab, the GUI does all
of its normal pre-flight work — register the source, mint a process ULID,
pre-create the run dir, build the `ns-process-data` argv + typed ProcessParams —
and then, INSTEAD of launching the job itself, drops a request file at
``<WAREHOUSE_DIR>/_jobs/<proc_id>.json``. vizer polls that directory, runs the
job with its existing live-snapshot machinery (colmap shim -> live point cloud,
native log panel, Start->Stop toggle), and writes the run's state back into the
same file so the GUI can follow it to completion.

Why a file in the warehouse and not an HTTP call: vizer is a viser websocket app
with no REST surface, and both processes already share the data_v2 warehouse
mount (the host ``data_v2/`` dir is ``/warehouse`` inside both the splat-ns-v2
and the viser-colmap containers, and is the GUI's own ``WAREHOUSE_DIR`` on the
host). A single ULID-named JSON per run, written atomically (``.tmp`` +
``os.replace``), is the least-moving-parts channel that survives either side
restarting. The ``_jobs/`` directory is invisible to the manifest index, which
only scans the five stage dirs (sources/processed/trained/exports/renders), so
it cannot disturb dataset discovery.

State machine (the ``state`` field):

    requested  -> the GUI wrote it; vizer has not claimed it yet
    running    -> vizer claimed it and launched ns-process-data
    done       -> ns-process-data exited 0 and vizer wrote the manifest
    failed     -> ns-process-data exited non-zero / was stopped / setup failed

Only vizer advances ``requested -> running -> {done, failed}``. The GUI only
ever writes the initial ``requested`` record and deletes the file once it has
observed a terminal state. The record's ``argv`` is the literal, fully-resolved
``ns-process-data`` invocation (container paths) the GUI already built, so vizer
honours every basic AND advanced parameter exactly — vizer never re-derives the
command from the form.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SCHEMA_VERSION = "1.0"
JOBS_DIRNAME = "_jobs"

# A `requested` job older than this (seconds) is treated as abandoned and is
# ignored by vizer's poller, so a stale request left over from a previous
# session can never trigger a surprise run. Sized generously so that a GUI which
# cold-starts the viser container still gets claimed once the server boots.
DEFAULT_FRESHNESS_S = 300.0

# Terminal (or long-abandoned `requested`) records older than this are pruned
# on scan so `_jobs/` doesn't accumulate. A `running` record is NEVER pruned by
# age — a long COLMAP run can legitimately exceed any threshold.
_PRUNE_AGE_S = 3600.0

STATE_REQUESTED = "requested"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"
_TERMINAL = (STATE_DONE, STATE_FAILED)


def jobs_dir(warehouse_dir: Path | str) -> Path:
    """Absolute path to the handoff directory, created on first use."""
    d = Path(warehouse_dir) / JOBS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def request_path(warehouse_dir: Path | str, proc_id: str) -> Path:
    """Path to a single run's request file (keyed by the process ULID)."""
    return jobs_dir(warehouse_dir) / f"{proc_id}.json"


def _atomic_write(path: Path, record: dict) -> None:
    """Write JSON atomically: a sibling .tmp then os.replace (same filesystem,
    so the rename is atomic and a reader never sees a half-written file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(tmp, path)


def write_request(warehouse_dir: Path | str, record: dict) -> Path:
    """Write the initial ``requested`` record. The caller supplies the payload
    (proc_id, source_id, argv, params, command, display_name, form, ...); this
    helper stamps schema_version/state/created_at/error and returns the path.

    ``record["proc_id"]`` is required (it names the file)."""
    rec = dict(record)
    rec.setdefault("schema_version", SCHEMA_VERSION)
    rec["state"] = STATE_REQUESTED
    rec["created_at"] = time.time()
    rec.setdefault("error", None)
    proc_id = rec["proc_id"]
    path = request_path(warehouse_dir, proc_id)
    _atomic_write(path, rec)
    return path


def read_record(path: Path | str) -> dict | None:
    """Parse a request file. Returns None if it's missing or unparseable
    (mid-rename / truncated), so callers can simply retry on the next tick."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def update_state(path: Path | str, state: str, error: str | None = None) -> None:
    """Advance a record's ``state`` (and optional ``error``) in place, atomically.
    No-op if the file vanished. Only vizer calls this."""
    path = Path(path)
    rec = read_record(path)
    if rec is None:
        return
    rec["state"] = state
    if error is not None:
        rec["error"] = error
    rec["updated_at"] = time.time()
    _atomic_write(path, rec)


def delete_request(path: Path | str) -> None:
    """Remove a request file. Idempotent. The GUI calls this after it has
    observed a terminal state."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def list_pending(
    warehouse_dir: Path | str, freshness_s: float = DEFAULT_FRESHNESS_S
) -> list[Path]:
    """Fresh, unclaimed (``state == requested``) request files, oldest first.

    Prunes terminal records — and `requested` records abandoned beyond
    ``_PRUNE_AGE_S`` — as a side effect so the directory stays small. A
    ``running`` record is left untouched regardless of age."""
    d = Path(warehouse_dir) / JOBS_DIRNAME
    if not d.is_dir():
        return []
    now = time.time()
    pending: list[tuple[float, Path]] = []
    for p in sorted(d.iterdir()):
        if p.suffix != ".json" or not p.is_file():
            continue
        rec = read_record(p)
        if rec is None:
            continue
        state = rec.get("state")
        created = float(rec.get("created_at") or 0.0)
        age = now - created
        if state == STATE_REQUESTED:
            if age <= freshness_s:
                pending.append((created, p))
            elif age > _PRUNE_AGE_S:
                delete_request(p)
        elif state in _TERMINAL and age > _PRUNE_AGE_S:
            delete_request(p)
    pending.sort(key=lambda t: t[0])
    return [p for _, p in pending]

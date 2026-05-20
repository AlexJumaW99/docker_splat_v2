"""
Warehouse operations.

`manifest.py` defines the data model. This module is the side-effecting
layer: mints ULIDs, computes checksums, copies source files into the
warehouse, and writes manifests beside their artifacts.

All paths produced by these helpers are relative to `WAREHOUSE_DIR` and use
POSIX separators, matching the path convention documented in `manifest.py`.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    CONFIG_DIR,
    CONTAINER_NAME,
    CONTAINER_WAREHOUSE,
    WAREHOUSE_DIR,
)
from app.manifest import (
    ExportManifest,
    ExportParams,
    Host,
    Manifest,
    Parent,
    ProcessManifest,
    ProcessParams,
    RenderManifest,
    RenderParams,
    SourceManifest,
    SourceParams,
    TrainManifest,
    TrainParams,
    ULID_PATTERN,
    _MANIFEST_ADAPTER,
    dump_manifest as _raw_dump_manifest,
    load_manifest,
)


# ---------------------------------------------------------------- index-aware manifest write


def dump_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write a manifest atomically, then refresh the in-memory index."""
    _raw_dump_manifest(manifest, path)
    from app.manifest_index import get_index
    get_index().refresh()


# ---------------------------------------------------------------- constants


# Crockford base32 alphabet (used by ULID spec) — excludes I, L, O, U to
# avoid visual ambiguity with 1/0 and to keep words like "FAIL" out of IDs.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Subdirectory under WAREHOUSE_DIR for each stage.
_STAGE_DIRS = {
    "source":  "sources",
    "process": "processed",
    "train":   "trained",
    "export":  "exports",
    "render":  "renders",
}


# ---------------------------------------------------------------- ULIDs


def mint_ulid() -> str:
    """Mint a fresh Crockford base32 ULID — 26 chars, time-prefixed so a
    lexicographic sort by ID is also a chronological sort. Encoding follows
    https://github.com/ulid/spec; we roll it inline to avoid adding a
    runtime dependency for ~15 lines of code.
    """

    timestamp_ms = int(time.time() * 1000)
    # 80 bits of randomness, per the spec.
    randomness = secrets.token_bytes(10)
    value = (timestamp_ms << 80) | int.from_bytes(randomness, "big")
    chars: list[str] = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


# ---------------------------------------------------------------- paths


def stage_root(stage: str) -> Path:
    """Absolute host path to the stage's directory (created on first use)."""

    p = WAREHOUSE_DIR / _STAGE_DIRS[stage]
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(stage: str, run_id: str) -> Path:
    """Absolute host path to a single run's directory under the warehouse."""

    return stage_root(stage) / run_id


# Output-type sub-folders under the "export" stage. ns-export's mesh
# variants write directories of related files (mesh.ply + .obj + .mtl +
# texture PNGs), so the export stage uses a two-level layout
# (`exports/<kind>/<ULID>/`) where every export — regardless of kind —
# is contained in its own ULID directory. The kind sub-level is the
# user-facing separation requested for the warehouse.
ExportKind = Literal["splat", "pcd", "mesh"]


def export_dir(kind: ExportKind, run_id: str) -> Path:
    """Absolute host path to one export run's directory under
    `WAREHOUSE_DIR/exports/<kind>/<run_id>/`. Diverges from `run_dir()`
    by interposing `<kind>/` so the on-disk tree is human-browsable and
    a future mesh export's texture sidecars stay grouped with their ply."""

    if kind not in ("splat", "pcd", "mesh"):
        raise ValueError(
            f"export kind must be one of splat|pcd|mesh, got {kind!r}"
        )
    parent = WAREHOUSE_DIR / _STAGE_DIRS["export"] / kind
    parent.mkdir(parents=True, exist_ok=True)
    return parent / run_id


def to_warehouse_relpath(absolute: Path) -> str:
    """Translate an absolute host path under WAREHOUSE_DIR into a POSIX
    relpath. Used for the relpath strings stored in manifests."""

    rel = Path(absolute).resolve().relative_to(WAREHOUSE_DIR.resolve())
    return rel.as_posix()


def host_to_container_warehouse(absolute: Path) -> str:
    """Translate a host path under WAREHOUSE_DIR into its container-side
    path under CONTAINER_WAREHOUSE. Mirrors `_host_to_container` for the
    legacy /workspace mount, but for the new bind."""

    rel = Path(absolute).resolve().relative_to(WAREHOUSE_DIR.resolve())
    return f"{CONTAINER_WAREHOUSE}/{rel.as_posix()}"


# ---------------------------------------------------------------- checksums


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return `sha256:<hex>` for a file. Streams in 1 MiB chunks so multi-GB
    videos don't balloon memory."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_dir_listing(folder: Path) -> str:
    """Stable identity hash for an image folder, based on the sorted list
    of (relative posix path, size) tuples. Cheaper than hashing contents
    of hundreds of photos and adequate for the "is this the same source?"
    question — collisions imply identical file inventory.
    """

    folder = folder.resolve()
    h = hashlib.sha256()
    for root, dirs, files in os.walk(folder):
        # Stable enumeration order — os.walk doesn't guarantee one.
        dirs.sort()
        for name in sorted(files):
            full = Path(root) / name
            rel = full.relative_to(folder).as_posix()
            try:
                size = full.stat().st_size
            except OSError:
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(str(size).encode("ascii"))
            h.update(b"\0")
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------- host info


_HOST_CACHE: Host | None = None


def host_info(refresh: bool = False) -> Host:
    """Best-effort reproducibility info for the host + container.
    Cached for the lifetime of the process; pass refresh=True to re-probe
    (e.g. after a container rebuild). Every probe is wrapped in a wide
    try/except — capturing host info must never fail a real job."""

    global _HOST_CACHE
    if _HOST_CACHE is not None and not refresh:
        return _HOST_CACHE

    host = Host()
    try:
        host.hostname = os.uname().nodename
        host.os = f"{os.uname().sysname} {os.uname().release}"
    except Exception:
        pass
    try:
        # First-only GPU model. nvidia-smi lives on the host, not inside
        # the container — we want the HOST gpu, which the container then
        # accesses via --gpus all.
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2.0,
        )
        if r.returncode == 0:
            host.gpu = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        # The image SHA the running container was started from.
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=2.0,
        )
        if r.returncode == 0:
            host.docker_image = r.stdout.strip()
    except Exception:
        pass
    try:
        # COLMAP commit SHA is baked into the image at /etc/colmap-commit.sha
        # by Stage 3 of the Dockerfile. Lives only inside the container.
        r = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "cat", "/etc/colmap-commit.sha"],
            capture_output=True, text=True, timeout=2.0,
        )
        if r.returncode == 0:
            host.colmap_commit = r.stdout.strip()
    except Exception:
        pass

    _HOST_CACHE = host
    return host


# ---------------------------------------------------------------- parent validation


def _validate_parent_exists(parent_stage: str, parent_run_id: str) -> None:
    """Verify the parent run's manifest.json exists on disk. Raises
    FileNotFoundError with a clear message if the lineage target is missing."""

    if parent_stage == "export":
        # Exports use a two-level layout; check all kind subdirs.
        exports_root = WAREHOUSE_DIR / _STAGE_DIRS["export"]
        if exports_root.is_dir():
            for kind_dir in exports_root.iterdir():
                candidate = kind_dir / parent_run_id / "manifest.json"
                if candidate.is_file():
                    return
    else:
        parent_dir = run_dir(parent_stage, parent_run_id)
        if (parent_dir / "manifest.json").is_file():
            return
    raise FileNotFoundError(
        f"Parent {parent_stage}/{parent_run_id} has no manifest.json. "
        f"Refusing to write a child manifest with a broken lineage pointer."
    )


# ---------------------------------------------------------------- source lookup


def find_source_by_checksum(checksum: str) -> SourceManifest | None:
    """Look up a source by SHA-256 checksum via the ManifestIndex.
    Returns the existing SourceManifest if found, None otherwise."""

    from app.manifest_index import get_index
    return get_index().source_by_checksum(checksum)


def _copy_source_file(src: Path, dst_dir: Path) -> Path:
    """Copy `src` into `dst_dir` (which is created), preserving the file
    name. Returns the destination path. Atomic-ish: copy then fsync; on
    failure the partial file is removed so re-registration is safe."""

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        raise
    return dst


def _copy_source_dir(src: Path, dst_dir: Path) -> Path:
    """Copy an image folder into `dst_dir / src.name`, preserving structure.
    Returns the destination directory."""

    dst = dst_dir / src.name
    if dst.exists():
        # Allow re-registration to retry into the same destination — wipe
        # any partial leftover first.
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def register_source_video(
    source_path: Path,
    *,
    display_name: str | None = None,
    video_info=None,  # type: app.ffprobe.VideoInfo | None
) -> SourceManifest:
    """Register a video source. Idempotent: if a SourceManifest already
    exists with the same SHA256, returns it without copying.

    Computes SHA256 (streamed), copies the video to data_v2/sources/<ULID>/,
    captures ffprobe metadata if `video_info` is supplied, and writes the
    SourceManifest. Returns the manifest."""

    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    checksum = sha256_file(source_path)
    existing = find_source_by_checksum(checksum)
    if existing is not None:
        return existing

    run_id = mint_ulid()
    dest_dir = run_dir("source", run_id)
    copied = _copy_source_file(source_path, dest_dir)
    copied_rel = to_warehouse_relpath(copied)

    params = SourceParams(
        kind="video",
        original_filename=source_path.name,
        size_bytes=source_path.stat().st_size,
        fps=(video_info.fps if video_info and getattr(video_info, "fps", None) else None),
        width=(video_info.width if video_info else None),
        height=(video_info.height if video_info else None),
        duration_seconds=(video_info.duration_s if video_info else None),
        codec=(video_info.codec if video_info else None),
    )

    manifest = SourceManifest(
        run_id=run_id,
        display_name=display_name or source_path.name,
        created_at=datetime.now(timezone.utc),
        host=host_info(),
        params=params,
        outputs={"file": copied_rel},
        checksums={copied_rel: checksum},
    )
    dump_manifest(manifest, dest_dir / "manifest.json")
    return manifest


def register_source_image_folder(
    source_path: Path,
    *,
    display_name: str | None = None,
    image_count: int | None = None,
    extensions: list[str] | None = None,
    heic_originals_count: int | None = None,
) -> SourceManifest:
    """Register an image folder source. Identity is a hash of the sorted
    (relpath, size) listing — cheaper than hashing contents, sufficient
    for "same folder as last time?" semantics."""

    source_path = Path(source_path).resolve()
    if not source_path.is_dir():
        raise NotADirectoryError(source_path)

    checksum = sha256_dir_listing(source_path)
    existing = find_source_by_checksum(checksum)
    if existing is not None:
        return existing

    run_id = mint_ulid()
    dest_dir = run_dir("source", run_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_source_dir(source_path, dest_dir)
    copied_rel = to_warehouse_relpath(copied)

    total_size = 0
    for root, _dirs, files in os.walk(copied):
        for name in files:
            try:
                total_size += (Path(root) / name).stat().st_size
            except OSError:
                continue

    params = SourceParams(
        kind="image_folder",
        original_filename=source_path.name,
        size_bytes=total_size,
        image_count=image_count,
        extensions=extensions,
        heic_originals_count=heic_originals_count,
    )

    manifest = SourceManifest(
        run_id=run_id,
        display_name=display_name or source_path.name,
        created_at=datetime.now(timezone.utc),
        host=host_info(),
        params=params,
        outputs={"directory": copied_rel},
        checksums={copied_rel: checksum},
    )
    dump_manifest(manifest, dest_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------- process manifest


def collect_process_metrics(output_dir: Path) -> dict[str, float]:
    """Post-hoc metrics scraped from the on-disk output of ns-process-data.
    Cheap, no stdout parsing required: just file-system inspection. More
    detailed metrics (pose recovery %) would need streaming-log capture
    and can be added later — this is the bare minimum that beats nothing.
    """

    metrics: dict[str, float] = {}
    images = output_dir / "images"
    if images.is_dir():
        # ns-process-data writes JPEG frames (or symlinks for images mode)
        # into images/ — count is the extracted frame count.
        metrics["extracted_frame_count"] = float(
            sum(1 for p in images.iterdir() if p.is_file())
        )
    sparse_root = output_dir / "colmap" / "sparse"
    if sparse_root.is_dir():
        submaps = [p for p in sparse_root.iterdir() if p.is_dir() and p.name.isdigit()]
        metrics["colmap_submaps_total"] = float(len(submaps))
        # After the COLMAP shim runs, the largest submap is promoted into
        # sparse/0. Recording "0" here is honest: the consumed submap is
        # always 0, the original-largest number is lost (would require
        # parsing the shim's log). Future improvement.
        if submaps:
            metrics["colmap_submap_chosen"] = 0.0
    return metrics


def write_process_manifest(
    *,
    output_dir: Path,
    run_id: str,
    parent_source_id: str,
    display_name: str,
    params: ProcessParams,
    command: str,
    duration_seconds: float | None,
    start_time: datetime,
) -> ProcessManifest:
    """Build + persist a ProcessManifest at `output_dir/manifest.json`.

    `output_dir` is the absolute host path to the run's folder under
    data_v2/processed/<ULID>/, which ns-process-data has just written into.
    The manifest is written last, atomically (.tmp + rename via
    dump_manifest), so its presence is the canonical "this run succeeded"
    signal.
    """

    _validate_parent_exists("source", parent_source_id)

    metrics = collect_process_metrics(output_dir)
    if duration_seconds is not None:
        metrics["process_duration_seconds"] = float(duration_seconds)

    outputs: dict[str, str] = {}
    transforms = output_dir / "transforms.json"
    if transforms.is_file():
        outputs["transforms_json"] = to_warehouse_relpath(transforms)
    sparse0 = output_dir / "colmap" / "sparse" / "0"
    if sparse0.is_dir():
        outputs["colmap_sparse"] = to_warehouse_relpath(sparse0)
    images = output_dir / "images"
    if images.is_dir():
        outputs["images_dir"] = to_warehouse_relpath(images)
    depth = output_dir / "depth"
    if depth.is_dir():
        outputs["depth_dir"] = to_warehouse_relpath(depth)

    checksums: dict[str, str] = {}
    if transforms.is_file():
        checksums[outputs["transforms_json"]] = sha256_file(transforms)

    manifest = ProcessManifest(
        run_id=run_id,
        parent=Parent(stage="source", run_id=parent_source_id),
        display_name=display_name,
        created_at=start_time,
        host=host_info(),
        command=command,
        params=params,
        outputs=outputs,
        metrics=metrics,
        checksums=checksums,
    )
    dump_manifest(manifest, output_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------- form params -> ProcessParams


def process_params_from_form(form_params: dict, mode: str) -> ProcessParams:
    """Translate the Process tab's form dict (see _collect_params in
    views/process.py) into a strongly-typed ProcessParams. Centralized
    here so the wiring in process.py stays a one-liner and any change to
    the form schema has a single landing spot."""

    crop = (
        float(form_params.get("crop_top", 0.0) or 0.0),
        float(form_params.get("crop_bottom", 0.0) or 0.0),
        float(form_params.get("crop_left", 0.0) or 0.0),
        float(form_params.get("crop_right", 0.0) or 0.0),
    )
    crop_factor = crop if any(c > 0.0 for c in crop) else None

    radius = form_params.get("percent_radius_crop")
    if radius is not None:
        radius = float(radius)
        if radius >= 1.0:
            radius = None  # ns-process-data's default; don't record as a knob

    extra = form_params.get("extra_flags") or ""
    import shlex
    extra_list = shlex.split(extra) if extra.strip() else []

    return ProcessParams(
        mode=mode,  # type: ignore[arg-type]  # "video" | "images" validated in model
        matching_method=form_params["matching_method"],
        num_frames_target=(
            int(form_params["num_frames_target"]) if mode == "video" else None
        ),
        num_downscales=int(form_params.get("num_downscales", 3)),
        camera_type=form_params.get("camera_type", "perspective"),
        refine_intrinsics=bool(form_params.get("refine_intrinsics", True)),
        same_dimensions=bool(form_params.get("same_dimensions", True)),
        use_gpu=bool(form_params.get("gpu_enabled", True)),
        use_sfm_depth=bool(form_params.get("use_sfm_depth", False)),
        crop_factor=crop_factor,
        percent_radius_crop=radius,
        images_per_equirect=(
            int(form_params["images_per_equirect"])
            if form_params.get("camera_type") == "equirectangular"
            and int(form_params.get("images_per_equirect", 8)) in (8, 14)
            else None
        ),
        random_seed=form_params.get("random_seed"),
        eval_random_seed=form_params.get("eval_random_seed"),
        eval_data=form_params.get("eval_data"),
        extra_flags=extra_list,
    )


# ---------------------------------------------------------------- train parent lookup


def find_parent_process_manifest(processed_dir: Path) -> ProcessManifest | None:
    """Return the ProcessManifest at processed_dir/manifest.json, or None if
    the directory predates the warehouse (legacy data/processed/*_splat/ has
    no manifest). Callers use the None case to gate-out legacy datasets with
    a friendly error rather than write a parent-less TrainManifest."""

    manifest_path = Path(processed_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        m = load_manifest(manifest_path)
    except Exception:
        return None
    if not isinstance(m, ProcessManifest):
        return None
    return m


# ---------------------------------------------------------------- train output discovery


# `step-NNNNNNNNN.ckpt` — Nerfstudio zero-pads to 9 digits today (we've seen
# step-000029999.ckpt in real runs), but pattern accepts any non-empty digit
# run so we don't break if upstream changes the width.
_TRAIN_CKPT_RE = re.compile(r"^step-(\d+)\.ckpt$")


def find_train_run_subdir(output_dir: Path) -> Path | None:
    """Locate ns-train's auto-generated `<exp>/<method>/<ts>/` directory
    inside `output_dir`. ns-train always nests its outputs three levels deep
    under --output-dir; we find the dir by searching for the lone `config.yml`
    it writes at the leaf. Returns None when no run directory was created
    (e.g. ns-train crashed before bootstrapping its writer)."""

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    # rglob is bounded by the nested-3 layout — exactly one config.yml per run,
    # and we'll have just-created `output_dir`, so the walk is tiny.
    matches = list(output_dir.rglob("config.yml"))
    if not matches:
        return None
    # Take the most-recently-modified one as a tiebreaker — defensive against
    # a future ns-train change that drops more than one config.yml. In a
    # normal run there's exactly one.
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].parent


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
    """Return (path, step) for the highest-step ckpt under
    `run_dir/nerfstudio_models/`, or None if none are present. The step is
    parsed from the filename rather than the file contents — cheap and
    avoids loading a multi-GB tensor file just to learn its iteration."""

    ckpt_dir = Path(run_dir) / "nerfstudio_models"
    if not ckpt_dir.is_dir():
        return None
    best: tuple[Path, int] | None = None
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = _TRAIN_CKPT_RE.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if best is None or step > best[1]:
            best = (p, step)
    return best


# ---------------------------------------------------------------- train manifest


def collect_train_metrics(
    run_dir: Path | None,
    duration_seconds: float | None,
    latest_step: int | None,
) -> dict[str, float]:
    """Cheap, log-free metrics scraped from the on-disk run directory.
    Richer scalars (psnr/ssim/lpips/num_gaussians) live in
    `events.out.tfevents.*` and require a tensorboard dep + EventAccumulator
    — deferred to a later turn per the task brief."""

    metrics: dict[str, float] = {}
    if latest_step is not None:
        metrics["latest_step"] = float(latest_step)
    if duration_seconds is not None:
        metrics["train_duration_seconds"] = float(duration_seconds)
    return metrics


def train_params_from_form(
    form_params: dict,
    *,
    experiment_name: str,
    dataparser: str,
    timestamp_dir: str,
) -> TrainParams:
    """Translate the Train tab's form dict (see _collect_params in
    views/train.py) into a strongly-typed TrainParams. `experiment_name`,
    `dataparser`, and `timestamp_dir` aren't on the form — they come from
    the project record and from ns-train's own auto-generated slug."""

    import shlex
    extra = form_params.get("extra_flags") or ""
    extra_list = shlex.split(extra) if extra.strip() else []

    return TrainParams(
        method=form_params["method"],  # type: ignore[arg-type]
        experiment_name=experiment_name,
        timestamp_dir=timestamp_dir,
        max_num_iterations=int(form_params["max_num_iterations"]),
        steps_per_save=int(form_params["steps_per_save"]),
        vis=form_params["vis"],  # type: ignore[arg-type]
        viewer_websocket_host="0.0.0.0",
        viewer_quit_on_train_completion=bool(
            form_params.get("quit_on_train_completion", False)
        ),
        dataparser=dataparser,  # type: ignore[arg-type]
        extra_flags=extra_list,
    )


def write_train_manifest(
    *,
    output_dir: Path,
    run_id: str,
    parent_process_id: str,
    display_name: str,
    params: TrainParams,
    command: str,
    duration_seconds: float | None,
    start_time: datetime,
) -> TrainManifest:
    """Build + persist a TrainManifest at `output_dir/manifest.json`.

    `output_dir` is the absolute host path to the train run's folder under
    data_v2/trained/<train_ULID>/. ns-train has already written its outputs
    into the nested `<experiment-name>/<method>/<timestamp_dir>/` subdir.
    The manifest sits at the top of the ULID directory so any consumer
    walking `data_v2/trained/*/manifest.json` finds one entry per run.
    Atomic via .tmp + rename (dump_manifest) — manifest presence is the
    canonical "this run succeeded" marker.
    """

    _validate_parent_exists("process", parent_process_id)

    nested = find_train_run_subdir(output_dir)
    latest_ckpt = find_latest_checkpoint(nested) if nested is not None else None
    latest_step = latest_ckpt[1] if latest_ckpt is not None else None

    outputs: dict[str, str] = {}
    if nested is not None:
        cfg = nested / "config.yml"
        if cfg.is_file():
            outputs["config_yml"] = to_warehouse_relpath(cfg)
        dp_xforms = nested / "dataparser_transforms.json"
        if dp_xforms.is_file():
            outputs["dataparser_transforms"] = to_warehouse_relpath(dp_xforms)
        # The tensorboard event-log directory IS the run dir. Recording its
        # relpath here lets TB-launchers point at the right --logdir without
        # re-walking the warehouse.
        outputs["tb_logdir"] = to_warehouse_relpath(nested)
    if latest_ckpt is not None:
        outputs["latest_checkpoint"] = to_warehouse_relpath(latest_ckpt[0])

    checksums: dict[str, str] = {}
    if latest_ckpt is not None:
        # Checkpoints are multi-MB; sha256 of the latest is the load-bearing
        # output for any downstream export, so hashing pays for itself.
        checksums[outputs["latest_checkpoint"]] = sha256_file(latest_ckpt[0])

    metrics = collect_train_metrics(nested, duration_seconds, latest_step)

    manifest = TrainManifest(
        run_id=run_id,
        parent=Parent(stage="process", run_id=parent_process_id),
        display_name=display_name,
        created_at=start_time,
        host=host_info(),
        command=command,
        params=params,
        outputs=outputs,
        metrics=metrics,
        checksums=checksums,
    )
    dump_manifest(manifest, output_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------- pending sidecar


# Filename convention for the "this run was started" breadcrumb. Sits next to
# the final manifest.json under data_v2/trained/<ULID>/. Three filesystem
# states encode three run states:
#   manifest.json present           → run succeeded (canonical)
#   only pending.json present       → run started but never finalized (the GUI
#                                     died, the user cancelled, ns-train
#                                     crashed, etc.)
#   neither file present            → run was never attempted (or the dir
#                                     hasn't been created yet)
PENDING_FILENAME = "pending.json"


class PendingTrainRun(BaseModel):
    """Write-ahead-log record for a train run. Persisted BEFORE ns-train
    starts so that lineage (parent.run_id) is durable even if the GUI dies
    between job-start and job-finish — the manifest write previously
    depended on the GUI being alive at completion, which proved fragile.

    Lives at data_v2/trained/<ULID>/pending.json. Finalized into manifest.json
    by either (a) the Train tab's success handler or (b) the backfill helper
    in this module. Schema is intentionally minimal: no metrics, no outputs,
    no checksums — those become knowable only after the run completes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    # Discriminator kept for symmetry with the real Manifest union; consumers
    # walking data_v2/trained/*/pending.json need no extra logic.
    stage: Literal["train"] = "train"
    run_id: str = Field(pattern=ULID_PATTERN)
    parent: Parent  # parent.stage == "process", parent.run_id == ProcessManifest.run_id
    started_at: datetime  # UTC, same instant captured for the finalized manifest's created_at
    command: str  # full `docker exec ... ns-train ...` invocation
    display_name: str | None = None
    host: Host | None = None
    # Verbatim form params (the dict _collect_params() returns). Kept as a
    # dict — pre-TrainParams — so the breadcrumb doesn't need timestamp_dir,
    # which ns-train only picks AFTER it starts.
    form_params: dict
    # Project's dataparser kind ("nerfstudio-data" | "colmap"), captured so
    # backfill doesn't have to re-derive it from config.yml.
    dataparser: Literal["nerfstudio-data", "colmap"]
    # Project ID we'll re-use as experiment-name. Lets a future scan_catalog
    # pass associate the run with its project without parsing the command.
    experiment_name: str


def pending_path(train_dir: Path) -> Path:
    """Absolute path to the pending sidecar inside a train run dir."""

    return Path(train_dir) / PENDING_FILENAME


def write_pending_train(
    train_dir: Path,
    *,
    run_id: str,
    parent_process_id: str,
    started_at: datetime,
    command: str,
    display_name: str | None,
    form_params: dict,
    dataparser: str,
    experiment_name: str,
) -> PendingTrainRun:
    """Atomically write the pending sidecar at `train_dir/pending.json`.
    Called from _on_run BEFORE the JobManager starts ns-train, so the
    breadcrumb is on disk even if the very next instruction crashes."""

    pending = PendingTrainRun(
        run_id=run_id,
        parent=Parent(stage="process", run_id=parent_process_id),
        started_at=started_at,
        command=command,
        display_name=display_name,
        host=host_info(),
        form_params=form_params,
        dataparser=dataparser,  # type: ignore[arg-type]
        experiment_name=experiment_name,
    )
    target = pending_path(train_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(pending.model_dump_json(indent=2, exclude_none=True))
    tmp.replace(target)
    return pending


def load_pending_train(train_dir: Path) -> PendingTrainRun | None:
    """Return the PendingTrainRun at train_dir/pending.json, or None if no
    pending sidecar is present (run never started, or already finalized)."""

    p = pending_path(train_dir)
    if not p.is_file():
        return None
    try:
        return PendingTrainRun.model_validate_json(p.read_bytes())
    except Exception:
        return None


def clear_pending_train(train_dir: Path) -> None:
    """Delete the pending sidecar — called from the Train tab's success
    handler immediately after manifest.json is persisted. Idempotent: a
    missing pending.json is fine (e.g. backfill already wrote the manifest
    in a prior session)."""

    p = pending_path(train_dir)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- backfill


# Minimal hand-rolled extraction for the fields backfill needs out of
# nerfstudio's config.yml. Avoids a yaml dep AND avoids `yaml.unsafe_load`,
# which would execute the `!!python/object:` tags as real constructors —
# unsafe for arbitrary files. The format is generated by yaml.dump with
# consistent indentation, so anchored regex on top-level keys is reliable.


# `data: ... !!python/object/apply:pathlib.PosixPath` followed by list of
# path components on subsequent `- <component>` lines until indentation
# changes. We accept either an anchor (`&id003`) or no anchor.
_POSIXPATH_HEADER_RE = re.compile(
    r"^(?P<key>\w+):\s*(?:&\w+\s+)?!!python/object/apply:pathlib\.PosixPath\s*$"
)
_POSIXPATH_COMPONENT_RE = re.compile(r"^- (?P<part>.*)$")
# Top-level scalar `key: value` (where value has no inline tag/anchor).
_TOP_SCALAR_RE = re.compile(r"^(?P<key>[a-z_][a-z0-9_]*):\s*(?P<val>.*?)\s*$")
# `dataparser: !!python/object:nerfstudio.data.dataparsers.<X>_dataparser.<Y>DataParserConfig`
_DATAPARSER_TAG_RE = re.compile(
    r"^\s*dataparser:\s*!!python/object:"
    r"nerfstudio\.data\.dataparsers\.(?P<module>[a-z_]+)_dataparser\."
    r"(?P<cls>\w+)DataParserConfig\s*$"
)
# Nested `viewer:` block — quit_on_train_completion lives 2 spaces in.
_VIEWER_QOC_RE = re.compile(
    r"^  quit_on_train_completion:\s*(?P<val>true|false|True|False)\s*$"
)


def _extract_posixpath(lines: list[str], start_idx: int) -> Path | None:
    """Given a line index pointing at a `key: !!python/object/apply:pathlib.PosixPath`
    line, walk the subsequent `- <part>` lines and assemble them into a Path.
    Returns None if no component lines follow."""

    parts: list[str] = []
    for line in lines[start_idx + 1:]:
        m = _POSIXPATH_COMPONENT_RE.match(line)
        if m is None:
            break
        parts.append(m.group("part"))
    if not parts:
        return None
    # First part is "/" for absolute paths; subsequent are joined.
    if parts[0] == "/":
        return Path("/").joinpath(*parts[1:])
    return Path(*parts)


def _parse_train_config_yml(config_path: Path) -> dict:
    """Extract the subset of fields needed to rebuild a TrainManifest from a
    config.yml written by nerfstudio's Trainer. Returns a flat dict; missing
    fields are simply absent. Caller decides which are mandatory.

    Recognised keys: data (Path), output_dir (Path), experiment_name (str),
    method_name (str), max_num_iterations (int), steps_per_save (int),
    timestamp (str), vis (str), dataparser_kind ("nerfstudio-data"|"colmap"),
    viewer_quit_on_train_completion (bool)."""

    lines = config_path.read_text(encoding="utf-8").splitlines()
    out: dict = {}
    for i, line in enumerate(lines):
        pp = _POSIXPATH_HEADER_RE.match(line)
        if pp is not None and pp.group("key") in ("data", "output_dir"):
            path = _extract_posixpath(lines, i)
            if path is not None:
                out[pp.group("key")] = path
            continue
        dp = _DATAPARSER_TAG_RE.match(line)
        if dp is not None:
            module = dp.group("module")
            if module == "nerfstudio":
                out["dataparser_kind"] = "nerfstudio-data"
            elif module == "colmap":
                out["dataparser_kind"] = "colmap"
            else:
                out["dataparser_kind"] = module  # best-effort, unknown variant
            continue
        vq = _VIEWER_QOC_RE.match(line)
        if vq is not None:
            out["viewer_quit_on_train_completion"] = vq.group("val").lower() == "true"
            continue
        # Top-level (column 0) scalars.
        if line and not line.startswith(" "):
            sc = _TOP_SCALAR_RE.match(line)
            if sc is not None:
                key = sc.group("key")
                val = sc.group("val")
                if key == "max_num_iterations":
                    try:
                        out[key] = int(val)
                    except ValueError:
                        pass
                elif key == "steps_per_save":
                    try:
                        out[key] = int(val)
                    except ValueError:
                        pass
                elif key in ("experiment_name", "method_name", "timestamp", "vis"):
                    out[key] = val
    return out


def _train_dir_time_window(nested: Path) -> tuple[datetime | None, float | None]:
    """Best-effort (start_time, duration_seconds) from filesystem mtimes.
    `start_time` = earliest mtime in the nested dir (~ config.yml).
    `duration_seconds` = (latest mtime - earliest mtime). Both fields are
    treated as informational only — the canonical start_time for a backfilled
    manifest comes from the pending sidecar if one is present."""

    mtimes: list[float] = []
    for root, _dirs, files in os.walk(nested):
        for name in files:
            try:
                mtimes.append((Path(root) / name).stat().st_mtime)
            except OSError:
                continue
    if not mtimes:
        return None, None
    start = datetime.fromtimestamp(min(mtimes), tz=timezone.utc)
    duration = max(mtimes) - min(mtimes)
    return start, duration


def backfill_train_manifest(train_dir: Path) -> TrainManifest:
    """Reconstruct a TrainManifest for an orphan train run (one with on-disk
    artifacts but no manifest.json). Prefers fields from the pending sidecar
    when present, falls back to scraping config.yml. The resulting manifest
    is written with `provenance="backfilled"`.

    Raises FileNotFoundError if config.yml can't be located; ValueError if
    required fields can't be recovered."""

    train_dir = Path(train_dir).resolve()
    if not train_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {train_dir}")
    if (train_dir / "manifest.json").is_file():
        raise FileExistsError(
            f"manifest.json already exists at {train_dir} — refusing to overwrite. "
            "Delete it first if you really want to re-backfill."
        )

    nested = find_train_run_subdir(train_dir)
    if nested is None:
        raise FileNotFoundError(
            f"no config.yml under {train_dir} — nothing to backfill from."
        )
    cfg_fields = _parse_train_config_yml(nested / "config.yml")

    pending = load_pending_train(train_dir)

    # Resolve every required field, preferring pending sidecar (authoritative
    # since it was captured at run-start, before any post-hoc filesystem
    # mtimes drift) and falling back to config.yml scraping.
    run_id = train_dir.name  # ULID is the directory name by construction
    if not re.fullmatch(ULID_PATTERN, run_id):
        raise ValueError(
            f"directory name {run_id!r} isn't a ULID — refusing to backfill an "
            "ad-hoc path. data_v2/trained/<ULID>/ is the only valid layout."
        )

    # Parent process ULID — pending sidecar is authoritative; config.yml's
    # `data:` path is the fallback (last component after /warehouse/processed/).
    parent_id: str | None = None
    if pending is not None:
        parent_id = pending.parent.run_id
    elif (data_path := cfg_fields.get("data")) is not None:
        # Expecting /warehouse/processed/<ULID>; the trailing component is the
        # parent run_id. Validate it before trusting it.
        candidate = Path(data_path).name
        if re.fullmatch(ULID_PATTERN, candidate):
            parent_id = candidate
    if parent_id is None:
        raise ValueError(
            f"couldn't recover parent process ULID for {train_dir} — neither "
            f"pending.json nor config.yml's `data:` field is usable."
        )

    method = cfg_fields.get("method_name") or (
        pending.form_params.get("method") if pending is not None else None
    )
    if method is None:
        raise ValueError(f"couldn't recover method for {train_dir}")

    experiment_name = (
        (pending.experiment_name if pending is not None else None)
        or cfg_fields.get("experiment_name")
        or run_id
    )
    timestamp_dir = cfg_fields.get("timestamp") or nested.name
    max_iter = cfg_fields.get("max_num_iterations") or (
        int(pending.form_params.get("max_num_iterations", 0))
        if pending is not None else 0
    )
    steps_per_save = cfg_fields.get("steps_per_save") or (
        int(pending.form_params.get("steps_per_save", 0))
        if pending is not None else 0
    )
    vis = cfg_fields.get("vis") or (
        pending.form_params.get("vis") if pending is not None else None
    ) or "viewer+tensorboard"
    qoc = cfg_fields.get("viewer_quit_on_train_completion")
    if qoc is None and pending is not None:
        qoc = bool(pending.form_params.get("quit_on_train_completion", False))
    if qoc is None:
        qoc = False
    dataparser_kind = (
        (pending.dataparser if pending is not None else None)
        or cfg_fields.get("dataparser_kind")
        or "nerfstudio-data"
    )

    fs_start, fs_duration = _train_dir_time_window(nested)
    start_time = (
        pending.started_at if pending is not None
        else (fs_start or datetime.now(timezone.utc))
    )

    params = TrainParams(
        method=method,  # type: ignore[arg-type]
        experiment_name=experiment_name,
        timestamp_dir=timestamp_dir,
        max_num_iterations=int(max_iter) if max_iter else 1,
        steps_per_save=int(steps_per_save) if steps_per_save else 1,
        vis=vis,  # type: ignore[arg-type]
        viewer_websocket_host="0.0.0.0",
        viewer_quit_on_train_completion=bool(qoc),
        dataparser=dataparser_kind,  # type: ignore[arg-type]
        extra_flags=[],
    )

    # Outputs + checksums — same logic as write_train_manifest, but tolerant
    # of missing pieces (a partial run might lack a checkpoint).
    outputs: dict[str, str] = {}
    checksums: dict[str, str] = {}
    cfg = nested / "config.yml"
    if cfg.is_file():
        outputs["config_yml"] = to_warehouse_relpath(cfg)
    dp_xforms = nested / "dataparser_transforms.json"
    if dp_xforms.is_file():
        outputs["dataparser_transforms"] = to_warehouse_relpath(dp_xforms)
    outputs["tb_logdir"] = to_warehouse_relpath(nested)
    latest_ckpt = find_latest_checkpoint(nested)
    if latest_ckpt is not None:
        outputs["latest_checkpoint"] = to_warehouse_relpath(latest_ckpt[0])
        checksums[outputs["latest_checkpoint"]] = sha256_file(latest_ckpt[0])

    latest_step = latest_ckpt[1] if latest_ckpt is not None else None
    metrics = collect_train_metrics(nested, fs_duration, latest_step)

    command = pending.command if pending is not None else ""

    manifest = TrainManifest(
        run_id=run_id,
        parent=Parent(stage="process", run_id=parent_id),
        display_name=(pending.display_name if pending is not None else experiment_name),
        created_at=start_time,
        host=(pending.host if pending is not None else host_info()),
        command=command,
        params=params,
        outputs=outputs,
        metrics=metrics,
        checksums=checksums,
        provenance="backfilled",
    )
    dump_manifest(manifest, train_dir / "manifest.json")
    # The pending sidecar has now served its purpose — replace-then-remove
    # mirrors the success-handler contract.
    clear_pending_train(train_dir)
    return manifest


# ---------------------------------------------------------------- export manifest


# Logical key used in the manifest's `outputs` map for each export kind's
# primary payload .ply file. Consumers grep `outputs.splat_ply` /
# `outputs.pcd_ply` / `outputs.mesh_ply` to locate the artifact without
# scanning the directory.
_EXPORT_PAYLOAD_KEY: dict[str, str] = {
    "splat": "splat_ply",
    "pcd":   "pcd_ply",
    "mesh":  "mesh_ply",
}


def write_export_manifest(
    *,
    output_dir: Path,
    run_id: str,
    parent_train_id: str,
    display_name: str,
    params: ExportParams,
    command: str,
    duration_seconds: float | None,
    start_time: datetime,
    payload_filename: str,
) -> ExportManifest:
    """Build + persist an ExportManifest at `output_dir/manifest.json`.

    `output_dir` is the absolute host path returned by `export_dir()`
    (`data_v2/exports/<kind>/<run_id>/`). The exporter has just written
    `<payload_filename>` there — `splat.ply` for `ns-export gaussian-splat`,
    `point_cloud.ply` / `poisson_mesh.ply` / `mesh.obj` / etc. for the NeRF
    exporters. Any additional files in the directory are recorded as
    `sidecar:<basename>` outputs so mesh-export texture atlases come along
    for the ride. The manifest is written last, atomically, so its
    presence is the canonical "this export succeeded" signal.
    """

    _validate_parent_exists("train", parent_train_id)

    output_dir = Path(output_dir)
    outputs: dict[str, str] = {}
    checksums: dict[str, str] = {}

    payload = output_dir / payload_filename
    payload_key = _EXPORT_PAYLOAD_KEY[params.kind]
    if payload.is_file():
        outputs[payload_key] = to_warehouse_relpath(payload)
        checksums[outputs[payload_key]] = sha256_file(payload)

    # Pick up any sidecar files (texture atlas, .obj/.mtl, point-cloud helper).
    # Skips the payload itself (recorded above), the manifest we're about to
    # write, and any leftover tmp files. Sorted for deterministic output.
    for p in sorted(output_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in (payload_filename, "manifest.json"):
            continue
        if p.name.endswith(".tmp"):
            continue
        outputs[f"sidecar:{p.name}"] = to_warehouse_relpath(p)

    metrics: dict[str, float] = {}
    if duration_seconds is not None:
        metrics["export_duration_seconds"] = float(duration_seconds)
    if payload.is_file():
        metrics["payload_size_bytes"] = float(payload.stat().st_size)

    manifest = ExportManifest(
        run_id=run_id,
        parent=Parent(stage="train", run_id=parent_train_id),
        display_name=display_name,
        created_at=start_time,
        host=host_info(),
        command=command,
        params=params,
        outputs=outputs,
        metrics=metrics,
        checksums=checksums,
    )
    dump_manifest(manifest, output_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------- render manifest


def write_render_manifest(
    *,
    output_dir: Path,
    run_id: str,
    parent_train_id: str,
    display_name: str,
    params: RenderParams,
    command: str,
    duration_seconds: float | None,
    start_time: datetime,
    payload_filename: str,
    extra_outputs: dict[str, str] | None = None,
) -> RenderManifest:
    """Build + persist a RenderManifest at `output_dir/manifest.json`.

    `output_dir` is the absolute host path to the render run's folder
    under `data_v2/renders/<run_id>/`. `payload_filename` is whatever
    name `ns-render` wrote (typically `render.mp4` or the user-supplied
    stem). `extra_outputs` is a logical-name → relpath dict for files
    the caller pre-staged into `output_dir` before the job ran (most
    commonly a snapshot of the camera_path.json used)."""

    _validate_parent_exists("train", parent_train_id)

    output_dir = Path(output_dir)
    outputs: dict[str, str] = dict(extra_outputs or {})
    checksums: dict[str, str] = {}

    payload = output_dir / payload_filename
    if payload.is_file():
        outputs["video"] = to_warehouse_relpath(payload)
        checksums[outputs["video"]] = sha256_file(payload)

    metrics: dict[str, float] = {}
    if duration_seconds is not None:
        metrics["render_duration_seconds"] = float(duration_seconds)
    if payload.is_file():
        metrics["payload_size_bytes"] = float(payload.stat().st_size)

    manifest = RenderManifest(
        run_id=run_id,
        parent=Parent(stage="train", run_id=parent_train_id),
        display_name=display_name,
        created_at=start_time,
        host=host_info(),
        command=command,
        params=params,
        outputs=outputs,
        metrics=metrics,
        checksums=checksums,
    )
    dump_manifest(manifest, output_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------- stale pending detection


def find_stale_pending(max_age_hours: float = 24.0) -> list[Path]:
    """Return train dirs with pending.json but no manifest.json, older
    than *max_age_hours*. These are abandoned runs that can be backfilled
    or cleaned up."""

    trained_root = WAREHOUSE_DIR / "trained"
    if not trained_root.is_dir():
        return []
    stale: list[Path] = []
    cutoff = time.time() - (max_age_hours * 3600)
    for sub in trained_root.iterdir():
        if not sub.is_dir():
            continue
        pending = sub / PENDING_FILENAME
        manifest = sub / "manifest.json"
        if pending.is_file() and not manifest.is_file():
            try:
                if pending.stat().st_mtime < cutoff:
                    stale.append(sub)
            except OSError:
                continue
    return sorted(stale)


# ---------------------------------------------------------------- checksum verification


def verify_manifest_checksums(
    manifest: "Manifest",
    warehouse_dir: Path = WAREHOUSE_DIR,
) -> list[tuple[str, str, str]]:
    """Verify all checksums in a manifest. Returns a list of
    (relpath, expected, actual) tuples for mismatches. Empty = all OK.

    Not called automatically — hashing a multi-GB checkpoint is too
    expensive for routine reads. Exposed as a CLI subcommand and as
    a pre-export integrity check."""

    mismatches: list[tuple[str, str, str]] = []
    for relpath, expected in manifest.checksums.items():
        full = warehouse_dir / relpath
        if not full.is_file():
            mismatches.append((relpath, expected, "MISSING"))
            continue
        actual = sha256_file(full)
        if actual != expected:
            mismatches.append((relpath, expected, actual))
    return mismatches


# ---------------------------------------------------------------- CLI


def _cli_backfill_train(argv: list[str]) -> int:
    """Backfill one or more orphan train run dirs. Used by `python -m
    app.warehouse backfill-train <dir>...`. Returns a shell exit code."""

    if not argv:
        print("usage: python -m app.warehouse backfill-train <train_dir> [<train_dir> ...]")
        return 2
    rc = 0
    for raw in argv:
        td = Path(raw).resolve()
        try:
            m = backfill_train_manifest(td)
        except Exception as e:
            print(f"FAILED  {td}: {e!r}")
            rc = 1
            continue
        print(
            f"BACKFILLED  {td}\n"
            f"  run_id     = {m.run_id}\n"
            f"  parent     = process/{m.parent.run_id}\n"
            f"  method     = {m.params.method}\n"
            f"  experiment = {m.params.experiment_name}\n"
            f"  timestamp  = {m.params.timestamp_dir}\n"
            f"  metrics    = {dict(m.metrics)}"
        )
    return rc


def _cli_verify(argv: list[str]) -> int:
    """Verify checksums for one or more manifest.json files."""
    if not argv:
        print("usage: python -m app.warehouse verify <manifest.json> [...]")
        return 2
    rc = 0
    for raw in argv:
        mf = Path(raw).resolve()
        try:
            m = load_manifest(mf)
        except Exception as e:
            print(f"FAILED  {mf}: {e!r}")
            rc = 1
            continue
        mismatches = verify_manifest_checksums(m)
        if mismatches:
            rc = 1
            for relpath, expected, actual in mismatches:
                print(f"MISMATCH  {relpath}  expected={expected}  actual={actual}")
        else:
            n = len(m.checksums)
            print(f"OK  {mf}  ({n} checksum(s) verified)")
    return rc


def _cli_stale_pending(_argv: list[str]) -> int:
    """List stale pending.json files (no manifest, >24h old)."""
    stale = find_stale_pending()
    if not stale:
        print("No stale pending runs found.")
        return 0
    print(f"Found {len(stale)} stale pending run(s):")
    for d in stale:
        pending = load_pending_train(d)
        label = pending.display_name if pending else "?"
        print(f"  {d.name}  ({label})")
    return 0


def _cli_orphans(_argv: list[str]) -> int:
    """List manifests whose parent.run_id doesn't resolve."""
    from app.manifest_index import get_index
    orphans = get_index().orphans()
    if not orphans:
        print("No orphaned manifests found. Warehouse is clean.")
        return 0
    print(f"Found {len(orphans)} orphan(s):")
    for m in orphans:
        parent = m.parent if hasattr(m, "parent") and m.parent else None
        pid = f"{parent.stage}/{parent.run_id}" if parent else "?"
        print(f"  {m.stage}/{m.run_id}  parent={pid}  (MISSING)")
    return 1


if __name__ == "__main__":
    import sys
    cmds = {
        "backfill-train": _cli_backfill_train,
        "verify": _cli_verify,
        "stale-pending": _cli_stale_pending,
        "orphans": _cli_orphans,
    }
    if len(sys.argv) >= 2 and sys.argv[1] in cmds:
        sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
    print(
        "usage: python -m app.warehouse <command> [args]\n\n"
        "commands:\n"
        "  backfill-train <train_dir> [<train_dir> ...]\n"
        "      Reconstruct manifest.json for an orphan run from its config.yml\n"
        "      and (optionally) pending.json. provenance=\"backfilled\".\n"
        "  verify <manifest.json> [...]\n"
        "      Verify SHA-256 checksums recorded in one or more manifests.\n"
        "  stale-pending\n"
        "      List train runs with pending.json but no manifest (>24h old).\n"
        "  orphans\n"
        "      List manifests whose parent.run_id doesn't resolve."
    )
    sys.exit(2)

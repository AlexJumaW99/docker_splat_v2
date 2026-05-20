"""
Project store.

A project bundles one source (video file or image folder) + its processed
directory + its training runs. State persists to
~/.config/docker_splat_v2_gui/projects.json.

The store is a QObject singleton; views connect to its signals so they stay
in sync when the active project changes.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from pydantic import BaseModel, Field

from app import warehouse as wh
from app.config import (
    CONTAINER_WAREHOUSE,
    CONTAINER_WORKSPACE,
    DATA_DIR,
    PROJECTS_FILE,
    WAREHOUSE_DIR,
)
from app.manifest_index import get_index
from app.paths import unique_path


# Nerfstudio dataparser subcommand name. "nerfstudio-data" expects a
# transforms.json at the dataset root (what ns-process-data produces).
# "colmap" reads a COLMAP sparse reconstruction directly.
DataparserKind = Literal["nerfstudio-data", "colmap"]

# What the Process tab is feeding ns-process-data this run. "imported" means
# the project was created from an already-processed folder and the Process
# tab is gated off entirely.
SourceKind = Literal["video", "images", "imported"]


# Image extensions ns-process-data's `images` subcommand accepts upstream,
# matching nerfstudio.process_data.process_data_utils.allowed_exts. Centralized
# so the New-Project dialog's live image-count readout, the Process tab's
# pre-flight, and any future validator all agree on the same set.
ALLOWED_IMAGE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".cr2",
})

CONVERTIBLE_IMAGE_EXTS: frozenset[str] = frozenset({".heic", ".heif"})


# Conventional location for image folders. Not enforced (a folder anywhere
# under DATA_DIR is technically valid), but the New-Project dialog defaults
# the picker here and surfaces a soft warn when users browse elsewhere.
INPUT_IMAGES_DIR = DATA_DIR / "input_images"


def count_images(folder: Path) -> tuple[int, set[str]]:
    """Return (count, extensions_seen) for files under `folder` whose suffix
    matches ALLOWED_IMAGE_EXTS (case-insensitive). Returns (0, set()) for a
    missing folder so callers can treat "not yet created" and "empty" the
    same way."""
    if not folder.is_dir():
        return 0, set()
    seen: set[str] = set()
    n = 0
    for p in folder.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in ALLOWED_IMAGE_EXTS:
            n += 1
            seen.add(suf)
    return n, seen


_log = logging.getLogger(__name__)


def count_heic_images(folder: Path) -> int:
    """Return the number of HEIC/HEIF files in *folder* that have no
    corresponding JPEG already (i.e. files that still need conversion)."""
    if not folder.is_dir():
        return 0
    n = 0
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in CONVERTIBLE_IMAGE_EXTS:
            jpg = p.with_suffix(".jpg")
            if not jpg.exists():
                n += 1
    return n


def convert_heic_images(folder: Path) -> int:
    """Convert every HEIC/HEIF file in *folder* to a maximum-quality JPEG,
    preserving EXIF metadata (critical for COLMAP focal-length extraction).

    Skips files that already have a corresponding .jpg. After conversion,
    originals are moved into a ``heic_originals/`` subdirectory so the
    input folder only contains clean JPEGs for ns-process-data."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()

    originals: list[Path] = []
    converted = 0
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in CONVERTIBLE_IMAGE_EXTS:
            continue
        dest = p.with_suffix(".jpg")
        if dest.exists():
            originals.append(p)
            _log.debug("skip %s — %s already exists", p.name, dest.name)
            continue
        img = Image.open(p)
        exif = img.info.get("exif", b"")
        save_kw: dict = {"quality": 100, "subsampling": 0}
        if exif:
            save_kw["exif"] = exif
        img.save(dest, "JPEG", **save_kw)
        _log.info("converted %s → %s", p.name, dest.name)
        originals.append(p)
        converted += 1

    if originals:
        stash = folder / "heic_originals"
        stash.mkdir(exist_ok=True)
        for p in originals:
            p.rename(stash / p.name)
        _log.info("moved %d HEIC original(s) to %s", len(originals), stash)
    return converted


def source_kind(project: "Project") -> SourceKind:
    """Determine which Process-tab flow a project belongs to.

    Derived rather than stored so it can never drift from the underlying
    fields. video_path and images_dir are mutually exclusive (enforced at
    create time in `create()` / `create_from_images()`). create_from_processed
    sets neither, yielding "imported"."""
    if project.video_path is not None:
        return "video"
    if project.images_dir is not None:
        return "images"
    return "imported"


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_-]+", "_", text)
    return text or "project"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Candidate relative paths to the COLMAP model dir (the folder that holds
# cameras.bin / images.bin / points3D.bin), in preference order. "sparse/0" is
# the Tanks & Temples / plain-COLMAP layout; "colmap/sparse/0" is what
# ns-process-data produces and what Nerfstudio's ColmapDataParserConfig
# defaults to. Order matters when both happen to exist — first hit wins.
_COLMAP_PATH_CANDIDATES = ("sparse/0", "colmap/sparse/0")


def detect_colmap_path(processed_dir: Path) -> str | None:
    """Return the posix-relative path to the COLMAP model dir inside
    `processed_dir`, or None if no recognized COLMAP layout is present."""
    for rel in _COLMAP_PATH_CANDIDATES:
        if (processed_dir / rel / "cameras.bin").is_file():
            return rel
    return None


def detect_dataset_format(processed_dir: Path) -> DataparserKind | None:
    """Identify which Nerfstudio dataparser can train on this folder.

    Returns the dataparser name, or None if the folder holds no format we
    recognize. Check order matters: transforms.json wins if both are present,
    since ns-process-data writes it alongside sparse/ for COLMAP-sourced runs.
    """
    if (processed_dir / "transforms.json").is_file():
        return "nerfstudio-data"
    if detect_colmap_path(processed_dir) is not None:
        return "colmap"
    return None


class Project(BaseModel):
    id: str
    name: str
    # Source of the dataset. `video_path` and `images_dir` are mutually
    # exclusive — set exactly one, both None for an imported project. The
    # legacy schema only had `video_path`, so existing records load unchanged
    # (images_dir defaults to None, source_kind() returns "video").
    video_path: str | None = None
    images_dir: str | None = None
    processed_dir: str  # absolute host path, must live under DATA_DIR
    # Which ns-train dataparser this project's processed_dir uses. Defaults
    # to nerfstudio-data so legacy records (pre-import-flow) load unchanged.
    dataparser: DataparserKind = "nerfstudio-data"
    # Only meaningful when dataparser == "colmap": posix-relative path from
    # processed_dir to the folder holding cameras.bin (e.g. "sparse/0"). This
    # gets forwarded as `colmap --colmap-path <colmap_path>` at train time so
    # Nerfstudio doesn't fall back to its default of "colmap/sparse/0".
    colmap_path: str | None = None
    created_at: str = Field(default_factory=_iso_now)
    # Per-project parameter presets (form values keyed by preset name)
    presets_process: dict[str, dict] = Field(default_factory=dict)
    presets_train: dict[str, dict] = Field(default_factory=dict)
    last_process_params: dict | None = None
    last_train_params: dict | None = None
    # ULID of the most recent successful TrainManifest under
    # data_v2/trained/<ULID>/. Set by the Train tab on a 0-exit run, alongside
    # the manifest write. Lets the Exports tab (once wired) jump straight to
    # the run dir without filesystem-walking data_v2/trained/.
    last_train_run_id: str | None = None


class _Root(BaseModel):
    projects: list[Project] = Field(default_factory=list)
    active_project_id: str | None = None


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _host_to_container(path: Path) -> str | None:
    """Map a host path to its container equivalent, or None if outside both
    DATA_DIR (/workspace) and WAREHOUSE_DIR (/warehouse)."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(WAREHOUSE_DIR.resolve())
        return f"{CONTAINER_WAREHOUSE}/{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(DATA_DIR.resolve())
        return f"{CONTAINER_WORKSPACE}/{rel.as_posix()}"
    except ValueError:
        return None


def _docker_rm(container_path: str) -> bool:
    """``docker exec <container> rm -rf <path>`` — returns True on success."""
    import subprocess
    from app.config import CONTAINER_NAME
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        r = subprocess.run(
            [docker, "exec", CONTAINER_NAME, "rm", "-rf", container_path],
            capture_output=True, text=True, timeout=60.0,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _rm_tree(path: Path) -> bool:
    """Remove a directory, using docker exec for root-owned files.

    Returns True if the directory was successfully removed (or didn't exist).
    """
    if not path.is_dir():
        return True
    c_path = _host_to_container(path)
    if c_path and _docker_rm(c_path):
        _log.info("deleted (via container): %s", path)
        return True
    try:
        shutil.rmtree(path)
        _log.info("deleted: %s", path)
        return True
    except OSError as e:
        _log.warning("failed to delete %s: %s", path, e)
        return False


def _rm_file(path: Path) -> None:
    """Remove a single file, using docker exec for root-owned files."""
    if not path.is_file():
        return
    try:
        path.unlink()
        _log.info("deleted: %s", path)
    except PermissionError:
        c_path = _host_to_container(path)
        if c_path and _docker_rm(c_path):
            _log.info("deleted (via container): %s", path)
        else:
            _log.warning("failed to delete %s: permission denied", path)
    except OSError as e:
        _log.warning("failed to delete %s: %s", path, e)


@dataclass
class ProjectDataScan:
    """Result of scanning the filesystem for all data tied to a project.

    Carries both legacy (`data/`) artifacts and warehouse (`data_v2/`)
    artifacts. The fields without a `warehouse_` prefix refer to legacy
    layout; the warehouse fields each hold one ULID directory per artifact
    (which itself contains `manifest.json` + the payload(s))."""
    processed_dirs: list[Path] = field(default_factory=list)
    outputs_dir: Path | None = None
    render_dirs: list[Path] = field(default_factory=list)
    # Warehouse-keyed artifacts whose lineage (TrainManifest.parent.run_id /
    # ExportManifest.parent.run_id / RenderManifest.parent.run_id) descends
    # from this project's processed ULID.
    warehouse_train_dirs: list[Path] = field(default_factory=list)
    warehouse_export_dirs: list[Path] = field(default_factory=list)
    warehouse_render_dirs: list[Path] = field(default_factory=list)
    warehouse_source_dirs: list[Path] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def has_data(self) -> bool:
        return bool(
            self.processed_dirs or self.outputs_dir or self.render_dirs
            or self.warehouse_train_dirs or self.warehouse_export_dirs
            or self.warehouse_render_dirs or self.warehouse_source_dirs
        )

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.processed_dirs:
            names = ", ".join(p.name for p in self.processed_dirs)
            lines.append(f"Processed datasets ({len(self.processed_dirs)}): {names}")
        if self.outputs_dir:
            methods = [d.name for d in self.outputs_dir.iterdir() if d.is_dir()] \
                if self.outputs_dir.is_dir() else []
            lines.append(f"Training outputs: {self.outputs_dir.name}/"
                         + (f" ({', '.join(methods)})" if methods else ""))
        if self.render_dirs:
            names = ", ".join(p.name for p in self.render_dirs)
            lines.append(f"Rendered videos ({len(self.render_dirs)}): {names}")
        if self.warehouse_train_dirs:
            lines.append(
                f"Warehouse train runs ({len(self.warehouse_train_dirs)}): "
                + ", ".join(p.name[:8] + "…" for p in self.warehouse_train_dirs)
            )
        if self.warehouse_export_dirs:
            kinds: dict[str, int] = {}
            for d in self.warehouse_export_dirs:
                kinds[d.parent.name] = kinds.get(d.parent.name, 0) + 1
            tally = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
            lines.append(
                f"Warehouse exports ({len(self.warehouse_export_dirs)}): "
                f"{tally}"
            )
        if self.warehouse_render_dirs:
            lines.append(
                f"Warehouse renders ({len(self.warehouse_render_dirs)}): "
                + ", ".join(p.name[:8] + "…" for p in self.warehouse_render_dirs)
            )
        if self.warehouse_source_dirs:
            lines.append(
                f"Warehouse sources ({len(self.warehouse_source_dirs)}): "
                + ", ".join(p.name[:8] + "…" for p in self.warehouse_source_dirs)
            )
        if self.total_bytes:
            lines.append(f"Total size: {_fmt_size(self.total_bytes)}")
        return lines


def _project_warehouse_train_ulids(project: Project) -> set[str]:
    """Return the set of train ULIDs in `data_v2/trained/` whose
    `parent.run_id` matches this project's processed ULID. Empty when the
    project is legacy (processed_dir not under WAREHOUSE_DIR) or when no
    warehouse train runs descend from this project yet."""

    processed = Path(project.processed_dir).resolve()
    try:
        processed.relative_to(WAREHOUSE_DIR.resolve())
    except ValueError:
        return set()
    processed_ulid = processed.name
    if not re.fullmatch(wh.ULID_PATTERN, processed_ulid):
        return set()

    return {m.run_id for m in get_index().trains_for_process(processed_ulid)}


def scan_project_data(project: Project) -> ProjectDataScan:
    """Inventory all filesystem artifacts associated with *project*.

    Walks both the legacy `data/` tree (processed/, outputs/, renders/)
    and the warehouse `data_v2/` tree (trained/, exports/, renders/),
    filtering warehouse artifacts by manifest lineage so we never claim
    another project's exports."""
    scan = ProjectDataScan()
    pid = project.id

    # -- legacy processed dirs: the stored one + any sibling _splat* variants --
    processed_parent = DATA_DIR / "processed"
    seen: set[Path] = set()
    stored = Path(project.processed_dir)
    if stored.is_dir():
        seen.add(stored.resolve())
    if processed_parent.is_dir():
        for d in processed_parent.iterdir():
            if not d.is_dir():
                continue
            if d.name == f"{pid}_splat" or d.name.startswith(f"{pid}_splat_"):
                seen.add(d.resolve())
    scan.processed_dirs = sorted(seen)

    # -- legacy training outputs --
    out = DATA_DIR / "outputs" / pid
    if out.is_dir():
        scan.outputs_dir = out

    # Legacy `data/exports/splat/*.ply` is intentionally not scanned anymore.
    # Per the warehouse migration: those files stay on disk but are no
    # longer surfaced by the GUI; new exports live under data_v2/exports/.

    # -- legacy render dirs (keyed by processed_dir basename) --
    renders_parent = DATA_DIR / "renders"
    if renders_parent.is_dir():
        basenames = {p.name for p in scan.processed_dirs}
        for d in renders_parent.iterdir():
            if d.is_dir() and d.name in basenames:
                scan.render_dirs.append(d)
        scan.render_dirs.sort()

    # -- warehouse train runs descending from this project's processed ULID --
    train_ulids = _project_warehouse_train_ulids(project)
    if train_ulids:
        scan.warehouse_train_dirs = sorted(
            WAREHOUSE_DIR / "trained" / tid for tid in train_ulids
        )

    # -- warehouse exports + renders via ManifestIndex --
    if train_ulids:
        idx = get_index()
        from app.manifest import ExportManifest, RenderManifest
        for tid in train_ulids:
            for em in idx.exports_for_train(tid):
                kind = em.params.kind
                d = WAREHOUSE_DIR / "exports" / kind / em.run_id
                if d.is_dir():
                    scan.warehouse_export_dirs.append(d)
            for rm in idx.renders_for_train(tid):
                d = WAREHOUSE_DIR / "renders" / rm.run_id
                if d.is_dir():
                    scan.warehouse_render_dirs.append(d)
        scan.warehouse_export_dirs.sort()
        scan.warehouse_render_dirs.sort()

    # -- warehouse source dirs (the copied video/images in data_v2/sources/) --
    # Trace from processed → source via parent pointer.  Only include the
    # source dir when no *other* processed manifest also references it (the
    # warehouse deduplicates sources by SHA-256, so two projects using the
    # same video share one source ULID).
    processed = Path(project.processed_dir).resolve()
    try:
        processed.relative_to(WAREHOUSE_DIR.resolve())
        processed_ulid = processed.name
    except ValueError:
        processed_ulid = None
    if processed_ulid and re.fullmatch(wh.ULID_PATTERN, processed_ulid):
        idx = get_index()
        from app.manifest import ProcessManifest
        proc_manifest = idx.get(processed_ulid)
        if isinstance(proc_manifest, ProcessManifest):
            source_ulid = proc_manifest.parent.run_id
            other_refs = [
                m for m in idx.children_of(source_ulid)
                if isinstance(m, ProcessManifest) and m.run_id != processed_ulid
            ]
            if not other_refs:
                src_dir = WAREHOUSE_DIR / "sources" / source_ulid
                if src_dir.is_dir():
                    scan.warehouse_source_dirs.append(src_dir)

    # -- total size --
    total = 0
    for d in scan.processed_dirs:
        total += _dir_size(d)
    if scan.outputs_dir:
        total += _dir_size(scan.outputs_dir)
    for d in scan.render_dirs:
        total += _dir_size(d)
    for d in scan.warehouse_train_dirs:
        total += _dir_size(d)
    for d in scan.warehouse_export_dirs:
        total += _dir_size(d)
    for d in scan.warehouse_render_dirs:
        total += _dir_size(d)
    for d in scan.warehouse_source_dirs:
        total += _dir_size(d)
    scan.total_bytes = total

    return scan


class ProjectStore(QObject):
    projects_changed = Signal()            # list changed
    active_changed = Signal(object)        # Project | None

    def __init__(self, path: Path = PROJECTS_FILE, parent: QObject | None = None):
        super().__init__(parent)
        self._path = path
        self._root = self._load()
        if self._heal():
            self._save()

    # ---- persistence ----
    def _load(self) -> _Root:
        if not self._path.exists():
            return _Root()
        try:
            return _Root.model_validate_json(self._path.read_text())
        except Exception:
            backup = self._path.with_suffix(".corrupt.json")
            try:
                self._path.rename(backup)
            except OSError:
                pass
            return _Root()

    def _heal(self) -> bool:
        """Migrate records written before newer fields existed. Returns True
        if anything was changed (the caller then persists). Currently: COLMAP
        projects missing colmap_path get it back-filled from the filesystem so
        ns-train's --colmap-path flag can be emitted correctly."""
        changed = False
        for p in self._root.projects:
            if p.dataparser == "colmap" and not p.colmap_path:
                found = detect_colmap_path(Path(p.processed_dir))
                if found is not None:
                    p.colmap_path = found
                    changed = True
        return changed

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(self._root.model_dump_json(indent=2))
        tmp.replace(self._path)

    # ---- read ----
    @property
    def projects(self) -> list[Project]:
        return list(self._root.projects)

    @property
    def active(self) -> Project | None:
        if self._root.active_project_id is None:
            return None
        for p in self._root.projects:
            if p.id == self._root.active_project_id:
                return p
        return None

    def get(self, project_id: str) -> Project | None:
        for p in self._root.projects:
            if p.id == project_id:
                return p
        return None

    # ---- mutate ----
    def set_active(self, project_id: str | None) -> None:
        self._root.active_project_id = project_id
        self._save()
        self.active_changed.emit(self.active)

    def _reserve_pid(self, name: str) -> str:
        """Resolve a unique project id for `name`. Raises if the name is taken."""
        if any(p.name == name for p in self._root.projects):
            raise ValueError(f"A project named '{name}' already exists.")
        base = _slugify(name)
        pid = base
        i = 2
        taken = {p.id for p in self._root.projects}
        while pid in taken:
            pid = f"{base}_{i}"
            i += 1
        return pid

    def _insert(self, project: Project) -> Project:
        self._root.projects.append(project)
        self._root.active_project_id = project.id
        self._save()
        self.projects_changed.emit()
        self.active_changed.emit(project)
        return project

    def create(self, name: str, video_path: Path) -> Project:
        """Creates a project from a source video. Raises ValueError if the name
        conflicts or the video is outside the data/ bind mount."""
        name = name.strip()
        if not name:
            raise ValueError("Project name can't be empty.")

        try:
            video_path.resolve().relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"Video must live inside {DATA_DIR}.\n\n"
                "The container only sees that folder (it's bind-mounted as /workspace)."
            ) from None

        pid = self._reserve_pid(name)
        # unique_path covers the edge case where a previous project under the
        # same slug was deleted from the store but its processed/<pid>_splat
        # folder is still on disk — the new project gets a clean _2/_3 suffix
        # so it never adopts stale data without consent.
        processed_dir = unique_path(DATA_DIR / "processed", f"{pid}_splat")
        return self._insert(Project(
            id=pid,
            name=name,
            video_path=str(video_path.resolve()),
            processed_dir=str(processed_dir),
            dataparser="nerfstudio-data",
        ))

    def create_from_images(self, name: str, images_dir: Path) -> Project:
        """Creates a project from a folder of images. Raises ValueError if the
        name conflicts, the folder is outside the data/ bind mount, doesn't
        exist, or contains no supported image files."""
        name = name.strip()
        if not name:
            raise ValueError("Project name can't be empty.")

        resolved = images_dir.resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory:\n{resolved}")

        try:
            resolved.relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"Image folder must live inside {DATA_DIR}.\n\n"
                "The container only sees that folder (it's bind-mounted as /workspace)."
            ) from None

        n, _exts = count_images(resolved)
        n_heic = count_heic_images(resolved)
        if n == 0 and n_heic == 0:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
            raise ValueError(
                f"No supported images found in:\n{resolved}\n\n"
                f"Accepted extensions: {allowed}"
            )

        if any(
            p.images_dir and Path(p.images_dir).resolve() == resolved
            for p in self._root.projects
        ):
            raise ValueError(
                "Another project is already pointed at this image folder."
            )

        pid = self._reserve_pid(name)
        processed_dir = unique_path(DATA_DIR / "processed", f"{pid}_splat")
        return self._insert(Project(
            id=pid,
            name=name,
            images_dir=str(resolved),
            processed_dir=str(processed_dir),
            dataparser="nerfstudio-data",
        ))

    def create_from_processed(self, name: str, processed_dir: Path) -> Project:
        """Creates a project from an already-processed dataset folder.

        The folder must live under DATA_DIR (so the container can see it) and
        hold a recognized dataparser format. No video is recorded — the Process
        tab will render a read-only "imported" state for these projects.
        """
        name = name.strip()
        if not name:
            raise ValueError("Project name can't be empty.")

        resolved = processed_dir.resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory:\n{resolved}")

        try:
            resolved.relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"Processed folder must live inside {DATA_DIR}.\n\n"
                "The container only sees that folder (it's bind-mounted as /workspace)."
            ) from None

        fmt = detect_dataset_format(resolved)
        if fmt is None:
            raise ValueError(
                f"Couldn't detect a supported dataset format in:\n{resolved}\n\n"
                "Expected either transforms.json (Nerfstudio) "
                "or sparse/0/cameras.bin (COLMAP)."
            )

        if any(Path(p.processed_dir).resolve() == resolved for p in self._root.projects):
            raise ValueError(
                "Another project is already pointed at this folder."
            )

        colmap_path = detect_colmap_path(resolved) if fmt == "colmap" else None

        pid = self._reserve_pid(name)
        return self._insert(Project(
            id=pid,
            name=name,
            video_path=None,
            processed_dir=str(resolved),
            dataparser=fmt,
            colmap_path=colmap_path,
        ))

    def update_video_path(self, project_id: str, video_path: Path) -> None:
        try:
            video_path.resolve().relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(f"Video must live inside {DATA_DIR}.") from None
        p = self.get(project_id)
        if p is None:
            return
        p.video_path = str(video_path.resolve())
        self._save()
        self.active_changed.emit(self.active)

    def update_images_dir(self, project_id: str, images_dir: Path) -> None:
        resolved = images_dir.resolve()
        try:
            resolved.relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(f"Image folder must live inside {DATA_DIR}.") from None
        if not resolved.is_dir():
            raise ValueError(f"Not a directory:\n{resolved}")
        n, _ = count_images(resolved)
        n_heic = count_heic_images(resolved)
        if n == 0 and n_heic == 0:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
            raise ValueError(
                f"No supported images in:\n{resolved}\n\nAccepted: {allowed}"
            )
        p = self.get(project_id)
        if p is None:
            return
        p.images_dir = str(resolved)
        self._save()
        self.active_changed.emit(self.active)

    def update_processed_dir(self, project_id: str, processed_dir: Path) -> None:
        """Repoint a project at a new processed folder. Used by the Process
        tab after a successful run — see views/process.py:_on_job_finished.

        Old processed folders are not deleted; downstream artifacts (training
        runs in outputs/, rendered videos in renders/) keep the literal path
        baked into their config.yml so they remain functional in isolation.

        Accepts paths under either DATA_DIR (legacy `data/`, bind-mounted at
        /workspace) or WAREHOUSE_DIR (`data_v2/`, bind-mounted at /warehouse).
        New runs from the manifest-aware Process tab land in WAREHOUSE_DIR;
        legacy projects continue to point into DATA_DIR until rebuilt."""
        resolved = processed_dir.resolve()
        roots = (DATA_DIR.resolve(), WAREHOUSE_DIR.resolve())
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError(
                f"Processed folder must live inside {DATA_DIR} or {WAREHOUSE_DIR}."
            )
        p = self.get(project_id)
        if p is None:
            return
        p.processed_dir = str(processed_dir.resolve())
        self._save()
        self.active_changed.emit(self.active)

    def update_last_process_params(self, project_id: str, params: dict) -> None:
        p = self.get(project_id)
        if p is None:
            return
        p.last_process_params = params
        self._save()

    def update_last_train_params(self, project_id: str, params: dict) -> None:
        p = self.get(project_id)
        if p is None:
            return
        p.last_train_params = params
        self._save()

    def update_last_train_run_id(self, project_id: str, run_id: str) -> None:
        """Record the ULID of the most recent successful training run.
        Called by the Train tab's success handler right after the
        TrainManifest is persisted. See Project.last_train_run_id."""
        p = self.get(project_id)
        if p is None:
            return
        p.last_train_run_id = run_id
        self._save()

    def save_preset(self, project_id: str, kind: str, name: str, params: dict) -> None:
        """kind: 'process' | 'train'"""
        p = self.get(project_id)
        if p is None:
            return
        store = p.presets_process if kind == "process" else p.presets_train
        store[name] = params
        self._save()

    def delete(
        self, project_id: str, delete_data: bool = False
    ) -> list[Path]:
        """Remove a project from the store, optionally deleting all its data.

        Data created by ns-process-data / ns-train inside the container is
        owned by root. We delete via ``docker exec rm -rf`` when the container
        is running (same privilege that created the files). Falls back to
        ``shutil.rmtree`` for user-owned dirs or when the container is down.

        Returns a (possibly empty) list of directories that could not be
        removed — typically root-owned dirs when the container is not running.
        """
        p = self.get(project_id)
        if p is None:
            raise ValueError(f"No project with id '{project_id}'.")

        failed: list[Path] = []
        if delete_data:
            scan = scan_project_data(p)
            dirs = list(scan.processed_dirs)
            if scan.outputs_dir:
                dirs.append(scan.outputs_dir)
            dirs.extend(scan.render_dirs)
            dirs.extend(scan.warehouse_train_dirs)
            dirs.extend(scan.warehouse_export_dirs)
            dirs.extend(scan.warehouse_render_dirs)
            dirs.extend(scan.warehouse_source_dirs)
            for d in dirs:
                if not _rm_tree(d):
                    failed.append(d)

        self._root.projects = [
            proj for proj in self._root.projects if proj.id != project_id
        ]
        was_active = self._root.active_project_id == project_id
        if was_active:
            self._root.active_project_id = (
                self._root.projects[0].id if self._root.projects else None
            )
        self._save()
        self.projects_changed.emit()
        if was_active:
            self.active_changed.emit(self.active)
        return failed

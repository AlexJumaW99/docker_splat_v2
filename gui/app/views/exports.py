"""
Exports view — runs ns-export gaussian-splat and ns-render camera-path inside
splat-ns, writing every artifact into the ULID-keyed warehouse with a
manifest that records lineage back to the train run.

Two artifacts per trained run, both warehouse-keyed and manifest-tracked:

* `.ply` splat — exported under
  `data_v2/exports/splat/<EXPORT_ULID>/<stem>.ply` alongside a
  `manifest.json` whose `parent.run_id` is the train ULID. ns-export
  always writes `splat.ply` into its --output-dir; we rename to the
  user's stem inside the same ULID dir so the on-disk name is
  human-meaningful while lineage is encoded in the manifest, not the
  filename. The `mesh/` and `pointcloud/` sub-folders are reserved by
  the warehouse layout but not wired up here — `ns-export
  poisson/tsdf/marching-cubes/pointcloud` all require the volumetric-NeRF
  model interface that `SplatfactoModel` doesn't implement.
* `.mp4` video — rendered under `data_v2/renders/<RENDER_ULID>/<stem>.mp4`
  with its own `manifest.json` (parent = train ULID). A snapshot of the
  camera-path JSON used is dropped into the same dir for reproducibility.

Both cards share the same run picker. Only one JobManager job runs at a
time, so a single Cancel button + status pill covers both.
"""
from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app import warehouse as wh
from app.config import CONTAINER_NAME, WAREHOUSE_DIR
from app.jobs import JobManager, JobSpec
from app.manifest import ExportParams, RenderParams, TrainManifest
from app.manifest_index import get_index
from app.store import Project, ProjectStore
from app.widgets import HelpBadge, ProjectBar, card, pill, section_title, set_pill


_HELP = {
    "run": (
        "Training run",
        "A finished ns-train run from this project under "
        "<code>data_v2/trained/&lt;ULID&gt;/</code>. Latest runs appear first; "
        "pick the run you want to export a .ply from or render a video of. "
        "Each run's <code>manifest.json</code> encodes the full lineage chain "
        "back to the source video."
    ),
    "ply_name": (
        "Export filename",
        "Name for the exported <code>.ply</code>. Lands at "
        "<code>data_v2/exports/splat/&lt;EXPORT_ULID&gt;/&lt;name&gt;.ply</code> "
        "next to a <code>manifest.json</code> recording lineage to the train run. "
        "Each export gets its own ULID directory, so collisions can't happen — "
        "the name is purely for human readability."
    ),
    "camera_path": (
        "Camera path",
        "A keyframe path saved from viser's keyframe editor, stored as "
        "<code>&lt;processed_dir&gt;/camera_paths/*.json</code>. Open the run "
        "in viser, build a path, click <b>Save</b>, then hit ↻ here to "
        "rescan. The render uses the path verbatim, and a copy of the JSON is "
        "snapshotted next to the rendered video for reproducibility."
    ),
    "video_name": (
        "Render filename",
        "Name for the rendered <code>.mp4</code>. Lands at "
        "<code>data_v2/renders/&lt;RENDER_ULID&gt;/&lt;name&gt;.mp4</code> with "
        "a <code>manifest.json</code> whose <code>parent.run_id</code> is the "
        "train ULID. Each render gets its own ULID directory — no collisions."
    ),
}


def _hb(key: str) -> HelpBadge:
    title, body = _HELP[key]
    return HelpBadge(title, body)


# splatfacto-w / -light need a different exporter because SplatfactoWModel
# doesn't subclass SplatfactoModel and computes SH coefficients per-frame
# from a learned appearance embedding. The image ships a custom CLI
# `ns-export-splatfactow` that bakes the average appearance embedding into
# each gaussian and writes an inria-format .ply in the same splat.ply layout.
_SPLATFACTOW_METHODS = frozenset({"splatfacto-w", "splatfacto-w-light"})


# ---------- path helpers ----------


def _sanitize_stem(raw: str, ext: str) -> str:
    """Strip a trailing extension the user may have typed, drop path
    separators, and trim whitespace. We only ever write into a fresh ULID
    directory, so slashes are never meaningful here."""
    s = raw.strip()
    if ext and s.lower().endswith(ext.lower()):
        s = s[:-len(ext)]
    s = s.replace("/", "_").replace("\\", "_")
    return s.strip()


# ---------- training run discovery ----------

@dataclass(frozen=True)
class _Run:
    """One trained warehouse run this project can export from. Identity is
    the train ULID; the dropdown label includes a short prefix so two runs
    with the same method + timestamp (rare same-minute re-train) stay
    distinguishable."""

    train_run_id: str        # ULID under data_v2/trained/
    method: str              # e.g. "splatfacto-big" or "splatfacto-mcmc"
    timestamp_dir: str       # e.g. "2026-04-20_020610"
    display_name: str        # TrainManifest.display_name, seeds the default stem
    config_path: Path        # absolute host path to config.yml

    @property
    def label(self) -> str:
        return (
            f"{self.method} · {self.timestamp_dir}  "
            f"({self.train_run_id[:8]}…)"
        )

    @property
    def container_config(self) -> str:
        return wh.host_to_container_warehouse(self.config_path)


# upstream nerfstudio writes both `splatfacto` and `splatfacto-mcmc` runs
# under the same `splatfacto` method dir, differing only by `strategy:`
# inside config.yml. We peek so the dropdown labels mcmc runs accurately.
_STRATEGY_RE = re.compile(r"^\s*strategy:\s*([A-Za-z_-]+)\s*$", re.MULTILINE)


def _method_from_config(declared: str, config_path: Path) -> str:
    """Refine a TrainManifest.params.method against config.yml's `strategy:`
    field — only matters for the splatfacto / splatfacto-mcmc split."""
    if declared != "splatfacto":
        return declared
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return declared
    m = _STRATEGY_RE.search(text)
    if m and m.group(1) == "mcmc":
        return "splatfacto-mcmc"
    return declared


def _project_processed_ulid(project: Project) -> str | None:
    """Return the ULID of the project's processed run when its `processed_dir`
    lives under WAREHOUSE_DIR; None for legacy projects whose processed
    folder predates the warehouse. The Exports tab uses None to render a
    gated empty state instead of a broken run dropdown."""
    processed = Path(project.processed_dir).resolve()
    try:
        processed.relative_to(WAREHOUSE_DIR.resolve())
    except ValueError:
        return None
    name = processed.name
    if not re.fullmatch(wh.ULID_PATTERN, name):
        return None
    return name


def _discover_runs(project: Project) -> list[_Run]:
    """Query the ManifestIndex for TrainManifests whose parent.run_id matches
    this project's processed ULID. Sorted newest-first (ULID lexicographic
    sort = chronological sort). Legacy projects return []."""

    parent_id = _project_processed_ulid(project)
    if parent_id is None:
        return []

    runs: list[_Run] = []
    for m in get_index().trains_for_process(parent_id):
        cfg_rel = m.outputs.get("config_yml")
        cfg_path: Path | None = None
        if cfg_rel:
            cand = WAREHOUSE_DIR / cfg_rel
            if cand.is_file():
                cfg_path = cand
        if cfg_path is None:
            sub = WAREHOUSE_DIR / "trained" / m.run_id
            nested = wh.find_train_run_subdir(sub)
            if nested is not None:
                cand = nested / "config.yml"
                if cand.is_file():
                    cfg_path = cand
        if cfg_path is None:
            continue

        runs.append(_Run(
            train_run_id=m.run_id,
            method=_method_from_config(m.params.method, cfg_path),
            timestamp_dir=m.params.timestamp_dir,
            display_name=m.display_name or m.run_id,
            config_path=cfg_path,
        ))

    runs.sort(key=lambda r: r.train_run_id, reverse=True)
    return runs


def _default_ply_stem(run: _Run) -> str:
    """Default filename stem for an export. Mirrors the legacy convention
    (display_name + method + timestamp) so files copied out of the warehouse
    remain self-describing even without the manifest."""
    return f"{run.display_name}_{run.method}_{run.timestamp_dir}"


# ---------- camera-path discovery ----------

def _camera_paths_dir(project: Project) -> Path:
    return Path(project.processed_dir) / "camera_paths"


def _discover_camera_paths(project: Project) -> list[Path]:
    """Return saved camera paths for `project`, newest first (by mtime).
    Viser writes these into <processed_dir>/camera_paths/*.json when the
    user hits 'save' in the keyframe editor."""
    cp_dir = _camera_paths_dir(project)
    if not cp_dir.is_dir():
        return []
    paths = [p for p in cp_dir.glob("*.json") if p.is_file()]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


# ---------- empty states ----------

class _EmptyState(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        title = QLabel("No project yet")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel(
            "Exporting needs a project with at least one training run. "
            "Create a project and train it first."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(440)

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addStretch(1)


class _NoRunsState(QWidget):
    """Active project has no warehouse-keyed train runs to expose. Two cases:
    (a) genuinely no runs yet, or (b) a legacy project whose processed
    folder predates the warehouse migration. Both route the user to Train."""

    request_train = Signal()

    def __init__(self, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        title = QLabel("No trained runs yet")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        is_legacy = _project_processed_ulid(project) is None
        if is_legacy:
            body = (
                f"'{project.name}' is pointed at a legacy processed folder:\n"
                f"{project.processed_dir}\n\n"
                "Exports now require warehouse-keyed train runs so lineage "
                "can be traced via manifest.json. Reprocess via the Process "
                "tab, then retrain to make the run exportable."
            )
        else:
            body = (
                f"'{project.name}' has no finished training runs under\n"
                f"{WAREHOUSE_DIR / 'trained'}/ with this project's "
                f"processed ULID.\n\n"
                "Train a model first, then come back to export its splat."
            )
        sub = QLabel(body)
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(620)

        btn = QPushButton("Go to Train →")
        btn.setProperty("role", "primary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.request_train.emit)

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        v.addWidget(btn, 0, Qt.AlignHCenter)
        v.addStretch(1)


# ---------- form ----------

class _Form(QWidget):
    """Per-project export form. Rebuilt when the active project changes."""

    def __init__(
        self, store: ProjectStore, jobs: JobManager,
        project: Project, runs: list[_Run],
    ):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs
        self._project = project
        self._runs = runs
        self._camera_paths: list[Path] = []
        # "export" (ply) | "render" (video) | None — tracks which of our
        # buttons owns the currently running job, for cancel / pill routing.
        self._running_kind: str | None = None

        # Pending-state for the manifest write that fires on 0-exit. Captured
        # at job-start time so the on-finish handler doesn't have to re-derive
        # ULIDs, paths, params, or commands from UI state that may have
        # changed mid-run. Mirrors the pattern in views/train.py.
        self._pending_run: _Run | None = None
        self._pending_export_ulid: str | None = None
        self._pending_export_dir: Path | None = None
        self._pending_export_payload: str | None = None   # filename in the ULID dir
        self._pending_export_params: ExportParams | None = None
        self._pending_export_command: str | None = None
        self._pending_render_ulid: str | None = None
        self._pending_render_dir: Path | None = None
        self._pending_render_payload: str | None = None
        self._pending_render_params: RenderParams | None = None
        self._pending_render_command: str | None = None
        self._pending_start_time: datetime | None = None
        self._pending_display_name: str | None = None

        self._build()
        self._select_run(0)
        self._refresh_camera_paths()
        self._wire_jobs()

        # Catch up to any job already running at construction time.
        if self._jobs.is_running() and self._jobs.current is not None:
            self._on_job_started(self._jobs.current)

    # ---- build ----
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget(objectName="Canvas")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(32, 24, 32, 24)
        bl.setSpacing(14)

        title = QLabel("Exports")
        title.setObjectName("PageTitle")
        sub = QLabel(
            "Export a trained run as a .ply splat, or render a video from a "
            "saved viser keyframe path. Outputs land in the ULID-keyed "
            "warehouse under data_v2/exports/ and data_v2/renders/, each with "
            "a manifest.json recording lineage back to the train run."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        header = QVBoxLayout(); header.setSpacing(2)
        header.addWidget(title); header.addWidget(sub)
        bl.addLayout(header)

        bl.addWidget(self._run_card())
        bl.addWidget(self._ply_card())
        bl.addWidget(self._video_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- run card (shared) ----
    def _run_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        run_hdr = QHBoxLayout(); run_hdr.setSpacing(6); run_hdr.setContentsMargins(0, 0, 0, 0)
        run_hdr.addWidget(section_title("TRAINING RUN"))
        run_hdr.addWidget(_hb("run"))
        run_hdr.addStretch(1)
        run_wrap = QWidget(); run_wrap.setLayout(run_hdr)
        v.addWidget(run_wrap)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        for r in self._runs:
            self.run_combo.addItem(r.label)
        self.run_combo.currentIndexChanged.connect(self._select_run)
        row.addWidget(QLabel("Run"))
        row.addWidget(self.run_combo, 1)
        v.addLayout(row)

        self.config_line = QLineEdit()
        self.config_line.setReadOnly(True)
        self.config_line.setProperty("role", "readout")
        v.addWidget(self.config_line)

        caption = QLabel(
            "Host path to the run's config.yml. "
            "Latest runs appear first; pick an older one to export it instead."
        )
        caption.setProperty("role", "caption")
        caption.setWordWrap(True)
        v.addWidget(caption)
        return c

    # ---- ply card ----
    def _ply_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        ply_hdr = QHBoxLayout(); ply_hdr.setSpacing(6); ply_hdr.setContentsMargins(0, 0, 0, 0)
        ply_hdr.addWidget(section_title("EXPORT .PLY"))
        ply_hdr.addWidget(_hb("ply_name"))
        ply_hdr.addStretch(1)
        ply_wrap = QWidget(); ply_wrap.setLayout(ply_hdr)
        v.addWidget(ply_wrap)

        name_row = QHBoxLayout(); name_row.setSpacing(8)
        self.ply_name_edit = QLineEdit()
        self.ply_name_edit.setPlaceholderText("filename (without .ply)")
        self.ply_name_edit.textChanged.connect(self._refresh_ply_target)
        ext = QLabel(".ply"); ext.setProperty("role", "caption")
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self.ply_name_edit, 1)
        name_row.addWidget(ext)
        v.addLayout(name_row)

        self.ply_target_line = QLineEdit()
        self.ply_target_line.setReadOnly(True)
        self.ply_target_line.setProperty("role", "readout")
        v.addWidget(self.ply_target_line)

        self.ply_target_caption = QLabel("")
        self.ply_target_caption.setProperty("role", "caption")
        self.ply_target_caption.setWordWrap(True)
        v.addWidget(self.ply_target_caption)

        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export .ply")
        self.export_btn.setProperty("role", "primary")
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self._on_export_ply)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return c

    # ---- video card ----
    def _video_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        vid_hdr = QHBoxLayout(); vid_hdr.setSpacing(6); vid_hdr.setContentsMargins(0, 0, 0, 0)
        vid_hdr.addWidget(section_title("RENDER VIDEO"))
        vid_hdr.addWidget(_hb("camera_path"))
        vid_hdr.addStretch(1)
        vid_wrap = QWidget(); vid_wrap.setLayout(vid_hdr)
        v.addWidget(vid_wrap)

        cp_row = QHBoxLayout(); cp_row.setSpacing(8)
        self.cp_combo = QComboBox()
        self.cp_combo.setMinimumWidth(280)
        self.cp_combo.currentIndexChanged.connect(self._on_camera_path_selected)

        self.cp_refresh_btn = QPushButton("↻")
        self.cp_refresh_btn.setToolTip("Rescan camera_paths/")
        self.cp_refresh_btn.setProperty("role", "ghost")
        self.cp_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.cp_refresh_btn.setFixedWidth(30)
        self.cp_refresh_btn.clicked.connect(self._refresh_camera_paths)

        cp_row.addWidget(QLabel("Camera path"))
        cp_row.addWidget(self.cp_combo, 1)
        cp_row.addWidget(self.cp_refresh_btn)
        v.addLayout(cp_row)

        self.cp_source_line = QLineEdit()
        self.cp_source_line.setReadOnly(True)
        self.cp_source_line.setProperty("role", "readout")
        v.addWidget(self.cp_source_line)

        self.cp_caption = QLabel("")
        self.cp_caption.setProperty("role", "caption")
        self.cp_caption.setWordWrap(True)
        v.addWidget(self.cp_caption)

        name_row = QHBoxLayout(); name_row.setSpacing(8)
        self.video_name_edit = QLineEdit()
        self.video_name_edit.setPlaceholderText("filename (without .mp4)")
        self.video_name_edit.textChanged.connect(self._refresh_video_target)
        ext = QLabel(".mp4"); ext.setProperty("role", "caption")
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self.video_name_edit, 1)
        name_row.addWidget(ext)
        v.addLayout(name_row)

        self.video_target_line = QLineEdit()
        self.video_target_line.setReadOnly(True)
        self.video_target_line.setProperty("role", "readout")
        v.addWidget(self.video_target_line)

        self.video_target_caption = QLabel("")
        self.video_target_caption.setProperty("role", "caption")
        self.video_target_caption.setWordWrap(True)
        v.addWidget(self.video_target_caption)

        btn_row = QHBoxLayout()
        self.render_btn = QPushButton("Render video")
        self.render_btn.setProperty("role", "primary")
        self.render_btn.setCursor(Qt.PointingHandCursor)
        self.render_btn.clicked.connect(self._on_render_video)
        btn_row.addWidget(self.render_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return c

    # ---- actions (shared) ----
    def _actions_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        row = QHBoxLayout(); row.setSpacing(10)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("role", "secondary")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)

        self.status_pill = pill("Ready", "neutral")

        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.status_pill)
        v.addLayout(row)
        return c

    # ---- run selection ----
    def _current_run(self) -> _Run:
        return self._runs[self.run_combo.currentIndex()]

    def _select_run(self, idx: int) -> None:
        if not (0 <= idx < len(self._runs)):
            return
        r = self._runs[idx]
        self.config_line.setText(str(r.config_path))
        # Setting text fires textChanged → refresh previews.
        self.ply_name_edit.setText(_default_ply_stem(r))

    # ---- ply target preview ----
    def _refresh_ply_target(self, _text: str = "") -> None:
        stem = _sanitize_stem(self.ply_name_edit.text(), ".ply")
        if not stem:
            stem = _default_ply_stem(self._current_run())
        # We can't know the export ULID until the user actually clicks Export,
        # so the preview shows the parent dir with a placeholder. The actual
        # target is logged to the output drawer once the job starts.
        preview_parent = WAREHOUSE_DIR / "exports" / "splat" / "<EXPORT_ULID>"
        self.ply_target_line.setText(str(preview_parent / f"{stem}.ply"))
        self.ply_target_caption.setText(
            "EXPORT_ULID is minted at job start. The file lands in a fresh "
            "ULID directory alongside a manifest.json recording lineage back "
            "to the train run — collisions are impossible."
        )

    # ---- camera-path management ----
    def _refresh_camera_paths(self) -> None:
        self._camera_paths = _discover_camera_paths(self._project)

        self.cp_combo.blockSignals(True)
        self.cp_combo.clear()
        if self._camera_paths:
            for p in self._camera_paths:
                self.cp_combo.addItem(p.stem, userData=str(p))
        else:
            self.cp_combo.addItem("— none found —", userData=None)
        self.cp_combo.blockSignals(False)

        has_paths = bool(self._camera_paths)
        self.cp_combo.setEnabled(has_paths)
        self.video_name_edit.setEnabled(has_paths)
        self.render_btn.setEnabled(has_paths and self._running_kind is None
                                   and not self._jobs.is_running())

        cp_dir = _camera_paths_dir(self._project)
        if has_paths:
            self.cp_caption.setText(
                f"Found {len(self._camera_paths)} saved path(s) in {cp_dir}. "
                "Click ↻ to rescan after saving a new one in viser."
            )
            self._on_camera_path_selected(0)
        else:
            self.cp_source_line.setText("")
            if cp_dir.is_dir():
                self.cp_caption.setText(
                    f"No .json files in {cp_dir}.\n"
                    "Open the run in viser, build a keyframe path, and hit "
                    "the Save button in the keyframe editor. Then click ↻."
                )
            else:
                self.cp_caption.setText(
                    f"Nothing to render yet — {cp_dir} doesn't exist.\n"
                    "Open the run in viser, save a keyframe path, then click ↻."
                )
            self.video_target_line.setText("")
            self.video_target_caption.setText("")
            self.video_name_edit.clear()

    def _current_camera_path(self) -> Path | None:
        data = self.cp_combo.currentData()
        return Path(data) if data else None

    def _on_camera_path_selected(self, _idx: int = 0) -> None:
        cp = self._current_camera_path()
        if cp is None:
            self.cp_source_line.setText("")
            return
        self.cp_source_line.setText(str(cp))
        # Seed the video filename with the camera-path stem so rendered
        # files inherit the same timestamped name viser saved them under.
        self.video_name_edit.setText(cp.stem)

    def _refresh_video_target(self, _text: str = "") -> None:
        cp = self._current_camera_path()
        if cp is None:
            self.video_target_line.setText("")
            self.video_target_caption.setText("")
            return
        stem = _sanitize_stem(self.video_name_edit.text(), ".mp4")
        if not stem:
            stem = cp.stem
        preview_parent = WAREHOUSE_DIR / "renders" / "<RENDER_ULID>"
        self.video_target_line.setText(str(preview_parent / f"{stem}.mp4"))
        self.video_target_caption.setText(
            "RENDER_ULID is minted at job start. The .mp4 lands in a fresh "
            "ULID directory next to a snapshot of the camera_path.json used "
            "and a manifest.json with lineage to the train run."
        )

    # ---- preflight (shared) ----
    def _preflight(self) -> _Run | None:
        """Common guards before starting any export/render job. Returns the
        selected run on success, None (and pops a dialog) on failure."""
        if self._jobs.is_running():
            QMessageBox.information(
                self, "Busy",
                "Another job is already running. Cancel it first.",
            )
            return None
        state = dock.container_state(CONTAINER_NAME)
        if state.status != "running":
            QMessageBox.warning(
                self, "Container not running",
                f"The {CONTAINER_NAME} container isn't running. "
                "Start it from the Dashboard, then try again.",
            )
            return None
        run = self._current_run()
        if not run.config_path.is_file():
            QMessageBox.warning(
                self, "Run missing",
                f"Config file is gone:\n{run.config_path}\n\n"
                "The run folder may have been deleted. Switch projects "
                "and back to refresh the list.",
            )
            return None
        return run

    # ---- ply export ----
    def _on_export_ply(self) -> None:
        run = self._preflight()
        if run is None:
            return

        stem = _sanitize_stem(self.ply_name_edit.text(), ".ply") \
            or _default_ply_stem(run)
        payload_name = f"{stem}.ply"

        # Mint the export ULID, pre-create its dir host-side (so the post-run
        # manifest write doesn't EACCES against a root-owned container dir).
        export_ulid = wh.mint_ulid()
        out_dir = wh.export_dir("splat", export_ulid)
        out_dir.mkdir(parents=True, exist_ok=True)
        c_out_dir = wh.host_to_container_warehouse(out_dir)
        c_config = run.container_config

        # ns-export writes `splat.ply` inside --output-dir; rename to the
        # user's stem so the on-disk name carries meaning. ULID dir
        # guarantees uniqueness regardless of stem.
        if run.method in _SPLATFACTOW_METHODS:
            exporter_label = "ns-export-splatfactow"
            exporter_cmd = (
                "ns-export-splatfactow "
                f"--load-config {shlex.quote(c_config)} "
                f"--output-dir {shlex.quote(c_out_dir)}"
            )
        else:
            exporter_label = "ns-export gaussian-splat"
            exporter_cmd = (
                "ns-export gaussian-splat "
                f"--load-config {shlex.quote(c_config)} "
                f"--output-dir {shlex.quote(c_out_dir)}"
            )

        # `set -e` so the rename failure surfaces as a non-zero exit (don't
        # write a manifest claiming the .ply landed when it didn't).
        shell_cmd = (
            "set -e; "
            f"{exporter_cmd}; "
            f"mv {shlex.quote(c_out_dir)}/splat.ply "
            f"{shlex.quote(c_out_dir)}/{shlex.quote(payload_name)}"
        )
        display = (
            f"docker exec {CONTAINER_NAME} {exporter_cmd}   "
            f"# → {out_dir}/{payload_name}"
        )

        # Derive source_checkpoint_step from the parent train run's nested
        # dir so the manifest records exactly which ckpt was exported.
        nested = wh.find_train_run_subdir(WAREHOUSE_DIR / "trained" / run.train_run_id)
        ckpt = wh.find_latest_checkpoint(nested) if nested is not None else None
        source_step = ckpt[1] if ckpt is not None else 0

        params = ExportParams(
            kind="splat",
            exporter=exporter_label,
            source_checkpoint_step=source_step,
        )

        self._stash_export_pending(
            run=run,
            export_ulid=export_ulid,
            export_dir_path=out_dir,
            payload_name=payload_name,
            params=params,
            command=display,
        )

        self._jobs.start(JobSpec(
            kind="export",
            display_cmd=display,
            container_argv=["/bin/sh", "-c", shell_cmd],
            kill_pattern="ns-export",
        ))

    # ---- video render ----
    def _on_render_video(self) -> None:
        run = self._preflight()
        if run is None:
            return

        cp = self._current_camera_path()
        if cp is None or not cp.is_file():
            QMessageBox.warning(
                self, "Camera path missing",
                "Pick a saved camera path first. Click ↻ to rescan if you "
                "just saved one in viser.",
            )
            return

        stem = _sanitize_stem(self.video_name_edit.text(), ".mp4") or cp.stem
        payload_name = f"{stem}.mp4"

        render_ulid = wh.mint_ulid()
        out_dir = wh.run_dir("render", render_ulid)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the camera_path JSON next to the rendered video so the
        # render is reproducible without depending on the original (mutable)
        # file under the project's processed dir.
        snapshot_name = "camera_path.json"
        try:
            shutil.copy2(cp, out_dir / snapshot_name)
        except OSError as e:
            QMessageBox.warning(
                self, "Camera path snapshot failed",
                f"Couldn't copy {cp} into {out_dir}:\n{e}\n\n"
                "Render aborted.",
            )
            return

        c_out_dir = wh.host_to_container_warehouse(out_dir)
        c_out_path = f"{c_out_dir}/{payload_name}"
        c_camera = wh.host_to_container_warehouse(out_dir / snapshot_name)
        c_config = run.container_config

        # ns-render writes straight to --output-path — no tmp shuffle needed.
        shell_cmd = (
            "set -e; "
            "ns-render camera-path "
            f"--load-config {shlex.quote(c_config)} "
            f"--camera-path-filename {shlex.quote(c_camera)} "
            f"--output-path {shlex.quote(c_out_path)}"
        )
        display = (
            f"docker exec {CONTAINER_NAME} ns-render camera-path "
            f"--load-config {shlex.quote(c_config)} "
            f"--camera-path-filename {shlex.quote(c_camera)} "
            f"--output-path {shlex.quote(c_out_path)}"
        )

        params = RenderParams(
            camera_path_filename=wh.to_warehouse_relpath(out_dir / snapshot_name),
        )

        self._stash_render_pending(
            run=run,
            render_ulid=render_ulid,
            render_dir_path=out_dir,
            payload_name=payload_name,
            params=params,
            command=display,
        )

        self._jobs.start(JobSpec(
            kind="render",
            display_cmd=display,
            container_argv=["/bin/sh", "-c", shell_cmd],
            kill_pattern="ns-render",
        ))

    def _on_cancel(self) -> None:
        self._jobs.cancel()

    # ---- pending state (manifest write-ahead) ----
    def _stash_export_pending(
        self,
        *,
        run: _Run,
        export_ulid: str,
        export_dir_path: Path,
        payload_name: str,
        params: ExportParams,
        command: str,
    ) -> None:
        self._clear_pending()
        self._pending_run = run
        self._pending_export_ulid = export_ulid
        self._pending_export_dir = export_dir_path
        self._pending_export_payload = payload_name
        self._pending_export_params = params
        self._pending_export_command = command
        self._pending_start_time = datetime.now(timezone.utc)
        self._pending_display_name = Path(payload_name).stem

    def _stash_render_pending(
        self,
        *,
        run: _Run,
        render_ulid: str,
        render_dir_path: Path,
        payload_name: str,
        params: RenderParams,
        command: str,
    ) -> None:
        self._clear_pending()
        self._pending_run = run
        self._pending_render_ulid = render_ulid
        self._pending_render_dir = render_dir_path
        self._pending_render_payload = payload_name
        self._pending_render_params = params
        self._pending_render_command = command
        self._pending_start_time = datetime.now(timezone.utc)
        self._pending_display_name = Path(payload_name).stem

    def _clear_pending(self) -> None:
        self._pending_run = None
        self._pending_export_ulid = None
        self._pending_export_dir = None
        self._pending_export_payload = None
        self._pending_export_params = None
        self._pending_export_command = None
        self._pending_render_ulid = None
        self._pending_render_dir = None
        self._pending_render_payload = None
        self._pending_render_params = None
        self._pending_render_command = None
        self._pending_start_time = None
        self._pending_display_name = None

    def _duration(self) -> float | None:
        if self._pending_start_time is None:
            return None
        return (datetime.now(timezone.utc) - self._pending_start_time).total_seconds()

    def _finalize_export(self) -> Path | None:
        """Write the ExportManifest. Returns the on-disk payload path on
        success, or None if a required pending field is missing."""
        if (
            self._pending_run is None
            or self._pending_export_ulid is None
            or self._pending_export_dir is None
            or self._pending_export_payload is None
            or self._pending_export_params is None
            or self._pending_start_time is None
        ):
            return None
        try:
            wh.write_export_manifest(
                output_dir=self._pending_export_dir,
                run_id=self._pending_export_ulid,
                parent_train_id=self._pending_run.train_run_id,
                display_name=self._pending_display_name or self._pending_export_ulid,
                params=self._pending_export_params,
                command=self._pending_export_command or "",
                duration_seconds=self._duration(),
                start_time=self._pending_start_time,
                payload_filename=self._pending_export_payload,
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Export manifest write failed",
                f"ns-export succeeded but the ExportManifest couldn't be "
                f"written:\n{e}\n\nThe .ply on disk is intact at\n"
                f"{self._pending_export_dir}/{self._pending_export_payload}.",
            )
            return None
        return self._pending_export_dir / self._pending_export_payload

    def _finalize_render(self) -> Path | None:
        if (
            self._pending_run is None
            or self._pending_render_ulid is None
            or self._pending_render_dir is None
            or self._pending_render_payload is None
            or self._pending_render_params is None
            or self._pending_start_time is None
        ):
            return None
        try:
            wh.write_render_manifest(
                output_dir=self._pending_render_dir,
                run_id=self._pending_render_ulid,
                parent_train_id=self._pending_run.train_run_id,
                display_name=self._pending_display_name or self._pending_render_ulid,
                params=self._pending_render_params,
                command=self._pending_render_command or "",
                duration_seconds=self._duration(),
                start_time=self._pending_start_time,
                payload_filename=self._pending_render_payload,
                extra_outputs={
                    "camera_path": self._pending_render_params.camera_path_filename,
                },
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Render manifest write failed",
                f"ns-render succeeded but the RenderManifest couldn't be "
                f"written:\n{e}\n\nThe .mp4 on disk is intact at\n"
                f"{self._pending_render_dir}/{self._pending_render_payload}.",
            )
            return None
        return self._pending_render_dir / self._pending_render_payload

    # ---- job signals ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)
        self.render_btn.setEnabled(enabled and bool(self._camera_paths))

    def _on_job_started(self, spec: JobSpec) -> None:
        # Any job (ours or a foreign one) disables both action buttons;
        # only our jobs enable Cancel, so a stray train-tab cancel doesn't
        # kill another tab's work by accident.
        self._set_action_buttons_enabled(False)
        if spec.kind == "export":
            self._running_kind = "export"
            self.cancel_btn.setEnabled(True)
            set_pill(self.status_pill, "● Exporting", "warning")
        elif spec.kind == "render":
            self._running_kind = "render"
            self.cancel_btn.setEnabled(True)
            set_pill(self.status_pill, "● Rendering", "warning")
        else:
            self._running_kind = None
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Other job running", "warning")

    def _on_job_finished(self, code: int) -> None:
        kind = self._running_kind
        self._running_kind = None
        self.cancel_btn.setEnabled(False)
        self._set_action_buttons_enabled(True)
        if kind is None:
            set_pill(self.status_pill, "Ready", "neutral")
            self._clear_pending()
            return
        if code == 0:
            if kind == "export":
                payload = self._finalize_export()
                if payload is not None:
                    set_pill(self.status_pill, "Exported", "success")
                else:
                    set_pill(self.status_pill, "Exported · manifest skipped", "warning")
                self._refresh_ply_target()
            else:  # render
                payload = self._finalize_render()
                if payload is not None:
                    set_pill(self.status_pill, "Rendered", "success")
                else:
                    set_pill(self.status_pill, "Rendered · manifest skipped", "warning")
                self._refresh_video_target()
        else:
            set_pill(self.status_pill, f"Failed · exit {code}", "error")
        self._clear_pending()

    def _on_job_cancelled(self) -> None:
        kind = self._running_kind
        self._running_kind = None
        self.cancel_btn.setEnabled(False)
        self._set_action_buttons_enabled(True)
        if kind is None:
            set_pill(self.status_pill, "Ready", "neutral")
        else:
            set_pill(self.status_pill, "Cancelled", "error")
        self._clear_pending()

    def _on_job_failed(self, _msg: str) -> None:
        kind = self._running_kind
        self._running_kind = None
        self.cancel_btn.setEnabled(False)
        self._set_action_buttons_enabled(True)
        if kind is None:
            set_pill(self.status_pill, "Ready", "neutral")
        else:
            set_pill(self.status_pill, "Failed to start", "error")
        self._clear_pending()


# ---------- top-level view ----------

class ExportsView(QWidget):
    """Top-level Exports tab. Hosts ProjectBar + per-project form / gates."""

    request_navigate = Signal(int)  # tab index to switch to

    def __init__(self, store: ProjectStore, jobs: JobManager):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.project_bar = ProjectBar(store, jobs)
        outer.addWidget(self.project_bar)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        # Rebuild when a train job completes — a fresh run may have appeared.
        # Only rebuild if we're on a gate; a live form is fine as-is.
        self._last_started_kind: str | None = None
        self._jobs.started.connect(self._on_any_started)
        self._jobs.finished.connect(self._on_any_finished)

        self._rebuild()
        self._store.active_changed.connect(lambda _p: self._rebuild())
        self._store.projects_changed.connect(self._rebuild)

    def _on_any_started(self, spec: JobSpec) -> None:
        self._last_started_kind = spec.kind

    def _on_any_finished(self, code: int) -> None:
        was_train = self._last_started_kind == "train"
        self._last_started_kind = None
        if code != 0 or not was_train:
            return
        cur = self.stack.currentWidget() if self.stack.count() else None
        if not isinstance(cur, _Form):
            self._rebuild()

    def _rebuild(self) -> None:
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        active = self._store.active
        if active is None:
            self.stack.addWidget(_EmptyState())
            return

        runs = _discover_runs(active)
        if not runs:
            gate = _NoRunsState(active)
            gate.request_train.connect(lambda: self.request_navigate.emit(2))
            self.stack.addWidget(gate)
            return

        self.stack.addWidget(_Form(self._store, self._jobs, active, runs))

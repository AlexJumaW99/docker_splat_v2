"""
Exports view — runs ns-export gaussian-splat and ns-render camera-path inside
splat-ns.

Two artifacts per trained run:

* `.ply` splat — ns-export always writes `splat.ply` into its --output-dir, so
  we export into a hidden per-job temp dir and rename to `<unique>.ply`. A
  shell trap cleans the temp dir on cancel / error.
* `.mp4` video — ns-render writes directly to --output-path, so we just
  mkdir -p the parent and pick a non-colliding filename host-side.

Both cards share the same run picker. Only one JobManager job runs at a time,
so a single Cancel button + status pill covers both.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, DATA_DIR
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore
from app.widgets import HelpBadge, ProjectBar, card, labeled, pill, section_title, set_pill


_HELP = {
    "run": (
        "Training run",
        "A finished ns-train run under "
        "<code>data/outputs/&lt;project&gt;/&lt;method&gt;/&lt;timestamp&gt;/</code>. "
        "Each run has its own <code>config.yml</code> + checkpoints; pick the "
        "run you want to export a .ply from or render a video of. Latest "
        "runs appear first."
    ),
    "ply_name": (
        "Export filename",
        "Name for the exported <code>.ply</code> under "
        "<code>data/exports/splat/</code>. Don't include the extension. "
        "If the name already exists, the GUI appends <code>_2</code>, "
        "<code>_3</code>\u2026 automatically \u2014 nothing is ever overwritten."
    ),
    "camera_path": (
        "Camera path",
        "A keyframe path saved from viser's keyframe editor, stored as "
        "<code>&lt;processed_dir&gt;/camera_paths/*.json</code>. Open the run "
        "in viser, build a path, click <b>Save</b>, then hit \u21bb here to "
        "rescan. The render uses the path's keyframes + easing verbatim."
    ),
    "video_name": (
        "Render filename",
        "Name for the rendered <code>.mp4</code> under "
        "<code>data/renders/&lt;project&gt;/</code>. Don't include the "
        "extension; duplicates auto-suffix with <code>_2</code>, "
        "<code>_3</code>\u2026 so nothing is overwritten."
    ),
}


def _hb(key: str) -> HelpBadge:
    title, body = _HELP[key]
    return HelpBadge(title, body)


OUTPUTS_DIR = DATA_DIR / "outputs"
EXPORTS_DIR = DATA_DIR / "exports" / "splat"
RENDERS_DIR = DATA_DIR / "renders"

# splatfacto-w / -light need a different exporter because SplatfactoWModel
# doesn't subclass SplatfactoModel and computes SH coefficients per-frame
# from a learned appearance embedding. The image ships a custom CLI
# `ns-export-splatfactow` that bakes the average appearance embedding into
# each gaussian and writes an inria-format .ply in the same splat.ply layout.
_SPLATFACTOW_METHODS = frozenset({"splatfacto-w", "splatfacto-w-light"})


# ---------- path helpers ----------

def _host_to_container(path: Path) -> str:
    """Translate a host path under DATA_DIR to its /workspace-mounted form."""
    rel = path.resolve().relative_to(DATA_DIR.resolve())
    return f"/workspace/{rel.as_posix()}"


def _sanitize_stem(raw: str, ext: str) -> str:
    """Strip a trailing extension the user may have typed, drop path
    separators, and trim whitespace. We only ever write into the
    format-specific output dir, so slashes are never meaningful here."""
    s = raw.strip()
    if ext and s.lower().endswith(ext.lower()):
        s = s[:-len(ext)]
    s = s.replace("/", "_").replace("\\", "_")
    return s.strip()


def _unique_path(parent: Path, stem: str, ext: str) -> Path:
    """Return a path under `parent` whose `<stem><ext>` name is free.
    Appends `_2`, `_3`, … until a free slot is found."""
    target = parent / f"{stem}{ext}"
    if not target.exists():
        return target
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


# ---------- training run discovery ----------

@dataclass(frozen=True)
class _Run:
    project_id: str
    method: str         # e.g. "splatfacto"
    timestamp: str      # e.g. "2026-04-20_020610"
    config_path: Path   # absolute host path to config.yml

    @property
    def label(self) -> str:
        return f"{self.method} · {self.timestamp}"

    @property
    def container_config(self) -> str:
        return _host_to_container(self.config_path)


# upstream nerfstudio writes both `splatfacto` and `splatfacto-mcmc` runs under
# `<experiment>/splatfacto/<timestamp>/` (method_name='splatfacto' for both),
# differing only by `strategy: mcmc` vs `strategy: default` inside config.yml.
# We peek at that field so the Exports dropdown labels mcmc runs accurately
# and .ply filenames derived from `run.method` carry the strategy through.
_STRATEGY_RE = re.compile(r"^\s*strategy:\s*([A-Za-z_-]+)\s*$", re.MULTILINE)


def _method_from_config(dir_method: str, config_path: Path) -> str:
    if dir_method != "splatfacto":
        return dir_method
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return dir_method
    m = _STRATEGY_RE.search(text)
    if m and m.group(1) == "mcmc":
        return "splatfacto-mcmc"
    return dir_method


def _discover_runs(project_id: str) -> list[_Run]:
    root = OUTPUTS_DIR / project_id
    if not root.is_dir():
        return []
    runs: list[_Run] = []
    for method_dir in (p for p in root.iterdir() if p.is_dir()):
        for ts_dir in (p for p in method_dir.iterdir() if p.is_dir()):
            cfg = ts_dir / "config.yml"
            if cfg.is_file():
                runs.append(_Run(
                    project_id=project_id,
                    method=_method_from_config(method_dir.name, cfg),
                    timestamp=ts_dir.name,
                    config_path=cfg,
                ))
    # ns-train's timestamp folders sort lexically = chronologically.
    runs.sort(key=lambda r: (r.timestamp, r.method), reverse=True)
    return runs


def _default_ply_stem(run: _Run) -> str:
    return f"{run.project_id}_{run.method}_{run.timestamp}"


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


def _renders_out_dir(project: Project) -> Path:
    """Per-project host folder for rendered videos. Uses the processed folder's
    basename so the layout mirrors the 'renders/hrm_splat/…' pattern that
    ns-render's default viser export already uses."""
    return RENDERS_DIR / Path(project.processed_dir).name


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
    """Active project has no finished runs. Offers a jump to Train."""

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

        sub = QLabel(
            f"'{project.name}' has no finished training runs under\n"
            f"{OUTPUTS_DIR / project.id}\n\n"
            "Train a model first, then come back to export its splat."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(560)

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
            "saved viser keyframe path. Output filenames auto-suffix to avoid "
            "overwrites."
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
        target = _unique_path(EXPORTS_DIR, stem, ".ply")
        self.ply_target_line.setText(str(target))
        desired = f"{stem}.ply"
        if target.name != desired:
            self.ply_target_caption.setText(
                f"'{desired}' already exists — will save as '{target.name}' "
                "to avoid overwrites."
            )
        else:
            self.ply_target_caption.setText(
                "Filename is free. Re-running the export will auto-suffix "
                "instead of overwriting."
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
        out_dir = _renders_out_dir(self._project)
        target = _unique_path(out_dir, stem, ".mp4")
        self.video_target_line.setText(str(target))
        desired = f"{stem}.mp4"
        if target.name != desired:
            self.video_target_caption.setText(
                f"'{desired}' already exists in {out_dir} — will save as "
                f"'{target.name}' to avoid overwrites."
            )
        else:
            self.video_target_caption.setText(
                f"Filename is free. Will land in {out_dir}."
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
        host_target = _unique_path(EXPORTS_DIR, stem, ".ply")
        final_name = host_target.name

        # Both exporters write `splat.ply` into their --output-dir, so we
        # export into a private tmp subdir and rename. A trap cleans the tmp
        # on success, cancel, or error so nothing stale accumulates.
        tmp_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        c_exports = "/workspace/exports/splat"
        c_tmp = f"{c_exports}/.tmp_{tmp_tag}"
        c_final = f"{c_exports}/{final_name}"
        c_config = run.container_config

        if run.method in _SPLATFACTOW_METHODS:
            exporter_cmd = (
                "ns-export-splatfactow "
                f"--load-config {shlex.quote(c_config)} "
                "--output-dir \"$TMP\""
            )
            display_cmd = (
                f"ns-export-splatfactow "
                f"--load-config {shlex.quote(c_config)} "
                f"--output-dir {shlex.quote(c_exports)}"
            )
        else:
            exporter_cmd = (
                "ns-export gaussian-splat "
                f"--load-config {shlex.quote(c_config)} "
                "--output-dir \"$TMP\""
            )
            display_cmd = (
                f"ns-export gaussian-splat "
                f"--load-config {shlex.quote(c_config)} "
                f"--output-dir {shlex.quote(c_exports)}"
            )

        shell_cmd = (
            "set -e; "
            f"TMP={shlex.quote(c_tmp)}; "
            "trap 'rm -rf \"$TMP\"' EXIT; "
            f"mkdir -p {shlex.quote(c_exports)} \"$TMP\"; "
            f"{exporter_cmd}; "
            f"mv \"$TMP/splat.ply\" {shlex.quote(c_final)}"
        )
        display = f"docker exec {CONTAINER_NAME} {display_cmd}   # → {final_name}"

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
        out_dir = _renders_out_dir(self._project)
        host_target = _unique_path(out_dir, stem, ".mp4")
        final_name = host_target.name

        c_config = run.container_config
        c_camera = _host_to_container(cp)
        c_out_dir = _host_to_container(out_dir) if out_dir.exists() \
            else f"/workspace/renders/{out_dir.name}"
        c_out_path = f"{c_out_dir}/{final_name}"

        # ns-render writes straight to --output-path, so no tmp shuffle.
        # We still mkdir -p in case this is the project's first render.
        shell_cmd = (
            "set -e; "
            f"mkdir -p {shlex.quote(c_out_dir)}; "
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

        self._jobs.start(JobSpec(
            kind="render",
            display_cmd=display,
            container_argv=["/bin/sh", "-c", shell_cmd],
            kill_pattern="ns-render",
        ))

    def _on_cancel(self) -> None:
        self._jobs.cancel()

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
            return
        if code == 0:
            if kind == "export":
                set_pill(self.status_pill, "Exported", "success")
                self._refresh_ply_target()
            else:  # render
                set_pill(self.status_pill, "Rendered", "success")
                self._refresh_video_target()
        else:
            set_pill(self.status_pill, f"Failed · exit {code}", "error")

    def _on_job_cancelled(self) -> None:
        kind = self._running_kind
        self._running_kind = None
        self.cancel_btn.setEnabled(False)
        self._set_action_buttons_enabled(True)
        if kind is None:
            set_pill(self.status_pill, "Ready", "neutral")
        else:
            set_pill(self.status_pill, "Cancelled", "error")

    def _on_job_failed(self, _msg: str) -> None:
        kind = self._running_kind
        self._running_kind = None
        self.cancel_btn.setEnabled(False)
        self._set_action_buttons_enabled(True)
        if kind is None:
            set_pill(self.status_pill, "Ready", "neutral")
        else:
            set_pill(self.status_pill, "Failed to start", "error")


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

        self.project_bar = ProjectBar(store)
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

        runs = _discover_runs(active.id)
        if not runs:
            gate = _NoRunsState(active)
            gate.request_train.connect(lambda: self.request_navigate.emit(2))
            self.stack.addWidget(gate)
            return

        self.stack.addWidget(_Form(self._store, self._jobs, active, runs))

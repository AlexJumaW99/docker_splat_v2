"""
Viewers tab — open Nerfstudio's viser viewer and TensorBoard in the browser.

Two cards: Viser (port 7007) and TensorBoard (port 6006). Both show live
port-status pills polled every 2 s via container_port_in_use(). The
TensorBoard card can start a tensorboard server scoped to the active
project's output dir, or to all outputs.

The tab also discovers training runs for the active project (reuses the
same _discover_runs / _Run helpers from exports.py) so the user can
launch ns-viewer against a finished checkpoint or start TensorBoard
scoped to a specific run.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import webbrowser
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, DATA_DIR, TENSORBOARD_PORT, VISER_PORT
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore
from app.widgets import HelpBadge, ProjectBar, card, pill, section_title, set_pill


# ---------- help copy ----------

_HELP = {
    "viser": (
        "Nerfstudio Viewer (viser)",
        f"The real-time 3D viewer that ships with Nerfstudio, served on "
        f"<code>localhost:{VISER_PORT}</code>.<br><br>"
        "<b>During training</b> — ns-train starts viser automatically when "
        "the vis mode includes 'viewer'. The viewer streams live renders as "
        "the model converges.<br>"
        "<b>After training</b> — use <i>Launch ns-viewer</i> to load a "
        "finished checkpoint and explore the scene without retraining.",
    ),
    "tensorboard": (
        "TensorBoard",
        f"Training-metrics dashboard on <code>localhost:{TENSORBOARD_PORT}</code>. "
        "Shows loss curves, PSNR, SSIM, and other scalars logged during "
        "training.<br><br>"
        "<b>Important:</b> metrics are only written to TensorBoard event files "
        "when the Train tab's <i>Visualization</i> is set to "
        "<code>viewer+tensorboard</code> or <code>tensorboard</code>. Runs "
        "trained with <code>viewer</code> only have no event files to view.",
    ),
}

def _hb(key: str) -> HelpBadge:
    title, body = _HELP[key]
    return HelpBadge(title, body)


# ---------- constants ----------

OUTPUTS_DIR = DATA_DIR / "outputs"


# ---------- run discovery (shared with exports.py) ----------

_STRATEGY_RE = re.compile(r"^\s*strategy:\s*([A-Za-z_-]+)\s*$", re.MULTILINE)


class _Run:
    __slots__ = ("project_id", "method", "timestamp", "config_path", "has_events")

    def __init__(
        self, project_id: str, method: str, timestamp: str,
        config_path: Path, has_events: bool,
    ):
        self.project_id = project_id
        self.method = method
        self.timestamp = timestamp
        self.config_path = config_path
        self.has_events = has_events

    @property
    def label(self) -> str:
        return f"{self.method} · {self.timestamp}"

    @property
    def container_config(self) -> str:
        rel = self.config_path.resolve().relative_to(DATA_DIR.resolve())
        return f"/workspace/{rel.as_posix()}"


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
                has_events = any(ts_dir.glob("events.out.tfevents.*"))
                runs.append(_Run(
                    project_id=project_id,
                    method=_method_from_config(method_dir.name, cfg),
                    timestamp=ts_dir.name,
                    config_path=cfg,
                    has_events=has_events,
                ))
    runs.sort(key=lambda r: (r.timestamp, r.method), reverse=True)
    return runs


# ---------- background port poller ----------

class _PortSnapshot:
    __slots__ = ("viser", "tensorboard")

    def __init__(self, viser: bool | None, tensorboard: bool | None):
        self.viser = viser
        self.tensorboard = tensorboard


class _PollSignals(QObject):
    updated = Signal(object)


class _PollTask(QRunnable):
    def __init__(self, signals: _PollSignals):
        super().__init__()
        self.setAutoDelete(True)
        self._signals = signals

    @Slot()
    def run(self) -> None:
        viser = dock.container_port_in_use(VISER_PORT)
        tb = dock.container_port_in_use(TENSORBOARD_PORT)
        try:
            self._signals.updated.emit(_PortSnapshot(viser, tb))
        except RuntimeError:
            pass


class _PortPoller(QObject):
    updated = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._signals = _PollSignals()
        self._signals.updated.connect(self.updated)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self) -> None:
        if self._pool.activeThreadCount() > 0:
            return
        self._pool.start(_PollTask(self._signals))


# ---------- empty state ----------

class _EmptyState(QWidget):
    def __init__(self, store: ProjectStore):
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
            "Viewers need a project with at least one training run. "
            "Create a project and train it first."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(440)

        from app.widgets import NewProjectDialog
        btn = QPushButton("+  New project")
        btn.setProperty("role", "primary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: NewProjectDialog(store, self).exec())

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        v.addWidget(btn, 0, Qt.AlignHCenter)
        v.addStretch(1)


class _NoRunsState(QWidget):
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
            "Train a model first, then come back to view it."
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

    request_train = Signal()

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
        self._running_kind: str | None = None

        self._build()
        self._select_run(0)
        self._wire_jobs()

        self._poller = _PortPoller(self)
        self._poller.updated.connect(self._on_port_update)

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

        title = QLabel("Viewers")
        title.setObjectName("PageTitle")
        sub = QLabel(
            "Open the Nerfstudio 3D viewer or TensorBoard for the active project."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        header = QVBoxLayout(); header.setSpacing(2)
        header.addWidget(title); header.addWidget(sub)
        bl.addLayout(header)

        bl.addWidget(self._run_card())
        bl.addWidget(self._viser_card())
        bl.addWidget(self._tensorboard_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- run card ----
    def _run_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)

        hdr = QHBoxLayout(); hdr.setSpacing(6); hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(section_title("TRAINING RUN"))
        hdr.addStretch(1)
        hdr_w = QWidget(); hdr_w.setLayout(hdr)
        v.addWidget(hdr_w)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        for r in self._runs:
            self.run_combo.addItem(r.label)
        self.run_combo.currentIndexChanged.connect(self._select_run)
        row.addWidget(QLabel("Run"))
        row.addWidget(self.run_combo, 1)
        v.addLayout(row)

        self.run_meta_lbl = QLabel("")
        self.run_meta_lbl.setProperty("role", "caption")
        self.run_meta_lbl.setWordWrap(True)
        v.addWidget(self.run_meta_lbl)

        caption = QLabel(
            "Select a training run to launch ns-viewer against or to scope "
            "TensorBoard to. Latest runs appear first."
        )
        caption.setProperty("role", "caption")
        caption.setWordWrap(True)
        v.addWidget(caption)
        return c

    # ---- viser card ----
    def _viser_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        hdr = QHBoxLayout(); hdr.setSpacing(6); hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(section_title("NERFSTUDIO VIEWER"))
        hdr.addWidget(_hb("viser"))
        hdr.addStretch(1)
        self.viser_pill = pill("—", "neutral")
        hdr.addWidget(self.viser_pill)
        hdr_w = QWidget(); hdr_w.setLayout(hdr)
        v.addWidget(hdr_w)

        self.viser_status_lbl = QLabel("Checking port status…")
        self.viser_status_lbl.setProperty("role", "caption")
        self.viser_status_lbl.setWordWrap(True)
        v.addWidget(self.viser_status_lbl)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.open_viser_btn = QPushButton(f"Open http://localhost:{VISER_PORT}")
        self.open_viser_btn.setProperty("role", "primary")
        self.open_viser_btn.setCursor(Qt.PointingHandCursor)
        self.open_viser_btn.clicked.connect(self._open_viser)
        self.open_viser_btn.setEnabled(False)

        self.launch_viewer_btn = QPushButton("Launch ns-viewer")
        self.launch_viewer_btn.setProperty("role", "secondary")
        self.launch_viewer_btn.setCursor(Qt.PointingHandCursor)
        self.launch_viewer_btn.clicked.connect(self._on_launch_viewer)

        self.stop_viser_btn = QPushButton("Stop viewer")
        self.stop_viser_btn.setProperty("role", "ghost")
        self.stop_viser_btn.setCursor(Qt.PointingHandCursor)
        self.stop_viser_btn.clicked.connect(self._on_stop_viser)
        self.stop_viser_btn.hide()

        btn_row.addWidget(self.open_viser_btn)
        btn_row.addWidget(self.launch_viewer_btn)
        btn_row.addWidget(self.stop_viser_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return c

    # ---- tensorboard card ----
    def _tensorboard_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        hdr = QHBoxLayout(); hdr.setSpacing(6); hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(section_title("TENSORBOARD"))
        hdr.addWidget(_hb("tensorboard"))
        hdr.addStretch(1)
        self.tb_pill = pill("—", "neutral")
        hdr.addWidget(self.tb_pill)
        hdr_w = QWidget(); hdr_w.setLayout(hdr)
        v.addWidget(hdr_w)

        self.tb_status_lbl = QLabel("Checking port status…")
        self.tb_status_lbl.setProperty("role", "caption")
        self.tb_status_lbl.setWordWrap(True)
        v.addWidget(self.tb_status_lbl)

        self.tb_events_lbl = QLabel("")
        self.tb_events_lbl.setProperty("role", "caption")
        self.tb_events_lbl.setWordWrap(True)
        v.addWidget(self.tb_events_lbl)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)

        self.open_tb_btn = QPushButton(f"Open http://localhost:{TENSORBOARD_PORT}")
        self.open_tb_btn.setProperty("role", "primary")
        self.open_tb_btn.setCursor(Qt.PointingHandCursor)
        self.open_tb_btn.clicked.connect(self._open_tensorboard)
        self.open_tb_btn.setEnabled(False)

        self.start_tb_btn = QPushButton("Start TensorBoard")
        self.start_tb_btn.setProperty("role", "secondary")
        self.start_tb_btn.setCursor(Qt.PointingHandCursor)
        self.start_tb_btn.clicked.connect(self._on_start_tensorboard)

        self.stop_tb_btn = QPushButton("Stop TensorBoard")
        self.stop_tb_btn.setProperty("role", "ghost")
        self.stop_tb_btn.setCursor(Qt.PointingHandCursor)
        self.stop_tb_btn.clicked.connect(self._on_stop_tensorboard)
        self.stop_tb_btn.hide()

        btn_row.addWidget(self.open_tb_btn)
        btn_row.addWidget(self.start_tb_btn)
        btn_row.addWidget(self.stop_tb_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return c

    # ---- run selection ----
    def _current_run(self) -> _Run | None:
        idx = self.run_combo.currentIndex()
        if 0 <= idx < len(self._runs):
            return self._runs[idx]
        return None

    def _select_run(self, idx: int) -> None:
        if not (0 <= idx < len(self._runs)):
            return
        r = self._runs[idx]
        parts = [str(r.config_path.parent)]
        if r.has_events:
            parts.append("TensorBoard events: yes")
        else:
            parts.append("TensorBoard events: none (trained with --vis viewer)")
        self.run_meta_lbl.setText("  ·  ".join(parts))

        if r.has_events:
            self.tb_events_lbl.setText(
                f"✓ This run has TensorBoard event files."
            )
        else:
            self.tb_events_lbl.setText(
                "⚠ This run has no TensorBoard event files. "
                "To get metrics, retrain with Visualization set to "
                "'viewer+tensorboard'."
            )

    # ---- port status ----
    def _on_port_update(self, snap: _PortSnapshot) -> None:
        # Viser
        if snap.viser is True:
            set_pill(self.viser_pill, "● Listening", "success")
            self.viser_status_lbl.setText(
                f"Viser is live on localhost:{VISER_PORT}."
            )
            self.open_viser_btn.setEnabled(True)
            self.stop_viser_btn.show()
        elif snap.viser is False:
            set_pill(self.viser_pill, "Not listening", "neutral")
            self.viser_status_lbl.setText(
                f"Nothing is bound to port {VISER_PORT} inside the container. "
                "Start training (with viewer enabled) or launch ns-viewer."
            )
            self.open_viser_btn.setEnabled(False)
            self.stop_viser_btn.hide()
        else:
            set_pill(self.viser_pill, "Unknown", "warning")
            self.viser_status_lbl.setText(
                "Container not running — start it from the Dashboard."
            )
            self.open_viser_btn.setEnabled(False)
            self.stop_viser_btn.hide()

        # TensorBoard
        if snap.tensorboard is True:
            set_pill(self.tb_pill, "● Listening", "success")
            self.tb_status_lbl.setText(
                f"TensorBoard is live on localhost:{TENSORBOARD_PORT}."
            )
            self.open_tb_btn.setEnabled(True)
            self.stop_tb_btn.show()
            self.start_tb_btn.hide()
        elif snap.tensorboard is False:
            set_pill(self.tb_pill, "Not listening", "neutral")
            self.tb_status_lbl.setText(
                f"Nothing is bound to port {TENSORBOARD_PORT} inside the container."
            )
            self.open_tb_btn.setEnabled(False)
            self.stop_tb_btn.hide()
            self.start_tb_btn.show()
        else:
            set_pill(self.tb_pill, "Unknown", "warning")
            self.tb_status_lbl.setText(
                "Container not running — start it from the Dashboard."
            )
            self.open_tb_btn.setEnabled(False)
            self.stop_tb_btn.hide()
            self.start_tb_btn.show()

    # ---- viser actions ----
    def _open_viser(self) -> None:
        webbrowser.open(f"http://localhost:{VISER_PORT}")

    def _on_launch_viewer(self) -> None:
        if self._jobs.is_running():
            QMessageBox.information(
                self, "Busy",
                "Another job is already running. Cancel it first.",
            )
            return

        state = dock.container_state(CONTAINER_NAME)
        if state.status != "running":
            QMessageBox.warning(
                self, "Container not running",
                f"The {CONTAINER_NAME} container isn't running. "
                "Start it from the Dashboard, then try again.",
            )
            return

        run = self._current_run()
        if run is None:
            return
        if not run.config_path.is_file():
            QMessageBox.warning(
                self, "Run missing",
                f"Config file is gone:\n{run.config_path}",
            )
            return

        busy = dock.container_port_in_use(VISER_PORT)
        if busy is True:
            resp = QMessageBox.question(
                self, "Viser port is busy",
                f"Something inside {CONTAINER_NAME} is already bound to "
                f":{VISER_PORT}.\n\n"
                "Terminate it and continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return
            if not dock.free_container_port(VISER_PORT):
                QMessageBox.warning(
                    self, "Couldn't free port",
                    f"Port {VISER_PORT} is still busy. Inspect with:\n\n"
                    f"  docker exec {CONTAINER_NAME} ps -ef",
                )
                return

        c_config = run.container_config
        argv = [
            "ns-viewer",
            "--load-config", c_config,
            "--viewer.websocket-host", "0.0.0.0",
        ]
        display = (
            f"docker exec {CONTAINER_NAME} "
            + " ".join(shlex.quote(a) for a in argv)
        )

        self._jobs.start(JobSpec(
            kind="viewer",
            display_cmd=display,
            container_argv=argv,
            kill_pattern="ns-viewer",
        ))

    def _on_stop_viser(self) -> None:
        if self._jobs.is_running() and self._jobs.current is not None:
            if self._jobs.current.kind == "viewer":
                self._jobs.cancel()
                return
        dock.free_container_port(
            VISER_PORT, owners=("ns-viewer", "ns-train"),
        )

    # ---- tensorboard actions ----
    def _open_tensorboard(self) -> None:
        webbrowser.open(f"http://localhost:{TENSORBOARD_PORT}")

    def _on_start_tensorboard(self) -> None:
        state = dock.container_state(CONTAINER_NAME)
        if state.status != "running":
            QMessageBox.warning(
                self, "Container not running",
                f"The {CONTAINER_NAME} container isn't running. "
                "Start it from the Dashboard, then try again.",
            )
            return

        runs_with_events = [r for r in self._runs if r.has_events]
        if not runs_with_events:
            all_events = list(
                (OUTPUTS_DIR).rglob("events.out.tfevents.*")
            )
            if not all_events:
                QMessageBox.information(
                    self, "No TensorBoard data",
                    "No training runs have TensorBoard event files.\n\n"
                    "To generate them, set Visualization to "
                    "'viewer+tensorboard' on the Train tab before training.",
                )
                return

        busy = dock.container_port_in_use(TENSORBOARD_PORT)
        if busy is True:
            resp = QMessageBox.question(
                self, "TensorBoard port is busy",
                f"Something inside {CONTAINER_NAME} is already bound to "
                f":{TENSORBOARD_PORT}.\n\n"
                "Terminate it and start a new TensorBoard?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return
            _kill_tensorboard_in_container()

        c_logdir = f"/workspace/outputs/{self._project.id}"
        try:
            subprocess.Popen(
                [
                    "docker", "exec", "-d", CONTAINER_NAME,
                    "tensorboard",
                    "--logdir", c_logdir,
                    "--bind_all",
                    "--port", str(TENSORBOARD_PORT),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            QMessageBox.warning(
                self, "Failed to start TensorBoard", str(e),
            )

    def _on_stop_tensorboard(self) -> None:
        _kill_tensorboard_in_container()

    # ---- job signals ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _on_job_started(self, spec: JobSpec) -> None:
        if spec.kind == "viewer":
            self._running_kind = "viewer"
            self.launch_viewer_btn.setEnabled(False)
        elif spec.kind in ("process", "train", "export", "render"):
            self._running_kind = None
            self.launch_viewer_btn.setEnabled(False)

    def _on_job_finished(self, _code: int) -> None:
        self._running_kind = None
        self.launch_viewer_btn.setEnabled(True)

    def _on_job_cancelled(self) -> None:
        self._running_kind = None
        self.launch_viewer_btn.setEnabled(True)

    def _on_job_failed(self, _msg: str) -> None:
        self._running_kind = None
        self.launch_viewer_btn.setEnabled(True)


def _kill_tensorboard_in_container() -> None:
    try:
        subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "pkill", "-f", "tensorboard"],
            capture_output=True, timeout=5.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


# ---------- top-level view ----------

class ViewersView(QWidget):
    request_navigate = Signal(int)

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
            self.stack.addWidget(_EmptyState(self._store))
            return

        runs = _discover_runs(active.id)
        if not runs:
            gate = _NoRunsState(active)
            gate.request_train.connect(lambda: self.request_navigate.emit(2))
            self.stack.addWidget(gate)
            return

        form = _Form(self._store, self._jobs, active, runs)
        form.request_train.connect(lambda: self.request_navigate.emit(2))
        self.stack.addWidget(form)

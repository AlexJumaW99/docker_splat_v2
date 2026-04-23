from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable, QThreadPool, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, IMAGE_NAME
from app.nvidia_service import GpuStats, query_gpu
from app.widgets import HelpBadge


_DOCKER_HELP = (
    "Docker engine",
    "Nerfstudio runs inside a Docker container, so the daemon must be up "
    "before anything else works. This card polls <code>docker version</code> "
    "every 1.5 s. If the daemon is offline, the button uses polkit to run "
    "<code>systemctl start docker</code> \u2014 you'll see a password prompt "
    "once."
)
_CONTAINER_HELP = (
    "splat-ns container",
    f"The <code>{CONTAINER_NAME}</code> container ships ns-process-data, "
    "COLMAP, and ns-train. The GUI never creates or builds it \u2014 it only "
    "attaches to one you've made yourself. Click <b>Show create "
    "instructions</b> if it's missing; the dialog has the exact "
    "<code>docker create</code> command."
)
_GPU_HELP = (
    "GPU",
    "Live telemetry from <code>nvidia-smi --query-gpu</code>. <b>Utilization</b> "
    "is compute load (SM activity); <b>memory</b> is VRAM used / total; "
    "<b>mem I/O</b> is the memory controller's load. A training step that "
    "keeps utilization pegged and mem I/O low means you're compute-bound \u2014 "
    "lower resolution or fewer gaussians to speed up."
)


DOCKER_CREATE_CMD = """docker create \\
  --name splat-ns-v2 \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  --shm-size=16g \\
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \\
  -v ~/docker-stacks/docker_splat_v2/data:/workspace \\
  -p 7007:7007 \\
  -p 6006:6006 \\
  -it \\
  splat-nerfstudio-v2"""

DOCKER_BUILD_CMD = ("cd ~/docker-stacks/docker_splat_v2\n"
                   "docker build --target nerfstudio -t splat-nerfstudio-v2 .")


# ---------- helpers ----------

def _pill(text: str, kind: str = "neutral") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("pill", kind)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def _set_pill(lbl: QLabel, text: str, kind: str) -> None:
    lbl.setText(text)
    lbl.setProperty("pill", kind)
    lbl.style().unpolish(lbl); lbl.style().polish(lbl)


def _card() -> QFrame:
    f = QFrame()
    f.setProperty("role", "card")
    return f


# ---------- background poller ----------

@dataclass
class _Snapshot:
    docker_version: str | None
    container_state: dock.ContainerState | None
    image_exists: bool
    gpu: GpuStats | None


class _PollerSignals(QObject):
    updated = Signal(object)  # _Snapshot


class _PollTask(QRunnable):
    def __init__(self, signals: _PollerSignals):
        super().__init__()
        self.setAutoDelete(True)
        self._signals = signals

    @Slot()
    def run(self) -> None:
        ver = dock.daemon_version()
        state = dock.container_state() if ver else None
        img = dock.image_exists() if ver else False
        gpu = query_gpu()
        self._signals.updated.emit(_Snapshot(ver, state, img, gpu))


class StatusPoller(QObject):
    """Polls Docker + GPU state on a worker thread every 1.5 s."""
    updated = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._signals = _PollerSignals()
        self._signals.updated.connect(self.updated)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self) -> None:
        if self._pool.activeThreadCount() > 0:
            return  # previous poll still running; skip
        self._pool.start(_PollTask(self._signals))


# ---------- guide dialog ----------

class GuideDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Create the splat-ns-v2 container")
        self.setMinimumSize(640, 520)

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 24, 24, 20)
        v.setSpacing(12)

        title = QLabel("Create the splat-ns-v2 container")
        title.setObjectName("PageTitle")
        v.addWidget(title)

        intro = QLabel(
            "This GUI never builds or creates your Docker image or container — "
            "it only attaches to one you've already made. Run these in a terminal, "
            "then return here."
        )
        intro.setProperty("role", "caption")
        intro.setWordWrap(True)
        v.addWidget(intro)

        v.addWidget(self._step("1. Build the image (if not already built)"))
        self.cmd_build = self._code_block(DOCKER_BUILD_CMD, height=60)
        v.addWidget(self.cmd_build)

        v.addWidget(self._step("2. Create the container"))
        self.cmd_create = self._code_block(DOCKER_CREATE_CMD, height=240)
        v.addWidget(self.cmd_create)

        row = QHBoxLayout()
        row.addStretch(1)
        copy_build = QPushButton("Copy build command")
        copy_build.setProperty("role", "secondary")
        copy_build.setCursor(Qt.PointingHandCursor)
        copy_build.clicked.connect(lambda: self._copy(DOCKER_BUILD_CMD))

        copy_create = QPushButton("Copy create command")
        copy_create.setProperty("role", "secondary")
        copy_create.setCursor(Qt.PointingHandCursor)
        copy_create.clicked.connect(lambda: self._copy(DOCKER_CREATE_CMD))

        close = QPushButton("Close")
        close.setProperty("role", "primary")
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.accept)

        row.addWidget(copy_build)
        row.addWidget(copy_create)
        row.addWidget(close)
        v.addLayout(row)

    @staticmethod
    def _step(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: 600;")
        return lbl

    @staticmethod
    def _code_block(text: str, height: int) -> QPlainTextEdit:
        e = QPlainTextEdit(text)
        e.setReadOnly(True)
        e.setFixedHeight(height)
        return e

    @staticmethod
    def _copy(text: str) -> None:
        QGuiApplication.clipboard().setText(text)


# ---------- main view ----------

class DashboardView(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Canvas")

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(16)

        header = QLabel("Dashboard"); header.setObjectName("PageTitle")
        sub = QLabel(f"Live status of the Docker engine, the "
                     f"{CONTAINER_NAME} container, and your GPU.")
        sub.setProperty("role", "caption")

        root.addWidget(header)
        root.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._docker_card = self._build_docker_card()
        self._cont_card = self._build_container_card()
        self._gpu_card = self._build_gpu_card()

        grid.addWidget(self._docker_card, 0, 0)
        grid.addWidget(self._cont_card, 0, 1)
        grid.addWidget(self._gpu_card, 1, 0, 1, 2)
        root.addLayout(grid)
        root.addStretch(1)

        self._poller = StatusPoller(self)
        self._poller.updated.connect(self._on_update)

    # ---- cards ----

    def _build_docker_card(self) -> QFrame:
        f = _card()
        v = QVBoxLayout(f); v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)

        title = QLabel("DOCKER ENGINE"); title.setProperty("role", "cardTitle")
        title_row = QHBoxLayout(); title_row.setSpacing(6); title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title); title_row.addWidget(HelpBadge(*_DOCKER_HELP)); title_row.addStretch(1)
        row = QHBoxLayout()
        self.docker_value = QLabel("checking…")
        self.docker_value.setProperty("role", "cardValue")
        self.docker_pill = _pill("—", "neutral")
        row.addWidget(self.docker_value); row.addStretch(1); row.addWidget(self.docker_pill)

        self.docker_hint = QLabel(" "); self.docker_hint.setProperty("role", "caption")
        self.docker_hint.setWordWrap(True)

        self.docker_btn = QPushButton("Start Docker")
        self.docker_btn.setProperty("role", "primary")
        self.docker_btn.setCursor(Qt.PointingHandCursor)
        self.docker_btn.clicked.connect(self._on_start_daemon)
        self.docker_btn.hide()

        v.addLayout(title_row); v.addLayout(row); v.addWidget(self.docker_hint)
        v.addWidget(self.docker_btn, 0, Qt.AlignLeft)
        return f

    def _build_container_card(self) -> QFrame:
        f = _card()
        v = QVBoxLayout(f); v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)

        title = QLabel(f"CONTAINER · {CONTAINER_NAME}"); title.setProperty("role", "cardTitle")
        title_row = QHBoxLayout(); title_row.setSpacing(6); title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title); title_row.addWidget(HelpBadge(*_CONTAINER_HELP)); title_row.addStretch(1)
        row = QHBoxLayout()
        self.cont_value = QLabel("checking…")
        self.cont_value.setProperty("role", "cardValue")
        self.cont_pill = _pill("—", "neutral")
        row.addWidget(self.cont_value); row.addStretch(1); row.addWidget(self.cont_pill)

        self.cont_hint = QLabel(" "); self.cont_hint.setProperty("role", "caption")
        self.cont_hint.setWordWrap(True)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.cont_btn = QPushButton("Start container")
        self.cont_btn.setProperty("role", "primary")
        self.cont_btn.setCursor(Qt.PointingHandCursor)
        self.cont_btn.clicked.connect(self._on_start_container)
        self.cont_btn.hide()

        self.cont_help = QPushButton("Show create instructions")
        self.cont_help.setProperty("role", "secondary")
        self.cont_help.setCursor(Qt.PointingHandCursor)
        self.cont_help.clicked.connect(self._show_guide)
        self.cont_help.hide()

        btn_row.addWidget(self.cont_btn)
        btn_row.addWidget(self.cont_help)
        btn_row.addStretch(1)

        v.addLayout(title_row); v.addLayout(row); v.addWidget(self.cont_hint); v.addLayout(btn_row)
        return f

    def _build_gpu_card(self) -> QFrame:
        f = _card()
        v = QVBoxLayout(f); v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        title = QLabel("GPU"); title.setProperty("role", "cardTitle")
        title_row = QHBoxLayout(); title_row.setSpacing(6); title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title); title_row.addWidget(HelpBadge(*_GPU_HELP)); title_row.addStretch(1)
        self.gpu_name = QLabel("checking…"); self.gpu_name.setProperty("role", "cardValue")

        grid = QGridLayout(); grid.setHorizontalSpacing(42); grid.setVerticalSpacing(2)

        def _metric(caption: str) -> tuple[QLabel, QLabel]:
            cap = QLabel(caption); cap.setProperty("role", "cardTitle")
            val = QLabel("—"); val.setProperty("role", "metric")
            return cap, val

        (c1, self.gpu_util) = _metric("UTILIZATION")
        (c2, self.gpu_mem)  = _metric("MEMORY")
        (c3, self.gpu_temp) = _metric("TEMP")
        (c4, self.gpu_mem_util) = _metric("MEM I/O")

        for col, (cap, val) in enumerate([
            (c1, self.gpu_util), (c2, self.gpu_mem),
            (c3, self.gpu_temp), (c4, self.gpu_mem_util)
        ]):
            grid.addWidget(cap, 0, col)
            grid.addWidget(val, 1, col)

        v.addLayout(title_row); v.addWidget(self.gpu_name); v.addLayout(grid)
        return f

    # ---- update ----

    def _on_update(self, snap: _Snapshot) -> None:
        # Docker daemon card
        if snap.docker_version:
            self.docker_value.setText(f"v{snap.docker_version}")
            _set_pill(self.docker_pill, "● Running", "success")
            self.docker_hint.setText("Docker daemon is reachable.")
            self.docker_btn.hide()
        else:
            self.docker_value.setText("offline")
            _set_pill(self.docker_pill, "Stopped", "warning")
            self.docker_hint.setText(
                "Docker daemon isn't responding. Click to start it "
                "(you'll get a polkit prompt for your password)."
            )
            self.docker_btn.show()

        # Container card
        if snap.docker_version is None:
            self.cont_value.setText("unknown")
            _set_pill(self.cont_pill, "—", "neutral")
            self.cont_hint.setText("Start Docker first.")
            self.cont_btn.hide(); self.cont_help.hide()
        else:
            state = snap.container_state
            assert state is not None
            s = state.status
            if s == "running":
                self.cont_value.setText("running")
                _set_pill(self.cont_pill, "● Running", "success")
                self.cont_hint.setText(
                    "Ports 7007 (viser) and 6006 (tensorboard) forwarded to host."
                )
                self.cont_btn.hide(); self.cont_help.hide()
            elif s in ("exited", "created"):
                self.cont_value.setText(s)
                _set_pill(self.cont_pill, "Stopped", "warning")
                self.cont_hint.setText("Container exists but isn't running. Click Start.")
                self.cont_btn.show(); self.cont_help.hide()
            elif s == "paused":
                self.cont_value.setText("paused")
                _set_pill(self.cont_pill, "Paused", "warning")
                self.cont_hint.setText("Container is paused. Run `docker unpause splat-ns-v2` in a terminal.")
                self.cont_btn.hide(); self.cont_help.hide()
            elif s == "missing":
                if snap.image_exists:
                    self.cont_value.setText("not created")
                    _set_pill(self.cont_pill, "Missing", "error")
                    self.cont_hint.setText(
                        f"No container named '{CONTAINER_NAME}'. "
                        "This GUI won't create it — run the create command from USER_GUIDE.md."
                    )
                else:
                    self.cont_value.setText("image missing")
                    _set_pill(self.cont_pill, "Image not found", "error")
                    self.cont_hint.setText(
                        f"Image '{IMAGE_NAME}' isn't built yet. "
                        "This GUI won't build it — see USER_GUIDE.md."
                    )
                self.cont_btn.hide(); self.cont_help.show()
            else:
                self.cont_value.setText(s)
                _set_pill(self.cont_pill, s, "warning")
                self.cont_hint.setText("Unusual state — investigate with `docker inspect`.")
                self.cont_btn.hide(); self.cont_help.hide()

        # GPU card
        g = snap.gpu
        if g:
            self.gpu_name.setText(g.name)
            self.gpu_util.setText(f"{g.util_gpu}%")
            self.gpu_mem.setText(f"{g.mem_used_mb/1024:.1f} / {g.mem_total_mb/1024:.1f} GB")
            self.gpu_temp.setText(f"{g.temp_c}°C")
            self.gpu_mem_util.setText(f"{g.util_mem}%")
        else:
            self.gpu_name.setText("nvidia-smi unavailable")
            for lbl in (self.gpu_util, self.gpu_mem, self.gpu_temp, self.gpu_mem_util):
                lbl.setText("—")

    # ---- actions ----

    def _on_start_daemon(self) -> None:
        self.docker_btn.setEnabled(False)
        self.docker_btn.setText("Authorizing…")
        ok, msg = dock.start_daemon_via_polkit()
        self.docker_btn.setEnabled(True)
        self.docker_btn.setText("Start Docker")
        if not ok:
            QMessageBox.warning(self, "Couldn't start Docker", msg)

    def _on_start_container(self) -> None:
        self.cont_btn.setEnabled(False)
        self.cont_btn.setText("Starting…")
        try:
            dock.start_container(CONTAINER_NAME)
        except dock.DockerError as e:
            QMessageBox.warning(self, "Couldn't start container", str(e))
        finally:
            self.cont_btn.setEnabled(True)
            self.cont_btn.setText("Start container")

    def _show_guide(self) -> None:
        GuideDialog(self).exec()

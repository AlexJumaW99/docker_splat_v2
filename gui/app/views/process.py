"""
Process view — runs ns-process-data inside splat-ns.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, DATA_DIR
from app.ffprobe import VideoInfo, probe
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore
from app.widgets import ProjectBar, card, pill, section_title, set_pill


# ---------- helpers ----------

def _host_to_container(host_path: str | Path) -> str:
    p = Path(host_path).resolve()
    rel = p.relative_to(DATA_DIR.resolve())
    return f"/workspace/{rel.as_posix()}"


MATCH_METHODS = ["sequential", "vocab_tree", "exhaustive", "spatial"]


# ---------- async ffprobe ----------

class _ProbeSignals(QObject):
    done = Signal(object)   # VideoInfo | None


class _ProbeTask(QRunnable):
    def __init__(self, path: Path, sig: _ProbeSignals):
        super().__init__()
        self._path = path
        self._sig = sig
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self._sig.done.emit(probe(self._path))


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
            "A project is one video + its processed dataset + its runs. "
            "Create one to start."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(420)

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


# ---------- form ----------

class _Form(QWidget):
    """The per-project form. Rebuilt when the active project changes."""

    request_train = Signal()  # user clicked 'Go to Train'

    def __init__(self, store: ProjectStore, jobs: JobManager, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs
        self._project = project
        self._running_mine = False
        self._probe_signals = _ProbeSignals()
        self._probe_signals.done.connect(self._on_probe_done)

        self._build()
        self._load_project()
        self._wire_jobs()

        # Catch up if a job is already running (e.g. user switched projects
        # mid-run) so our UI reflects reality immediately.
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

        # Header
        header_row = QHBoxLayout()
        title = QLabel("Process")
        title.setObjectName("PageTitle")
        sub = QLabel("Prepare a video for splatfacto training.")
        sub.setProperty("role", "caption")

        self.advanced_cb = QCheckBox("Advanced mode")
        self.advanced_cb.toggled.connect(self._on_advanced)

        header_col = QVBoxLayout(); header_col.setSpacing(2)
        header_col.addWidget(title); header_col.addWidget(sub)
        header_row.addLayout(header_col, 1)
        header_row.addWidget(self.advanced_cb, 0, Qt.AlignBottom | Qt.AlignRight)
        bl.addLayout(header_row)

        bl.addWidget(self._video_card())
        bl.addWidget(self._params_card())
        bl.addWidget(self._output_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- video card ----
    def _video_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)
        v.addWidget(section_title("VIDEO"))

        row = QHBoxLayout(); row.setSpacing(8)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setProperty("role", "readout")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setProperty("role", "secondary")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._on_browse_video)
        row.addWidget(self.video_path_edit, 1)
        row.addWidget(self.browse_btn)
        v.addLayout(row)

        self.meta_lbl = QLabel("")
        self.meta_lbl.setProperty("role", "caption")
        v.addWidget(self.meta_lbl)
        return c

    # ---- params card ----
    def _params_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(section_title("PARAMETERS"))
        top.addStretch(1)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        self.save_preset_btn = QPushButton("Save as preset…")
        self.save_preset_btn.setProperty("role", "ghost")
        self.save_preset_btn.setCursor(Qt.PointingHandCursor)
        self.save_preset_btn.clicked.connect(self._on_save_preset)

        top.addWidget(QLabel("Preset"))
        top.addWidget(self.preset_combo)
        top.addWidget(self.save_preset_btn)
        v.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14); grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)

        # Target frames
        grid.addWidget(QLabel("Target frames"), 0, 0)
        frames_row = QHBoxLayout(); frames_row.setSpacing(10)
        self.frames_slider = QSlider(Qt.Horizontal)
        self.frames_slider.setRange(50, 500); self.frames_slider.setSingleStep(10)
        self.frames_slider.setPageStep(25)
        self.frames_value = QSpinBox()
        self.frames_value.setRange(50, 500); self.frames_value.setSingleStep(10)
        self.frames_value.setMinimumWidth(80)
        self.frames_slider.valueChanged.connect(self.frames_value.setValue)
        self.frames_value.valueChanged.connect(self.frames_slider.setValue)
        frames_row.addWidget(self.frames_slider, 1)
        frames_row.addWidget(self.frames_value)
        w = QWidget(); w.setLayout(frames_row)
        grid.addWidget(w, 0, 1)

        # Matching method
        grid.addWidget(QLabel("Matching method"), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(MATCH_METHODS)
        grid.addWidget(self.method_combo, 1, 1)

        # Downscale levels
        grid.addWidget(QLabel("Downscale levels"), 2, 0)
        self.downscales_spin = QSpinBox()
        self.downscales_spin.setRange(0, 5)
        self.downscales_spin.setMaximumWidth(80)
        ds_row = QHBoxLayout(); ds_row.addWidget(self.downscales_spin); ds_row.addStretch(1)
        ds_wrap = QWidget(); ds_wrap.setLayout(ds_row)
        grid.addWidget(ds_wrap, 2, 1)

        # Verbose
        self.verbose_cb = QCheckBox("Verbose output")
        grid.addWidget(self.verbose_cb, 3, 1)

        v.addLayout(grid)

        # Advanced panel
        self.advanced_wrap = QFrame()
        self.advanced_wrap.setProperty("role", "subcard")
        av = QVBoxLayout(self.advanced_wrap)
        av.setContentsMargins(0, 8, 0, 0); av.setSpacing(8)

        adv_title = section_title("ADVANCED · RAW CLI FLAGS")
        av.addWidget(adv_title)

        self.extra_flags_edit = QLineEdit()
        self.extra_flags_edit.setPlaceholderText(
            "--refine-pixsfm --camera-type perspective …"
        )
        av.addWidget(self.extra_flags_edit)

        hint = QLabel(
            "Passed through verbatim to ns-process-data. Use for flags not in the basic form. "
            "See: docker exec splat-ns ns-process-data video --help"
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        av.addWidget(hint)

        self.advanced_wrap.hide()
        v.addWidget(self.advanced_wrap)
        return c

    # ---- output card ----
    def _output_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        v.addWidget(section_title("OUTPUT DATASET"))
        self.output_lbl = QLineEdit()
        self.output_lbl.setReadOnly(True)
        self.output_lbl.setProperty("role", "readout")
        v.addWidget(self.output_lbl)
        hint = QLabel(
            "Host path. Container sees this as /workspace/processed/… automatically."
        )
        hint.setProperty("role", "caption")
        v.addWidget(hint)
        return c

    # ---- actions ----
    def _actions_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_btn = QPushButton("Run processing")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self._on_run)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("role", "secondary")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)

        self.status_pill = pill("Ready", "neutral")

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.status_pill)

        self.ready_to_train = QFrame()
        self.ready_to_train.setProperty("role", "subcard")
        rtt = QHBoxLayout(self.ready_to_train)
        rtt.setContentsMargins(14, 10, 14, 10)
        ok = QLabel("✓ Processed dataset ready.")
        ok.setStyleSheet("color: #22C55E; font-weight: 600;")
        train_btn = QPushButton("Go to Train →")
        train_btn.setProperty("role", "primary")
        train_btn.setCursor(Qt.PointingHandCursor)
        train_btn.clicked.connect(self.request_train.emit)
        rtt.addWidget(ok)
        rtt.addStretch(1)
        rtt.addWidget(train_btn)
        self.ready_to_train.hide()

        v.addLayout(row)
        v.addWidget(self.ready_to_train)
        return c

    # ---- advanced toggle ----
    def _on_advanced(self, checked: bool) -> None:
        self.advanced_wrap.setVisible(checked)

    # ---- project lifecycle ----
    def _load_project(self) -> None:
        p = self._project
        self.video_path_edit.setText(p.video_path)
        self.output_lbl.setText(p.processed_dir)

        # Populate form from last params or sensible defaults.
        params = p.last_process_params or {}
        self.frames_slider.setValue(int(params.get("num_frames_target", 250)))
        self.frames_value.setValue(int(params.get("num_frames_target", 250)))
        method = params.get("matching_method", "sequential")
        idx = MATCH_METHODS.index(method) if method in MATCH_METHODS else 0
        self.method_combo.setCurrentIndex(idx)
        self.downscales_spin.setValue(int(params.get("num_downscales", 3)))
        self.verbose_cb.setChecked(bool(params.get("verbose", True)))
        self.extra_flags_edit.setText(params.get("extra_flags", ""))

        self._refresh_presets()
        self._probe_video(Path(p.video_path))

    def _refresh_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— custom —", userData=None)
        for name in sorted(self._project.presets_process.keys()):
            self.preset_combo.addItem(name, userData=name)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int) -> None:
        name = self.preset_combo.itemData(idx)
        if not name:
            return
        preset = self._project.presets_process.get(name, {})
        if preset:
            self.frames_value.setValue(int(preset.get("num_frames_target", 250)))
            method = preset.get("matching_method", "sequential")
            if method in MATCH_METHODS:
                self.method_combo.setCurrentIndex(MATCH_METHODS.index(method))
            self.downscales_spin.setValue(int(preset.get("num_downscales", 3)))
            self.verbose_cb.setChecked(bool(preset.get("verbose", True)))
            self.extra_flags_edit.setText(preset.get("extra_flags", ""))

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._store.save_preset(self._project.id, "process", name, self._collect_params())
        self._refresh_presets()
        idx = self.preset_combo.findData(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    # ---- video probe ----
    def _on_browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video",
            str(DATA_DIR),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._store.update_video_path(self._project.id, Path(path))
        except ValueError as e:
            QMessageBox.warning(self, "Video outside data/", str(e))
            return
        self.video_path_edit.setText(path)
        self._probe_video(Path(path))

    def _probe_video(self, path: Path) -> None:
        self.meta_lbl.setText("reading video metadata…")
        QThreadPool.globalInstance().start(_ProbeTask(path, self._probe_signals))

    def _on_probe_done(self, info: VideoInfo | None) -> None:
        if info is None:
            self.meta_lbl.setText("video not found")
            return
        self.meta_lbl.setText(info.pretty())

    # ---- collect + run ----
    def _collect_params(self) -> dict:
        return {
            "num_frames_target": self.frames_value.value(),
            "matching_method": self.method_combo.currentText(),
            "num_downscales": self.downscales_spin.value(),
            "verbose": self.verbose_cb.isChecked(),
            "extra_flags": self.extra_flags_edit.text().strip(),
        }

    def _on_run(self) -> None:
        if self._jobs.is_running():
            QMessageBox.information(self, "Busy",
                "Another job is already running. Cancel it first.")
            return

        state = dock.container_state(CONTAINER_NAME)
        if state.status != "running":
            QMessageBox.warning(
                self, "Container not running",
                f"The {CONTAINER_NAME} container isn't running. "
                "Start it from the Dashboard, then try again.",
            )
            return

        video = Path(self.video_path_edit.text())
        if not video.exists():
            QMessageBox.warning(self, "Video not found", f"No such file:\n{video}")
            return

        params = self._collect_params()
        self._store.update_last_process_params(self._project.id, params)

        try:
            c_video = _host_to_container(video)
            c_out = _host_to_container(self._project.processed_dir)
        except ValueError as e:
            QMessageBox.warning(self, "Path outside data/", str(e))
            return

        argv = [
            "ns-process-data", "video",
            "--data", c_video,
            "--output-dir", c_out,
            "--num-frames-target", str(params["num_frames_target"]),
            "--matching-method", params["matching_method"],
            "--num-downscales", str(params["num_downscales"]),
        ]
        if params["verbose"]:
            argv.append("--verbose")
        if params["extra_flags"]:
            try:
                extras = shlex.split(params["extra_flags"])
            except ValueError as e:
                QMessageBox.warning(self, "Bad extra flags", str(e))
                return
            argv.extend(extras)

        display = f"docker exec {CONTAINER_NAME} " + " ".join(shlex.quote(a) for a in argv)

        spec = JobSpec(
            kind="process",
            display_cmd=display,
            container_argv=argv,
            kill_pattern="ns-process-data",
        )
        self.ready_to_train.hide()
        self._jobs.start(spec)

    def _on_cancel(self) -> None:
        self._jobs.cancel()

    # ---- job signals ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _on_job_started(self, spec: JobSpec) -> None:
        if spec.kind != "process":
            self._running_mine = False
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Other job running", "warning")
            return
        self._running_mine = True
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        set_pill(self.status_pill, "● Running", "warning")

    def _on_job_finished(self, code: int) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            set_pill(self.status_pill, "Done", "success")
            self.ready_to_train.show()
        else:
            set_pill(self.status_pill, f"Failed · exit {code}", "error")

    def _on_job_cancelled(self) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Cancelled", "error")

    def _on_job_failed(self, msg: str) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Failed to start", "error")

    def _reset_idle(self) -> None:
        """Another kind's job just ended — restore our idle UI."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Ready", "neutral")


# ---------- top-level view ----------

class ProcessView(QWidget):
    """Top-level Process tab. Host for ProjectBar + per-project form (or empty state)."""

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

        self._rebuild()
        self._store.active_changed.connect(lambda _p: self._rebuild())
        self._store.projects_changed.connect(self._rebuild)

    def _rebuild(self) -> None:
        # clear existing
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        active = self._store.active
        if active is None:
            self.stack.addWidget(_EmptyState(self._store))
            return
        form = _Form(self._store, self._jobs, active)
        form.request_train.connect(lambda: self.request_navigate.emit(2))  # Train tab idx
        self.stack.addWidget(form)

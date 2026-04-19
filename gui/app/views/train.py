"""
Train view — runs ns-train inside splat-ns.

Mirrors the Process tab: ProjectBar + per-project form, with a not-processed
gate for projects that haven't been through ns-process-data yet. Presets live
per-project in presets_train. Job signals are filtered by kind so that a
concurrent process/train run can't cross-wire UI state.
"""
from __future__ import annotations

import shlex
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSlider,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from app import docker_service as dock
from app.config import CONTAINER_NAME, DATA_DIR, VISER_PORT
from app.jobs import JobManager, JobSpec
from app.store import Project, ProjectStore
from app.widgets import ProjectBar, card, pill, section_title, set_pill


# ---------- constants ----------

OUTPUTS_DIR = DATA_DIR / "outputs"

METHODS = ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]
VIS_MODES = ["viewer+tensorboard", "viewer", "tensorboard"]


# ---------- helpers ----------

def _host_to_container(host_path: str | Path) -> str:
    p = Path(host_path).resolve()
    rel = p.relative_to(DATA_DIR.resolve())
    return f"/workspace/{rel.as_posix()}"


def _is_processed(processed_dir: str) -> bool:
    """A Nerfstudio-ready dataset always has transforms.json at its root."""
    return (Path(processed_dir) / "transforms.json").is_file()


def _count_frames(processed_dir: str) -> int | None:
    images = Path(processed_dir) / "images"
    if not images.is_dir():
        return None
    return sum(1 for p in images.iterdir() if p.is_file())


# ---------- empty state (no project) ----------

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
            "Training needs a processed dataset. "
            "Create a project on the Process tab to get started."
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


# ---------- not-processed gate ----------

class _NotProcessedState(QWidget):
    """Active project exists but has no transforms.json — push user to Process."""

    request_process = Signal()

    def __init__(self, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 80, 40, 40)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.setSpacing(10)

        title = QLabel("Not processed yet")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel(
            f"'{project.name}' has no processed dataset yet. "
            "Run it through Process first, then come back."
        )
        sub.setProperty("role", "caption")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setMaximumWidth(480)

        btn = QPushButton("Go to Process →")
        btn.setProperty("role", "primary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.request_process.emit)

        v.addWidget(title)
        v.addWidget(sub, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        v.addWidget(btn, 0, Qt.AlignHCenter)
        v.addStretch(1)


# ---------- form ----------

class _Form(QWidget):
    """Per-project training form. Rebuilt when the active project changes."""

    request_viewers = Signal()
    request_process = Signal()

    def __init__(self, store: ProjectStore, jobs: JobManager, project: Project):
        super().__init__()
        self.setObjectName("Canvas")
        self._store = store
        self._jobs = jobs
        self._project = project
        self._running_mine = False

        # Poll for viser readiness (port 7007) once training has started.
        self._viser_poll = QTimer(self)
        self._viser_poll.setInterval(1000)
        self._viser_poll.timeout.connect(self._check_viser)

        self._build()
        self._load_project()
        self._wire_jobs()

        # If a job is already running when this form is constructed (e.g. user
        # switched projects mid-run), catch up to that state.
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
        title = QLabel("Train")
        title.setObjectName("PageTitle")
        sub = QLabel("Train a gaussian splat from the processed dataset.")
        sub.setProperty("role", "caption")

        self.advanced_cb = QCheckBox("Advanced mode")
        self.advanced_cb.toggled.connect(self._on_advanced)

        header_col = QVBoxLayout(); header_col.setSpacing(2)
        header_col.addWidget(title); header_col.addWidget(sub)
        header_row.addLayout(header_col, 1)
        header_row.addWidget(self.advanced_cb, 0, Qt.AlignBottom | Qt.AlignRight)
        bl.addLayout(header_row)

        bl.addWidget(self._dataset_card())
        bl.addWidget(self._params_card())
        bl.addWidget(self._output_card())
        bl.addWidget(self._actions_card())

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- dataset card ----
    def _dataset_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(8)
        v.addWidget(section_title("DATASET"))

        self.dataset_path = QLineEdit()
        self.dataset_path.setReadOnly(True)
        self.dataset_path.setProperty("role", "readout")
        v.addWidget(self.dataset_path)

        self.dataset_meta = QLabel("")
        self.dataset_meta.setProperty("role", "caption")
        v.addWidget(self.dataset_meta)
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

        # Method
        grid.addWidget(QLabel("Method"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        grid.addWidget(self.method_combo, 0, 1)

        # Max iterations
        grid.addWidget(QLabel("Max iterations"), 1, 0)
        it_row = QHBoxLayout(); it_row.setSpacing(10)
        self.iters_slider = QSlider(Qt.Horizontal)
        self.iters_slider.setRange(1000, 30000)
        self.iters_slider.setSingleStep(500)
        self.iters_slider.setPageStep(1000)
        self.iters_value = QSpinBox()
        self.iters_value.setRange(1000, 30000)
        self.iters_value.setSingleStep(500)
        self.iters_value.setMinimumWidth(90)
        self.iters_slider.valueChanged.connect(self.iters_value.setValue)
        self.iters_value.valueChanged.connect(self.iters_slider.setValue)
        it_row.addWidget(self.iters_slider, 1)
        it_row.addWidget(self.iters_value)
        it_wrap = QWidget(); it_wrap.setLayout(it_row)
        grid.addWidget(it_wrap, 1, 1)

        # Steps per save
        grid.addWidget(QLabel("Steps per save"), 2, 0)
        self.save_step_spin = QSpinBox()
        self.save_step_spin.setRange(100, 10000)
        self.save_step_spin.setSingleStep(100)
        self.save_step_spin.setMaximumWidth(100)
        ss_row = QHBoxLayout(); ss_row.addWidget(self.save_step_spin); ss_row.addStretch(1)
        ss_wrap = QWidget(); ss_wrap.setLayout(ss_row)
        grid.addWidget(ss_wrap, 2, 1)

        # Visualization
        grid.addWidget(QLabel("Visualization"), 3, 0)
        self.vis_combo = QComboBox()
        self.vis_combo.addItems(VIS_MODES)
        grid.addWidget(self.vis_combo, 3, 1)

        # Quit on completion
        self.quit_on_completion_cb = QCheckBox("Quit viewer when training finishes")
        grid.addWidget(self.quit_on_completion_cb, 4, 1)

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
            "--pipeline.model.cull-alpha-thresh 0.005 …"
        )
        av.addWidget(self.extra_flags_edit)

        hint = QLabel(
            "Passed through verbatim to ns-train. "
            "See: docker exec splat-ns ns-train splatfacto --help"
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
        v.addWidget(section_title("OUTPUT RUN"))
        self.output_lbl = QLineEdit()
        self.output_lbl.setReadOnly(True)
        self.output_lbl.setProperty("role", "readout")
        v.addWidget(self.output_lbl)
        hint = QLabel(
            "Host path. ns-train appends /<method>/<timestamp>/ automatically "
            "so every launch gets its own run folder."
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return c

    # ---- actions ----
    def _actions_card(self) -> QFrame:
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(20, 16, 20, 18); v.setSpacing(10)

        row = QHBoxLayout(); row.setSpacing(10)
        self.run_btn = QPushButton("Run training")
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

        # Viser-ready banner — shows once port 7007 starts listening.
        self.viser_card = QFrame()
        self.viser_card.setProperty("role", "subcard")
        vs = QHBoxLayout(self.viser_card)
        vs.setContentsMargins(14, 10, 14, 10)
        ok = QLabel("✓ Viser server is live.")
        ok.setStyleSheet("color: #22C55E; font-weight: 600;")
        self.open_viser_btn = QPushButton(f"Open http://localhost:{VISER_PORT}")
        self.open_viser_btn.setProperty("role", "primary")
        self.open_viser_btn.setCursor(Qt.PointingHandCursor)
        self.open_viser_btn.clicked.connect(self._open_viser_in_browser)
        self.viewers_btn = QPushButton("Go to Viewers →")
        self.viewers_btn.setProperty("role", "secondary")
        self.viewers_btn.setCursor(Qt.PointingHandCursor)
        self.viewers_btn.clicked.connect(self.request_viewers.emit)
        vs.addWidget(ok)
        vs.addStretch(1)
        vs.addWidget(self.open_viser_btn)
        vs.addWidget(self.viewers_btn)
        self.viser_card.hide()

        v.addLayout(row)
        v.addWidget(self.viser_card)
        return c

    # ---- advanced toggle ----
    def _on_advanced(self, checked: bool) -> None:
        self.advanced_wrap.setVisible(checked)

    # ---- project lifecycle ----
    def _load_project(self) -> None:
        p = self._project
        self.dataset_path.setText(p.processed_dir)

        n = _count_frames(p.processed_dir)
        meta = "✓ transforms.json present"
        if n is not None:
            meta += f"  ·  {n} frames"
        self.dataset_meta.setText(meta)

        self.output_lbl.setText(str(OUTPUTS_DIR / p.id))

        params = p.last_train_params or {}
        method = params.get("method", "splatfacto")
        if method in METHODS:
            self.method_combo.setCurrentIndex(METHODS.index(method))
        iters = int(params.get("max_num_iterations", 7000))
        self.iters_slider.setValue(iters); self.iters_value.setValue(iters)
        self.save_step_spin.setValue(int(params.get("steps_per_save", 2000)))
        vis = params.get("vis", "viewer+tensorboard")
        if vis in VIS_MODES:
            self.vis_combo.setCurrentIndex(VIS_MODES.index(vis))
        self.quit_on_completion_cb.setChecked(
            bool(params.get("quit_on_train_completion", False))
        )
        self.extra_flags_edit.setText(params.get("extra_flags", ""))

        self._refresh_presets()

    def _refresh_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— custom —", userData=None)
        for name in sorted(self._project.presets_train.keys()):
            self.preset_combo.addItem(name, userData=name)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int) -> None:
        name = self.preset_combo.itemData(idx)
        if not name:
            return
        preset = self._project.presets_train.get(name, {})
        if not preset:
            return
        method = preset.get("method", "splatfacto")
        if method in METHODS:
            self.method_combo.setCurrentIndex(METHODS.index(method))
        iters = int(preset.get("max_num_iterations", 7000))
        self.iters_slider.setValue(iters); self.iters_value.setValue(iters)
        self.save_step_spin.setValue(int(preset.get("steps_per_save", 2000)))
        vis = preset.get("vis", "viewer+tensorboard")
        if vis in VIS_MODES:
            self.vis_combo.setCurrentIndex(VIS_MODES.index(vis))
        self.quit_on_completion_cb.setChecked(
            bool(preset.get("quit_on_train_completion", False))
        )
        self.extra_flags_edit.setText(preset.get("extra_flags", ""))

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._store.save_preset(
            self._project.id, "train", name, self._collect_params()
        )
        self._refresh_presets()
        idx = self.preset_combo.findData(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    # ---- collect + run ----
    def _collect_params(self) -> dict:
        return {
            "method": self.method_combo.currentText(),
            "max_num_iterations": self.iters_value.value(),
            "steps_per_save": self.save_step_spin.value(),
            "vis": self.vis_combo.currentText(),
            "quit_on_train_completion": self.quit_on_completion_cb.isChecked(),
            "extra_flags": self.extra_flags_edit.text().strip(),
        }

    def _on_run(self) -> None:
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

        if not _is_processed(self._project.processed_dir):
            QMessageBox.warning(
                self, "Missing dataset",
                "The processed dataset is gone. Re-run Process first.",
            )
            self.request_process.emit()
            return

        params = self._collect_params()

        # If the viewer was requested, make sure nothing INSIDE the container
        # is already bound to viser's port. The host-side port is always held
        # by docker-proxy while splat-ns is running — that's the port mapping
        # itself and is expected. A real conflict only exists when a leftover
        # ns-train/ns-viewer is still holding :VISER_PORT inside the container.
        if "viewer" in params["vis"]:
            busy = dock.container_port_in_use(VISER_PORT)
            if busy is True:
                resp = QMessageBox.question(
                    self, "Viser port is busy inside the container",
                    f"Something inside {CONTAINER_NAME} is already bound to "
                    f":{VISER_PORT} — most likely a leftover ns-train or "
                    "ns-viewer from a previous run.\n\n"
                    "Terminate it (SIGTERM → SIGKILL) and continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if resp != QMessageBox.Yes:
                    return
                if not dock.free_container_port(VISER_PORT):
                    QMessageBox.warning(
                        self, "Couldn't free port",
                        f"Port {VISER_PORT} inside {CONTAINER_NAME} is still "
                        "busy after SIGKILL. Inspect with:\n\n"
                        f"  docker exec {CONTAINER_NAME} ps -ef\n\n"
                        "and stop the offender manually.",
                    )
                    return

        self._store.update_last_train_params(self._project.id, params)

        try:
            c_data = _host_to_container(self._project.processed_dir)
            c_out = _host_to_container(OUTPUTS_DIR)
        except ValueError as e:
            QMessageBox.warning(self, "Path outside data/", str(e))
            return

        argv = [
            "ns-train", params["method"],
            "--experiment-name", self._project.id,
            "--data", c_data,
            "--output-dir", c_out,
            "--max-num-iterations", str(params["max_num_iterations"]),
            "--steps-per-save", str(params["steps_per_save"]),
            "--vis", params["vis"],
        ]
        if "viewer" in params["vis"]:
            argv.extend([
                "--viewer.websocket-host", "0.0.0.0",
                "--viewer.quit-on-train-completion",
                "True" if params["quit_on_train_completion"] else "False",
            ])
        if params["extra_flags"]:
            try:
                extras = shlex.split(params["extra_flags"])
            except ValueError as e:
                QMessageBox.warning(self, "Bad extra flags", str(e))
                return
            argv.extend(extras)

        display = (
            f"docker exec {CONTAINER_NAME} "
            + " ".join(shlex.quote(a) for a in argv)
        )

        spec = JobSpec(
            kind="train",
            display_cmd=display,
            container_argv=argv,
            kill_pattern="ns-train",
        )
        self.viser_card.hide()
        self._jobs.start(spec)

    def _on_cancel(self) -> None:
        self._jobs.cancel()

    # ---- viser readiness ----
    def _check_viser(self) -> None:
        # Probe inside the container, not the host — docker-proxy's host-side
        # listen is a false positive (it accepts before the backend exists).
        if dock.container_port_in_use(VISER_PORT) is True:
            self._viser_poll.stop()
            self.viser_card.show()

    def _open_viser_in_browser(self) -> None:
        webbrowser.open(f"http://localhost:{VISER_PORT}")

    # ---- job signals (filtered by kind) ----
    def _wire_jobs(self) -> None:
        self._jobs.started.connect(self._on_job_started)
        self._jobs.finished.connect(self._on_job_finished)
        self._jobs.cancelled.connect(self._on_job_cancelled)
        self._jobs.failed.connect(self._on_job_failed)

    def _on_job_started(self, spec: JobSpec) -> None:
        if spec.kind != "train":
            self._running_mine = False
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            set_pill(self.status_pill, "Other job running", "warning")
            return
        self._running_mine = True
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        set_pill(self.status_pill, "● Training", "warning")
        if "viewer" in self._collect_params()["vis"]:
            self._viser_poll.start()

    def _on_job_finished(self, code: int) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            set_pill(self.status_pill, "Done", "success")
        else:
            set_pill(self.status_pill, f"Failed · exit {code}", "error")

    def _on_job_cancelled(self) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self.viser_card.hide()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Cancelled", "error")

    def _on_job_failed(self, msg: str) -> None:
        if not self._running_mine:
            self._reset_idle()
            return
        self._running_mine = False
        self._viser_poll.stop()
        self.viser_card.hide()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Failed to start", "error")

    def _reset_idle(self) -> None:
        """Called when another kind's job ends — restore our idle UI."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        set_pill(self.status_pill, "Ready", "neutral")


# ---------- top-level view ----------

class TrainView(QWidget):
    """Top-level Train tab. Hosts ProjectBar + per-project form / gates."""

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

        # When a process job succeeds, the dataset may have just become valid;
        # re-evaluate so the not-processed gate drops out automatically.
        self._last_started_kind: str | None = None
        self._jobs.started.connect(self._on_any_started)
        self._jobs.finished.connect(self._on_any_finished)

    def _on_any_started(self, spec: JobSpec) -> None:
        self._last_started_kind = spec.kind

    def _on_any_finished(self, code: int) -> None:
        was_process = self._last_started_kind == "process"
        self._last_started_kind = None
        if code != 0 or not was_process:
            return
        # Only rebuild if the user is still on the gate. A live _Form owns
        # in-flight edits we don't want to discard.
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

        if not _is_processed(active.processed_dir):
            gate = _NotProcessedState(active)
            gate.request_process.connect(lambda: self.request_navigate.emit(1))
            self.stack.addWidget(gate)
            return

        form = _Form(self._store, self._jobs, active)
        form.request_viewers.connect(lambda: self.request_navigate.emit(3))
        form.request_process.connect(lambda: self.request_navigate.emit(1))
        self.stack.addWidget(form)

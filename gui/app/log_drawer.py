"""
Global log drawer — a collapsible bottom panel that displays live output from
the active JobManager. Shared across all tabs; survives tab switches.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.jobs import JobManager, JobSpec


class LogDrawer(QFrame):
    """A thin frame that can expand to show a 200+ px log area or collapse
    to just the status strip."""

    COLLAPSED_HEIGHT = 40
    EXPANDED_HEIGHT = 240

    def __init__(self, jobs: JobManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("LogDrawer")
        self.setFrameShape(QFrame.NoFrame)
        self._jobs = jobs
        self._start_ts: float | None = None

        self._build()
        self._wire()

        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._update_elapsed)

        self._expanded = True
        self._apply_state()

    # ---- build ----
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header strip
        header = QFrame(objectName="LogHeader")
        header.setFixedHeight(self.COLLAPSED_HEIGHT)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 10, 0)
        hl.setSpacing(10)

        self.toggle_btn = QPushButton("▾")
        self.toggle_btn.setObjectName("LogToggle")
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setFixedSize(24, 24)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("LogDot")
        self.status_dot.setProperty("state", "idle")

        self.status_lbl = QLabel("Idle — no job running")
        self.status_lbl.setProperty("role", "caption")

        self.elapsed_lbl = QLabel("")
        self.elapsed_lbl.setProperty("role", "mono_dim")

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setProperty("role", "ghost")
        self.copy_btn.setCursor(Qt.PointingHandCursor)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("role", "ghost")
        self.clear_btn.setCursor(Qt.PointingHandCursor)

        hl.addWidget(self.toggle_btn)
        hl.addWidget(self.status_dot)
        hl.addWidget(self.status_lbl, 1)
        hl.addWidget(self.elapsed_lbl)
        hl.addSpacing(8)
        hl.addWidget(self.copy_btn)
        hl.addWidget(self.clear_btn)

        # Log area
        self.log = QPlainTextEdit()
        self.log.setObjectName("LogArea")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root.addWidget(header)
        root.addWidget(self.log, 1)

    # ---- wire ----
    def _wire(self) -> None:
        self.toggle_btn.clicked.connect(self.toggle)
        self.copy_btn.clicked.connect(self._copy_all)
        self.clear_btn.clicked.connect(self.log.clear)

        self._jobs.started.connect(self._on_started)
        self._jobs.line.connect(self._on_line)
        self._jobs.finished.connect(self._on_finished)
        self._jobs.cancelled.connect(self._on_cancelled)
        self._jobs.failed.connect(self._on_failed)

    # ---- toggle ----
    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._apply_state()

    def _apply_state(self) -> None:
        if self._expanded:
            self.setFixedHeight(self.EXPANDED_HEIGHT)
            self.toggle_btn.setText("▾")
            self.log.show()
        else:
            self.setFixedHeight(self.COLLAPSED_HEIGHT)
            self.toggle_btn.setText("▸")
            self.log.hide()

    # ---- job signals ----
    def _set_state(self, state: str) -> None:
        self.status_dot.setProperty("state", state)
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)

    def _on_started(self, spec: JobSpec) -> None:
        self.log.clear()
        self.log.appendPlainText(f"$ {spec.display_cmd}\n")
        self._start_ts = time.monotonic()
        self._tick.start()
        self._set_state("running")
        self.status_lbl.setText(f"Running · {spec.kind}")
        if not self._expanded:
            self.toggle()

    def _on_line(self, text: str) -> None:
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, code: int) -> None:
        self._tick.stop()
        final = self._fmt_elapsed()
        if code == 0:
            self._set_state("ok")
            self.status_lbl.setText("Finished")
            self.log.appendPlainText(f"\n✓ Finished in {final} (exit 0)")
        else:
            self._set_state("err")
            self.status_lbl.setText(f"Failed · exit {code}")
            self.log.appendPlainText(f"\n✗ Exited with code {code} after {final}")

    def _on_cancelled(self) -> None:
        self._tick.stop()
        self._set_state("err")
        self.status_lbl.setText("Cancelled")
        self.log.appendPlainText(f"\n■ Cancelled after {self._fmt_elapsed()}")

    def _on_failed(self, msg: str) -> None:
        self._tick.stop()
        self._set_state("err")
        self.status_lbl.setText("Failed to start")
        self.log.appendPlainText(f"\n✗ {msg}")

    # ---- elapsed ----
    def _update_elapsed(self) -> None:
        self.elapsed_lbl.setText(self._fmt_elapsed())

    def _fmt_elapsed(self) -> str:
        if self._start_ts is None:
            return ""
        s = int(time.monotonic() - self._start_ts)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"

    # ---- copy ----
    def _copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.log.toPlainText())

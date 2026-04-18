"""Shared small widgets: cards, section titles, project bar, new-project dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from app.config import DATA_DIR
from app.store import Project, ProjectStore


# ---------- primitives ----------

def card() -> QFrame:
    f = QFrame()
    f.setProperty("role", "card")
    return f


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "cardTitle")
    return lbl


def pill(text: str, kind: str = "neutral") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("pill", kind)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def set_pill(lbl: QLabel, text: str, kind: str) -> None:
    lbl.setText(text)
    lbl.setProperty("pill", kind)
    lbl.style().unpolish(lbl); lbl.style().polish(lbl)


# ---------- project bar ----------

class ProjectBar(QFrame):
    """Top-of-page bar that shows the active project and lets the user switch
    or create a new one. Syncs to ProjectStore on both sides."""

    def __init__(self, store: ProjectStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ProjectBar")
        self._store = store

        h = QHBoxLayout(self)
        h.setContentsMargins(18, 10, 18, 10)
        h.setSpacing(12)

        lbl = QLabel("PROJECT")
        lbl.setProperty("role", "cardTitle")

        self.combo = QComboBox()
        self.combo.setMinimumWidth(260)
        self.combo.setCursor(Qt.PointingHandCursor)

        self.new_btn = QPushButton("+  New project")
        self.new_btn.setProperty("role", "secondary")
        self.new_btn.setCursor(Qt.PointingHandCursor)

        h.addWidget(lbl)
        h.addWidget(self.combo)
        h.addStretch(1)
        h.addWidget(self.new_btn)

        self._suppress = False
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.new_btn.clicked.connect(self._on_new)
        self._store.projects_changed.connect(self._refresh)
        self._store.active_changed.connect(lambda _p: self._refresh())

        self._refresh()

    def _refresh(self) -> None:
        self._suppress = True
        self.combo.clear()
        projects = self._store.projects
        for p in projects:
            self.combo.addItem(p.name, userData=p.id)

        if not projects:
            self.combo.addItem("— no projects —", userData=None)
            self.combo.setEnabled(False)
        else:
            self.combo.setEnabled(True)
            active = self._store.active
            if active:
                idx = next((i for i, p in enumerate(projects) if p.id == active.id), 0)
                self.combo.setCurrentIndex(idx)
        self._suppress = False

    def _on_combo_changed(self, idx: int) -> None:
        if self._suppress:
            return
        pid = self.combo.itemData(idx)
        if pid:
            self._store.set_active(pid)

    def _on_new(self) -> None:
        dlg = NewProjectDialog(self._store, self)
        dlg.exec()


# ---------- new project dialog ----------

class NewProjectDialog(QDialog):
    def __init__(self, store: ProjectStore, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 18)
        v.setSpacing(10)

        title = QLabel("New project")
        title.setObjectName("PageTitle")
        v.addWidget(title)

        hint = QLabel(
            "A project bundles one input video with its processed dataset "
            "and training runs."
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addSpacing(6)

        v.addWidget(section_title("NAME"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. kitchen_v1")
        v.addWidget(self.name_edit)

        v.addSpacing(4)
        v.addWidget(section_title("VIDEO"))
        row = QHBoxLayout(); row.setSpacing(8)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(str(DATA_DIR) + "/yourvideo.mp4")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setProperty("role", "secondary")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_btn)
        v.addLayout(row)

        constraint = QLabel(
            f"Must live inside {DATA_DIR}.  "
            "That's the folder bind-mounted into the container as /workspace."
        )
        constraint.setProperty("role", "caption")
        constraint.setWordWrap(True)
        v.addWidget(constraint)

        v.addSpacing(8)
        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.create_btn = btns.addButton("Create project", QDialogButtonBox.AcceptRole)
        self.create_btn.setProperty("role", "primary")
        self.create_btn.setCursor(Qt.PointingHandCursor)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _browse(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        start_dir = str(DATA_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video",
            start_dir,
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if path:
            self.path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).stem)

    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        raw = self.path_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Give this project a name.")
            return
        if not raw:
            QMessageBox.warning(self, "Missing video", "Pick a video file.")
            return
        path = Path(raw).expanduser()
        if not path.exists():
            QMessageBox.warning(self, "Video not found", f"No such file:\n{path}")
            return
        try:
            self._store.create(name, path)
        except ValueError as e:
            QMessageBox.warning(self, "Can't create project", str(e))
            return
        self.accept()

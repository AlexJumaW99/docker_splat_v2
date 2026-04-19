"""Shared small widgets: cards, section titles, project bar, new-project dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from app.config import DATA_DIR
from app.store import Project, ProjectStore, detect_dataset_format


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

_MODE_VIDEO = 0
_MODE_IMPORT = 1


class NewProjectDialog(QDialog):
    """Create a project either from a raw video (→ Process → Train) or by
    importing an already-processed dataset folder (→ Train directly).

    Mode selection toggles a QStackedWidget so the NAME field is shared and
    validation / accept dispatch lives in one place (`_accept`).
    """

    def __init__(self, store: ProjectStore, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("New project")
        self.setMinimumWidth(560)

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 18)
        v.setSpacing(10)

        title = QLabel("New project")
        title.setObjectName("PageTitle")
        v.addWidget(title)

        self.hint = QLabel()
        self.hint.setProperty("role", "caption")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        v.addSpacing(6)

        # Source selector
        v.addWidget(section_title("SOURCE"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("From video — run processing, then train", _MODE_VIDEO)
        self.mode_combo.addItem("Import processed dataset — train directly", _MODE_IMPORT)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        v.addWidget(self.mode_combo)

        v.addSpacing(4)
        v.addWidget(section_title("NAME"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. kitchen_v1")
        v.addWidget(self.name_edit)

        v.addSpacing(4)

        # Path row — swaps between video and folder pickers.
        self.path_stack = QStackedWidget()
        self.path_stack.addWidget(self._video_page())
        self.path_stack.addWidget(self._import_page())
        v.addWidget(self.path_stack)

        v.addSpacing(8)
        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.create_btn = btns.addButton("Create project", QDialogButtonBox.AcceptRole)
        self.create_btn.setProperty("role", "primary")
        self.create_btn.setCursor(Qt.PointingHandCursor)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self._on_mode_changed(_MODE_VIDEO)

    # ---- pages ----
    def _video_page(self) -> QWidget:
        w = QWidget()
        wv = QVBoxLayout(w)
        wv.setContentsMargins(0, 0, 0, 0)
        wv.setSpacing(6)

        wv.addWidget(section_title("VIDEO"))
        row = QHBoxLayout(); row.setSpacing(8)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText(str(DATA_DIR) + "/yourvideo.mp4")
        self.video_browse_btn = QPushButton("Browse…")
        self.video_browse_btn.setProperty("role", "secondary")
        self.video_browse_btn.setCursor(Qt.PointingHandCursor)
        self.video_browse_btn.clicked.connect(self._browse_video)
        row.addWidget(self.video_path_edit, 1)
        row.addWidget(self.video_browse_btn)
        wv.addLayout(row)

        note = QLabel(
            f"Must live inside {DATA_DIR}.  "
            "That's the folder bind-mounted into the container as /workspace."
        )
        note.setProperty("role", "caption")
        note.setWordWrap(True)
        wv.addWidget(note)
        return w

    def _import_page(self) -> QWidget:
        w = QWidget()
        wv = QVBoxLayout(w)
        wv.setContentsMargins(0, 0, 0, 0)
        wv.setSpacing(6)

        wv.addWidget(section_title("PROCESSED DATASET FOLDER"))
        row = QHBoxLayout(); row.setSpacing(8)
        self.import_path_edit = QLineEdit()
        self.import_path_edit.setPlaceholderText(
            str(DATA_DIR) + "/processed/yourdataset_splat"
        )
        self.import_path_edit.textChanged.connect(self._refresh_import_detection)
        self.import_browse_btn = QPushButton("Browse…")
        self.import_browse_btn.setProperty("role", "secondary")
        self.import_browse_btn.setCursor(Qt.PointingHandCursor)
        self.import_browse_btn.clicked.connect(self._browse_folder)
        row.addWidget(self.import_path_edit, 1)
        row.addWidget(self.import_browse_btn)
        wv.addLayout(row)

        self.import_detect_lbl = QLabel("")
        self.import_detect_lbl.setProperty("role", "caption")
        self.import_detect_lbl.setWordWrap(True)
        wv.addWidget(self.import_detect_lbl)

        note = QLabel(
            f"Folder must live inside {DATA_DIR} and contain either "
            "transforms.json (Nerfstudio) or sparse/0/cameras.bin (COLMAP)."
        )
        note.setProperty("role", "caption")
        note.setWordWrap(True)
        wv.addWidget(note)
        return w

    # ---- mode ----
    def _on_mode_changed(self, idx: int) -> None:
        self.path_stack.setCurrentIndex(idx)
        if idx == _MODE_VIDEO:
            self.hint.setText(
                "Pick a source video. ns-process-data will extract frames and "
                "build a Nerfstudio dataset before training."
            )
        else:
            self.hint.setText(
                "Pick a folder that's already been processed. The project will "
                "skip straight to the Train tab."
            )
            self._refresh_import_detection(self.import_path_edit.text())

    # ---- video mode helpers ----
    def _browse_video(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video",
            str(DATA_DIR),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if path:
            self.video_path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).stem)

    # ---- import mode helpers ----
    def _browse_folder(self) -> None:
        start_dir = str(DATA_DIR / "processed")
        if not Path(start_dir).exists():
            start_dir = str(DATA_DIR)
            Path(start_dir).mkdir(parents=True, exist_ok=True)
        path = QFileDialog.getExistingDirectory(
            self, "Pick a processed dataset folder", start_dir,
            QFileDialog.ShowDirsOnly,
        )
        if path:
            self.import_path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).name)

    def _refresh_import_detection(self, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            self.import_detect_lbl.setText("")
            return
        p = Path(raw).expanduser()
        if not p.is_dir():
            self.import_detect_lbl.setText("⚠ Folder does not exist.")
            return
        fmt = detect_dataset_format(p)
        if fmt == "nerfstudio-data":
            self.import_detect_lbl.setText("✓ Detected Nerfstudio format (transforms.json).")
        elif fmt == "colmap":
            self.import_detect_lbl.setText("✓ Detected COLMAP format (sparse/0/cameras.bin).")
        else:
            self.import_detect_lbl.setText(
                "⚠ No recognized dataset format in this folder."
            )

    # ---- accept ----
    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Give this project a name.")
            return

        mode = self.mode_combo.currentData()
        if mode == _MODE_VIDEO:
            raw = self.video_path_edit.text().strip()
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
        else:
            raw = self.import_path_edit.text().strip()
            if not raw:
                QMessageBox.warning(
                    self, "Missing folder", "Pick a processed dataset folder."
                )
                return
            path = Path(raw).expanduser()
            try:
                self._store.create_from_processed(name, path)
            except ValueError as e:
                QMessageBox.warning(self, "Can't import dataset", str(e))
                return
        self.accept()

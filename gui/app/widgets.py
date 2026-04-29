"""Shared small widgets: cards, section titles, help badges, project bar, new-project dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from app.config import DATA_DIR
from app.store import (
    INPUT_IMAGES_DIR, Project, ProjectStore, count_images,
    detect_dataset_format,
)


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


# ---------- help badge + popup ----------

class HelpPopup(QFrame):
    """Floating explanation popup anchored to a HelpBadge."""

    def __init__(self, title: str, body: str, parent: QWidget | None = None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("HelpPopup")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(340)
        self.setMaximumWidth(420)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        t = QLabel(title)
        t.setObjectName("HelpPopupTitle")
        t.setWordWrap(True)

        b = QLabel(body)
        b.setObjectName("HelpPopupBody")
        b.setWordWrap(True)
        b.setTextFormat(Qt.RichText)

        v.addWidget(t)
        v.addWidget(b)


class HelpBadge(QPushButton):
    """Tiny circular '?' next to a setting. Click shows a HelpPopup below it."""

    def __init__(self, title: str, body: str, parent: QWidget | None = None):
        super().__init__("?", parent)
        self.setProperty("role", "help")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFlat(True)
        self.setFixedSize(16, 16)
        self.setToolTip("What does this do?")
        self._title = title
        self._body = body
        self.clicked.connect(self._show_popup)

    def _show_popup(self) -> None:
        pop = HelpPopup(self._title, self._body, self.window())
        global_pos = self.mapToGlobal(QPoint(0, self.height() + 6))
        pop.adjustSize()
        screen = self.screen().availableGeometry() if self.screen() else None
        x, y = global_pos.x(), global_pos.y()
        if screen is not None:
            w = pop.sizeHint().width()
            h = pop.sizeHint().height()
            if x + w > screen.right() - 8:
                x = screen.right() - w - 8
            if x < screen.left() + 8:
                x = screen.left() + 8
            if y + h > screen.bottom() - 8:
                # Flip above the badge.
                y = self.mapToGlobal(QPoint(0, 0)).y() - h - 6
        pop.move(x, y)
        pop.show()


def labeled(text: str, help_title: str | None = None, help_body: str | None = None) -> QWidget:
    """A form-row label with an optional inline HelpBadge."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    lbl = QLabel(text)
    h.addWidget(lbl)
    if help_title and help_body:
        h.addWidget(HelpBadge(help_title, help_body))
    h.addStretch(1)
    w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    return w


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
_MODE_IMAGES = 1
_MODE_IMPORT = 2


class NewProjectDialog(QDialog):
    """Create a project from a raw video, a folder of images, or by importing
    an already-processed dataset folder.

    Mode selection toggles a QStackedWidget so the NAME field is shared and
    validation / accept dispatch lives in one place (`_accept`). Each mode's
    Process / Train flow:

      VIDEO   → ns-process-data video  → ns-train ...
      IMAGES  → ns-process-data images → ns-train ...
      IMPORT  → (skip Process)         → ns-train ...
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
        self.mode_combo.addItem("From image folder — run processing, then train", _MODE_IMAGES)
        self.mode_combo.addItem("Import processed dataset — train directly", _MODE_IMPORT)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        v.addWidget(self.mode_combo)

        v.addSpacing(4)
        v.addWidget(section_title("NAME"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. kitchen_v1")
        v.addWidget(self.name_edit)

        v.addSpacing(4)

        # Path row — swaps between video, images-folder, and import pickers.
        # Order MUST match the _MODE_* constants so setCurrentIndex(mode) works.
        self.path_stack = QStackedWidget()
        self.path_stack.addWidget(self._video_page())
        self.path_stack.addWidget(self._images_page())
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

    def _images_page(self) -> QWidget:
        w = QWidget()
        wv = QVBoxLayout(w)
        wv.setContentsMargins(0, 0, 0, 0)
        wv.setSpacing(6)

        wv.addWidget(section_title("IMAGE FOLDER"))
        row = QHBoxLayout(); row.setSpacing(8)
        self.images_path_edit = QLineEdit()
        self.images_path_edit.setPlaceholderText(
            str(INPUT_IMAGES_DIR) + "/your_image_set"
        )
        self.images_path_edit.textChanged.connect(self._refresh_images_detection)
        self.images_browse_btn = QPushButton("Browse…")
        self.images_browse_btn.setProperty("role", "secondary")
        self.images_browse_btn.setCursor(Qt.PointingHandCursor)
        self.images_browse_btn.clicked.connect(self._browse_images_folder)
        row.addWidget(self.images_path_edit, 1)
        row.addWidget(self.images_browse_btn)
        wv.addLayout(row)

        # Live readout — populated by _refresh_images_detection on textChanged
        # and after a successful browse. Format mirrors _import_page's detect
        # caption so the dialog feels consistent across modes.
        self.images_detect_lbl = QLabel("")
        self.images_detect_lbl.setProperty("role", "caption")
        self.images_detect_lbl.setWordWrap(True)
        wv.addWidget(self.images_detect_lbl)

        note = QLabel(
            f"Convention: drop image folders under {INPUT_IMAGES_DIR}/.\n"
            f"Anywhere inside {DATA_DIR}/ is technically accepted (the "
            "container needs to see the folder), but keeping image sets "
            "under input_images/ keeps the data/ tree organized."
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
        elif idx == _MODE_IMAGES:
            self.hint.setText(
                "Pick a folder of images. ns-process-data will run COLMAP on "
                "them and build a Nerfstudio dataset before training."
            )
            self._refresh_images_detection(self.images_path_edit.text())
        else:  # _MODE_IMPORT
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

    # ---- images mode helpers ----
    def _browse_images_folder(self) -> None:
        # Auto-create input_images/ on first use so the picker has somewhere
        # sensible to land. Doesn't force the user to keep folders there —
        # _accept only checks DATA_DIR membership, not input_images/.
        INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        start_dir = str(INPUT_IMAGES_DIR)
        path = QFileDialog.getExistingDirectory(
            self, "Pick a folder of images", start_dir,
            QFileDialog.ShowDirsOnly,
        )
        if path:
            self.images_path_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).name)

    def _refresh_images_detection(self, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            self.images_detect_lbl.setText("")
            return
        p = Path(raw).expanduser()
        if not p.is_dir():
            self.images_detect_lbl.setText("⚠ Folder does not exist.")
            return
        n, exts = count_images(p)
        if n == 0:
            self.images_detect_lbl.setText(
                "⚠ No supported images in this folder "
                "(need .jpg / .jpeg / .png / .tif / .tiff / .cr2)."
            )
            return
        ext_summary = ", ".join(sorted(exts))
        # Soft nudge when the folder is under DATA_DIR but not under
        # input_images/. We accept either, but the convention keeps things
        # tidy and matches the rest of the docs.
        try:
            p.resolve().relative_to(INPUT_IMAGES_DIR.resolve())
            location_note = ""
        except ValueError:
            try:
                p.resolve().relative_to(DATA_DIR.resolve())
                location_note = (
                    f"  ·  Tip: keep image sets under {INPUT_IMAGES_DIR}/."
                )
            except ValueError:
                location_note = ""
        self.images_detect_lbl.setText(
            f"✓ {n} image(s) detected ({ext_summary}).{location_note}"
        )

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
        elif mode == _MODE_IMAGES:
            raw = self.images_path_edit.text().strip()
            if not raw:
                QMessageBox.warning(
                    self, "Missing folder", "Pick a folder of images."
                )
                return
            path = Path(raw).expanduser()
            try:
                self._store.create_from_images(name, path)
            except ValueError as e:
                QMessageBox.warning(self, "Can't create project", str(e))
                return
        else:  # _MODE_IMPORT
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

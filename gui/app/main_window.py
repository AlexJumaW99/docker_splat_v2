from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from app.jobs import JobManager
from app.log_drawer import LogDrawer
from app.store import ProjectStore
from app.views.dashboard import DashboardView
from app.views.exports import ExportsView
from app.views.process import ProcessView
from app.views.train import TrainView
from app.views.viewers import ViewersView


NAV_ITEMS = [
    ("Dashboard", "dashboard"),
    ("Process",   "process"),
    ("Train",     "train"),
    ("Exports",   "exports"),
    ("Viewers",   "viewers"),
    ("Runs",      "runs"),
]


class StubView(QWidget):
    def __init__(self, title: str, note: str = ""):
        super().__init__()
        self.setObjectName("Canvas")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(8)
        header = QLabel(title); header.setObjectName("PageTitle")
        body = QLabel(note or "Coming up next.")
        body.setProperty("role", "caption")
        body.setWordWrap(True)
        lay.addWidget(header)
        lay.addWidget(body)
        lay.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self, store: ProjectStore, jobs: JobManager):
        super().__init__()
        self._store = store
        self._jobs = jobs

        self.setWindowTitle("docker_splat")
        self.resize(1240, 820)

        root = QWidget(objectName="Canvas")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        top = QHBoxLayout(); top.setContentsMargins(0, 0, 0, 0); top.setSpacing(0)
        self.stack = QStackedWidget()

        # Build views
        self.stack.addWidget(DashboardView())
        process_view = ProcessView(store, jobs)
        process_view.request_navigate.connect(self._nav_to)
        self.stack.addWidget(process_view)
        train_view = TrainView(store, jobs)
        train_view.request_navigate.connect(self._nav_to)
        self.stack.addWidget(train_view)
        exports_view = ExportsView(store, jobs)
        exports_view.request_navigate.connect(self._nav_to)
        self.stack.addWidget(exports_view)
        viewers_view = ViewersView(store, jobs)
        viewers_view.request_navigate.connect(self._nav_to)
        self.stack.addWidget(viewers_view)
        self.stack.addWidget(StubView("Runs", "Scan data/outputs/, list training runs, "
                                     "open in viser, export PLY, duplicate params, delete."))

        top.addWidget(self._build_sidebar())
        top.addWidget(self.stack, 1)

        top_wrap = QWidget(); top_wrap.setLayout(top)
        top_wrap.setObjectName("Canvas")
        outer.addWidget(top_wrap, 1)

        self.log_drawer = LogDrawer(jobs)
        outer.addWidget(self.log_drawer)

    def _build_sidebar(self) -> QFrame:
        frame = QFrame(objectName="Sidebar")
        frame.setFixedWidth(204)
        v = QVBoxLayout(frame)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        title = QLabel("docker_splat", objectName="AppTitle")
        subtitle = QLabel("NERFSTUDIO GUI", objectName="AppTitleSub")
        v.addWidget(title)
        v.addWidget(subtitle)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for i, (label, _key) in enumerate(NAV_ITEMS):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("role", "nav")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self._nav_to(idx))
            self._nav_group.addButton(btn, i)
            v.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
                btn.setProperty("active", "true")
        v.addStretch(1)
        return frame

    def _nav_to(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        active_btn = self._nav_group.button(idx)
        if active_btn is not None and not active_btn.isChecked():
            active_btn.setChecked(True)
        for b in self._nav_group.buttons():
            b.setProperty("active", "true" if b is active_btn else "false")
            b.style().unpolish(b)
            b.style().polish(b)

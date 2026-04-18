import sys

from PySide6.QtWidgets import QApplication

from app.jobs import JobManager
from app.main_window import MainWindow
from app.store import ProjectStore
from app.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("docker_splat")
    app.setOrganizationName("docker_splat_gui")
    apply_theme(app)

    store = ProjectStore()
    jobs = JobManager()

    w = MainWindow(store, jobs)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

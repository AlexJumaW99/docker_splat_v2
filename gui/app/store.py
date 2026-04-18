"""
Project store.

A project bundles one video + its processed directory + its training runs.
State persists to ~/.config/docker_splat_gui/projects.json.

The store is a QObject singleton; views connect to its signals so they stay
in sync when the active project changes.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from pydantic import BaseModel, Field

from app.config import DATA_DIR, PROJECTS_FILE


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_-]+", "_", text)
    return text or "project"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Project(BaseModel):
    id: str
    name: str
    video_path: str  # absolute host path
    processed_dir: str  # absolute host path under data/processed/
    created_at: str = Field(default_factory=_iso_now)
    # Per-project parameter presets (form values keyed by preset name)
    presets_process: dict[str, dict] = Field(default_factory=dict)
    presets_train: dict[str, dict] = Field(default_factory=dict)
    last_process_params: dict | None = None
    last_train_params: dict | None = None


class _Root(BaseModel):
    projects: list[Project] = Field(default_factory=list)
    active_project_id: str | None = None


class ProjectStore(QObject):
    projects_changed = Signal()            # list changed
    active_changed = Signal(object)        # Project | None

    def __init__(self, path: Path = PROJECTS_FILE, parent: QObject | None = None):
        super().__init__(parent)
        self._path = path
        self._root = self._load()

    # ---- persistence ----
    def _load(self) -> _Root:
        if not self._path.exists():
            return _Root()
        try:
            return _Root.model_validate_json(self._path.read_text())
        except Exception:
            backup = self._path.with_suffix(".corrupt.json")
            try:
                self._path.rename(backup)
            except OSError:
                pass
            return _Root()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(self._root.model_dump_json(indent=2))
        tmp.replace(self._path)

    # ---- read ----
    @property
    def projects(self) -> list[Project]:
        return list(self._root.projects)

    @property
    def active(self) -> Project | None:
        if self._root.active_project_id is None:
            return None
        for p in self._root.projects:
            if p.id == self._root.active_project_id:
                return p
        return None

    def get(self, project_id: str) -> Project | None:
        for p in self._root.projects:
            if p.id == project_id:
                return p
        return None

    # ---- mutate ----
    def set_active(self, project_id: str | None) -> None:
        self._root.active_project_id = project_id
        self._save()
        self.active_changed.emit(self.active)

    def create(self, name: str, video_path: Path) -> Project:
        """Creates a project. Raises ValueError if name conflicts or the video
        is outside the data/ bind mount."""
        name = name.strip()
        if not name:
            raise ValueError("Project name can't be empty.")

        try:
            video_path.resolve().relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"Video must live inside {DATA_DIR}.\n\n"
                "The container only sees that folder (it's bind-mounted as /workspace)."
            ) from None

        base = _slugify(name)
        pid = base
        i = 2
        taken = {p.id for p in self._root.projects}
        while pid in taken:
            pid = f"{base}_{i}"
            i += 1

        if any(p.name == name for p in self._root.projects):
            raise ValueError(f"A project named '{name}' already exists.")

        processed_dir = DATA_DIR / "processed" / f"{pid}_splat"

        project = Project(
            id=pid,
            name=name,
            video_path=str(video_path.resolve()),
            processed_dir=str(processed_dir),
        )
        self._root.projects.append(project)
        self._root.active_project_id = project.id
        self._save()
        self.projects_changed.emit()
        self.active_changed.emit(project)
        return project

    def update_video_path(self, project_id: str, video_path: Path) -> None:
        try:
            video_path.resolve().relative_to(DATA_DIR.resolve())
        except ValueError:
            raise ValueError(f"Video must live inside {DATA_DIR}.") from None
        p = self.get(project_id)
        if p is None:
            return
        p.video_path = str(video_path.resolve())
        self._save()
        self.active_changed.emit(self.active)

    def update_last_process_params(self, project_id: str, params: dict) -> None:
        p = self.get(project_id)
        if p is None:
            return
        p.last_process_params = params
        self._save()

    def update_last_train_params(self, project_id: str, params: dict) -> None:
        p = self.get(project_id)
        if p is None:
            return
        p.last_train_params = params
        self._save()

    def save_preset(self, project_id: str, kind: str, name: str, params: dict) -> None:
        """kind: 'process' | 'train'"""
        p = self.get(project_id)
        if p is None:
            return
        store = p.presets_process if kind == "process" else p.presets_train
        store[name] = params
        self._save()

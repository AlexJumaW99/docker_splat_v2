"""
In-memory index of all warehouse manifests.

Built once at startup, refreshed after writes. Replaces per-view filesystem
walks with dict lookups. At current scale (~50 manifests) refresh() takes
<10 ms; at 500 manifests it would still be <100 ms.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from app.config import WAREHOUSE_DIR
from app.manifest import (
    ExportManifest,
    Manifest,
    ProcessManifest,
    RenderManifest,
    SourceManifest,
    TrainManifest,
    load_manifest,
)

_log = logging.getLogger(__name__)


class ManifestIndex:
    """Load-once / query-many index over data_v2/*/manifest.json."""

    def __init__(self, warehouse_dir: Path = WAREHOUSE_DIR):
        self._warehouse_dir = warehouse_dir
        self._by_id: dict[str, Manifest] = {}
        self._children: dict[str, list[Manifest]] = {}
        self._source_checksums: dict[str, SourceManifest] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.refresh()

    def refresh(self) -> None:
        """Full rescan of the warehouse. O(total manifests)."""
        t0 = time.monotonic()
        self._by_id.clear()
        self._children.clear()
        self._source_checksums.clear()

        for stage_dir_name in ("sources", "processed", "trained"):
            root = self._warehouse_dir / stage_dir_name
            if not root.is_dir():
                continue
            for sub in root.iterdir():
                if not sub.is_dir():
                    continue
                self._ingest(sub / "manifest.json")

        exports_root = self._warehouse_dir / "exports"
        if exports_root.is_dir():
            for kind_dir in exports_root.iterdir():
                if not kind_dir.is_dir():
                    continue
                for sub in kind_dir.iterdir():
                    if not sub.is_dir():
                        continue
                    self._ingest(sub / "manifest.json")

        renders_root = self._warehouse_dir / "renders"
        if renders_root.is_dir():
            for sub in renders_root.iterdir():
                if not sub.is_dir():
                    continue
                self._ingest(sub / "manifest.json")

        self._loaded = True
        elapsed = (time.monotonic() - t0) * 1000
        _log.debug(
            "ManifestIndex.refresh: %d manifests in %.1f ms",
            len(self._by_id), elapsed,
        )

    def _ingest(self, manifest_path: Path) -> None:
        if not manifest_path.is_file():
            return
        try:
            m = load_manifest(manifest_path)
        except Exception:
            return
        self._by_id[m.run_id] = m
        if hasattr(m, "parent") and m.parent is not None:
            self._children.setdefault(m.parent.run_id, []).append(m)
        if isinstance(m, SourceManifest):
            for cksum in m.checksums.values():
                self._source_checksums[cksum] = m

    # ---- queries ----

    def get(self, run_id: str) -> Manifest | None:
        self._ensure_loaded()
        return self._by_id.get(run_id)

    def all_manifests(self) -> list[Manifest]:
        self._ensure_loaded()
        return list(self._by_id.values())

    def children_of(self, parent_run_id: str) -> list[Manifest]:
        self._ensure_loaded()
        return list(self._children.get(parent_run_id, []))

    def trains_for_process(self, process_run_id: str) -> list[TrainManifest]:
        return [
            m for m in self.children_of(process_run_id)
            if isinstance(m, TrainManifest)
        ]

    def exports_for_train(self, train_run_id: str) -> list[ExportManifest]:
        return [
            m for m in self.children_of(train_run_id)
            if isinstance(m, ExportManifest)
        ]

    def renders_for_train(self, train_run_id: str) -> list[RenderManifest]:
        return [
            m for m in self.children_of(train_run_id)
            if isinstance(m, RenderManifest)
        ]

    def source_by_checksum(self, checksum: str) -> SourceManifest | None:
        self._ensure_loaded()
        return self._source_checksums.get(checksum)

    # ---- diagnostics (Tier 3) ----

    def orphans(self) -> list[Manifest]:
        """Manifests whose parent.run_id doesn't resolve to any known manifest."""
        self._ensure_loaded()
        result: list[Manifest] = []
        for m in self._by_id.values():
            if hasattr(m, "parent") and m.parent is not None:
                if m.parent.run_id not in self._by_id:
                    result.append(m)
        return result


# Module-level singleton, lazily initialized.
_INDEX: ManifestIndex | None = None


def get_index() -> ManifestIndex:
    """Return the module-level ManifestIndex singleton, creating it on first call."""
    global _INDEX
    if _INDEX is None:
        _INDEX = ManifestIndex()
    return _INDEX

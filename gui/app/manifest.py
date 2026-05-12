"""
Manifest model.

Every pipeline stage (source, process, train, export, render) writes a
`manifest.json` beside its output. The manifest is the declarative record
of WHAT this artifact is, HOW it was produced, and WHICH upstream artifact
it descends from. Lineage is captured by `parent.run_id`, not inferred from
filenames or folder layout.

This module is the single source of truth for the manifest contract. The
companion `manifest_schema.json` is generated from it (see
`manifest_schema.json` in this directory and the `export_schema()` helper
at the bottom of this file) and is the published artifact for external
(non-Python) consumers of the warehouse.

Path convention: every path stored in a manifest is POSIX-style and
relative to the warehouse root (e.g. `data_v2/`). The consumer joins it
with their own absolute root. No `/workspace/...` or `~/docker-stacks/...`
absolute paths in manifests, ever.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


SCHEMA_VERSION: Literal["1.0"] = "1.0"

ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

Stage = Literal["source", "process", "train", "export", "render"]


# --------------------------------------------------------------------- shared


class Parent(BaseModel):
    """Declared lineage pointer. The `run_id` references another manifest's
    top-level `run_id`. Stage is denormalized for cheap joins without a
    second lookup."""

    model_config = ConfigDict(extra="forbid")

    stage: Stage
    run_id: str = Field(pattern=ULID_PATTERN)


class Host(BaseModel):
    """Reproducibility surface — captured at write time. Every field is
    optional because backfilled manifests may not have all of it, but
    declared (live-written) manifests should populate as much as possible.
    `extra='allow'` so callers can record extra environment without a
    schema bump."""

    model_config = ConfigDict(extra="allow")

    hostname: str | None = None
    os: str | None = None
    gpu: str | None = None
    cuda_version: str | None = None
    docker_image: str | None = None  # e.g. "splat-nerfstudio-v2:sha-69eaa50"
    colmap_commit: str | None = None
    nerfstudio_commit: str | None = None
    gsplat_version: str | None = None


# -------------------------------------------------------------- stage params


class SourceParams(BaseModel):
    """The original artifact a project started from. There is no upstream."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["video", "image_folder", "imported_dataset"]
    original_filename: str
    size_bytes: int = Field(ge=0)

    # video-only
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    frame_count: int | None = None

    # image_folder-only
    image_count: int | None = None
    extensions: list[str] | None = None
    heic_originals_count: int | None = None  # iPhone HEIC stash

    # imported_dataset-only
    dataset_format: Literal["colmap", "nerfstudio-data", "other"] | None = None


class ProcessParams(BaseModel):
    """The arguments handed to `ns-process-data {video|images}`. Mirrors
    the flags the Process tab exposes."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["video", "images"]
    matching_method: Literal["sequential", "vocab_tree", "exhaustive"]
    num_frames_target: int | None = None  # video only; None for images
    num_downscales: int = 3
    camera_type: Literal["perspective", "fisheye", "equirectangular"] = "perspective"
    refine_intrinsics: bool = True
    same_dimensions: bool = True
    use_gpu: bool = True
    use_sfm_depth: bool = False
    crop_factor: tuple[float, float, float, float] | None = None  # (top, bottom, left, right)
    percent_radius_crop: float | None = Field(default=None, ge=0.0, le=1.0)
    images_per_equirect: Literal[8, 14] | None = None
    random_seed: int | None = None
    eval_random_seed: int | None = None
    eval_data: str | None = None  # relpath to optional eval set
    extra_flags: list[str] = Field(default_factory=list)


TrainMethod = Literal[
    "splatfacto",
    "splatfacto-big",
    "splatfacto-mcmc",
    "splatfacto-w",
    "splatfacto-w-light",
]


class TrainParams(BaseModel):
    """Arguments handed to `ns-train <method>`. Captures both the live
    knobs the GUI exposes and the dataparser routing."""

    model_config = ConfigDict(extra="forbid")

    method: TrainMethod
    experiment_name: str
    # Legacy: ns-train picks its own `%Y-%m-%d_%H%M%S` slug. Kept here so
    # we can locate the original on-disk run dir from a manifest. After
    # the layout migration this becomes purely informational.
    timestamp_dir: str
    max_num_iterations: int = Field(ge=1)
    steps_per_save: int = Field(ge=1)
    vis: Literal["viewer", "tensorboard", "viewer+tensorboard"]
    viewer_websocket_host: str = "0.0.0.0"
    viewer_quit_on_train_completion: bool = False
    dataparser: Literal["nerfstudio-data", "colmap"]
    extra_flags: list[str] = Field(default_factory=list)


class ExportParams(BaseModel):
    """Arguments handed to `ns-export gaussian-splat` or the custom
    `ns-export-splatfactow` wrapper. For splatfacto-w runs, also captures
    which appearance embedding was baked into the per-gaussian SH."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["splat", "pcd", "mesh"]
    exporter: str  # "ns-export gaussian-splat" | "ns-export-splatfactow" | ...
    source_checkpoint_step: int = Field(ge=0)

    # splatfacto-w / -light only
    appearance_embedding_strategy: (
        Literal["average_training_cameras", "specific_camera", "manual"] | None
    ) = None
    appearance_embedding_camera_id: int | None = None


class RenderParams(BaseModel):
    """Arguments handed to `ns-render camera-path`. The camera path JSON
    itself lives under the processed run's `camera_paths/` dir; we record
    the relpath so the render can be reproduced."""

    model_config = ConfigDict(extra="forbid")

    camera_path_filename: str  # relpath, within parent processed run
    fps: float | None = None
    resolution_width: int | None = None
    resolution_height: int | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None


# ---------------------------------------------- per-stage manifest wrappers


class _CommonManifestFields(BaseModel):
    """Fields shared by every stage. Not used directly — subclassed to
    bolt on stage-specific `stage`, `parent`, and `params`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str = Field(pattern=ULID_PATTERN)
    display_name: str | None = None
    created_at: datetime  # ISO 8601, UTC

    host: Host | None = None
    command: str | None = None  # actual shell line that produced the artifact

    # Logical name → relpath under the warehouse root. Examples:
    #   process:  {"transforms_json": "processed/<id>/transforms.json", ...}
    #   train:    {"config_yml": "...", "latest_checkpoint": "..."}
    #   export:   {"ply": "exports/<id>/splat.ply"}
    outputs: dict[str, str] = Field(default_factory=dict)

    # Free-form numeric metrics. Examples:
    #   process: {"colmap_pose_recovery_pct": 98.12, "extracted_frame_count": 247}
    #   train:   {"latest_step": 7000, "final_psnr": 28.4, "final_num_gaussians": 84e6,
    #             "train_duration_seconds": 901.3}
    #   render:  {"duration_seconds": 12.5}
    metrics: dict[str, float] = Field(default_factory=dict)

    # relpath → "sha256:<64 hex>". Cover the load-bearing outputs (transforms.json,
    # final .ckpt, .ply, .mp4). Cheap to compute, makes the warehouse verifiable.
    checksums: dict[str, str] = Field(default_factory=dict)

    notes: str | None = None
    provenance: Literal["declared", "backfilled"] = "declared"


class SourceManifest(_CommonManifestFields):
    stage: Literal["source"] = "source"
    parent: None = None  # roots have no parent
    params: SourceParams


class ProcessManifest(_CommonManifestFields):
    stage: Literal["process"] = "process"
    parent: Parent
    params: ProcessParams


class TrainManifest(_CommonManifestFields):
    stage: Literal["train"] = "train"
    parent: Parent
    params: TrainParams


class ExportManifest(_CommonManifestFields):
    stage: Literal["export"] = "export"
    parent: Parent
    params: ExportParams


class RenderManifest(_CommonManifestFields):
    stage: Literal["render"] = "render"
    parent: Parent
    params: RenderParams


# Discriminated union — the top-level type for `manifest.json`.
Manifest = Annotated[
    Union[
        SourceManifest,
        ProcessManifest,
        TrainManifest,
        ExportManifest,
        RenderManifest,
    ],
    Field(discriminator="stage"),
]


_MANIFEST_ADAPTER: TypeAdapter[Manifest] = TypeAdapter(Manifest)


# ----------------------------------------------------------------- helpers


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate a `manifest.json`. Raises pydantic.ValidationError
    on schema mismatch."""

    raw = Path(path).read_bytes()
    return _MANIFEST_ADAPTER.validate_json(raw)


def dump_manifest(manifest: Manifest, path: str | Path) -> None:
    """Atomically write a manifest. Writes to `<path>.tmp` then renames so
    a partial write never replaces a good manifest."""

    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        _MANIFEST_ADAPTER.dump_json(manifest, indent=2, exclude_none=True).decode()
    )
    tmp.replace(target)


def export_schema() -> dict:
    """Return the JSON Schema (Draft 2020-12) for the discriminated
    Manifest union, with `$schema`, `$id`, and `title` headers attached so
    the file is self-describing for external (non-Python) consumers.

    Regenerate the on-disk artifact with:
        python -m app.manifest > app/manifest_schema.json
    """

    schema = _MANIFEST_ADAPTER.json_schema(
        by_alias=True,
        ref_template="#/$defs/{model}",
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = (
        "https://github.com/AlexJumaW99/docker_splat_v2/"
        f"manifest_schema/v{SCHEMA_VERSION}"
    )
    schema["title"] = "docker_splat_v2 Manifest"
    schema["description"] = (
        f"Per-stage manifest for the docker_splat_v2 warehouse. "
        f"Schema version {SCHEMA_VERSION}. Every artifact directory "
        f"(source, processed, train, export, render) contains one "
        f"manifest.json validated against this schema."
    )
    return schema


if __name__ == "__main__":  # python -m app.manifest > manifest_schema.json
    print(json.dumps(export_schema(), indent=2))

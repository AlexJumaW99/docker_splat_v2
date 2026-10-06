"""
Qt-free logic behind the Train tab's resolution / image-cache / eval-cadence
controls and the pose-recovery readout. Everything here is a pure function of
the processed dataset on disk plus the form's choices, so it can be unit
tested without a QApplication (see gui/tests/test_train_options.py).

Measured background (2026-10-05, RTX 5070 Ti; details in IMPROVEMENT_PLAN.md):

* nerfstudio-data trains 4K sources at 960 px because its auto rule picks the
  smallest existing images_<N> copy whose longest side is <= 1600 px. A splat
  is only sharp at the resolution it was trained at (960 px model viewed at
  1920 px: PSNR 28.2; 1920 px model: 32.5), so the training resolution is a
  first-class choice here.
* The default GPU image cache holds every frame as uint8 on the GPU: 266 4K
  frames = 6.2 GiB, which is what pushed splatfacto-big over 16 GB. Above
  CACHE_CPU_THRESHOLD_GIB we switch to the CPU cache (+34 % wall time at 4K).
* The VRAM high-water mark of every run used to be the FIRST eval image:
  nerfstudio's train.py enables cuDNN autotuning, and the first LPIPS call
  then tries every convolution algorithm at full resolution (+3.15 GiB at
  960 px, +12.4 GiB at 1920 px, measured in isolation). Image v4.8 turns that
  autotuning off (peak 11.15 -> 2.42 GiB at 1920 px, no speed change). The
  eval cadence default here (1000 instead of nerfstudio's 100) is about the
  other cost of frequent evals: ~180 MB of TensorBoard images per 1000 steps
  at 4K and one extra full-resolution render every 100 steps.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Same constant as nerfstudio_dataparser.MAX_AUTO_RESOLUTION / colmap_dataparser.
MAX_AUTO_RESOLUTION = 1600

# Auto image-cache placement flips to CPU above this many GiB of uint8 frames.
# 3 GiB leaves ~10 GiB of a 16 GB card for the model, rasterizer and desktop.
CACHE_CPU_THRESHOLD_GIB = 3.0

# Defaults the GUI emits (nerfstudio's own are 100 and 1000).
DEFAULT_STEPS_PER_EVAL_IMAGE = 1000
DEFAULT_STEPS_PER_EVAL_ALL_IMAGES = 1000

# Below this share of posed frames the Train tab asks for confirmation.
POSE_RECOVERY_BLOCK_RATIO = 0.5
# Below this it only warns.
POSE_RECOVERY_WARN_RATIO = 0.9

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_LEVEL_RE = re.compile(r"^images_(\d+)$")
_FRACTION_LABEL = {1: "Full", 2: "½", 4: "¼", 8: "⅛", 16: "1⁄16", 32: "1⁄32"}


# ------------------------------------------------------------------ dataset facts


@dataclass(frozen=True)
class ImageLevel:
    """One `images/` or `images_<N>/` folder of a processed dataset."""

    factor: int
    width: int
    height: int
    count: int
    path: Path


def _first_image(folder: Path) -> Path | None:
    try:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                return p
    except OSError:
        return None
    return None


def _image_size(path: Path) -> tuple[int, int] | None:
    """Read (width, height) from the file header only."""
    try:
        from PIL import Image  # pillow is a dependency of pillow-heif

        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def _count_images(folder: Path) -> int:
    try:
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    except OSError:
        return 0


def dataset_image_levels(processed_dir: Path | str) -> dict[int, ImageLevel]:
    """Map downscale factor -> ImageLevel for `images/` (factor 1) and every
    `images_<N>/` sibling that exists and holds at least one image. Sizes come
    from the first image's header, so this is cheap even for 4K datasets."""
    root = Path(processed_dir)
    levels: dict[int, ImageLevel] = {}
    candidates: list[tuple[int, Path]] = []
    if (root / "images").is_dir():
        candidates.append((1, root / "images"))
    try:
        for p in root.iterdir():
            m = _LEVEL_RE.match(p.name)
            if m and p.is_dir():
                candidates.append((int(m.group(1)), p))
    except OSError:
        return levels
    for factor, folder in candidates:
        first = _first_image(folder)
        if first is None:
            continue
        size = _image_size(first)
        if size is None:
            continue
        levels[factor] = ImageLevel(
            factor=factor, width=size[0], height=size[1], count=_count_images(folder), path=folder
        )
    return levels


def nerfstudio_auto_factor(levels: dict[int, ImageLevel]) -> int:
    """Replicates nerfstudio_dataparser._get_fname's auto rule: keep doubling
    while the longest side is still above MAX_AUTO_RESOLUTION *and* the next
    images_<N> folder exists."""
    if 1 not in levels:
        return 1
    full = levels[1]
    df = 1
    while max(full.width, full.height) / df > MAX_AUTO_RESOLUTION and (df * 2) in levels:
        df *= 2
    return df


# --------------------------------------------------------------- resolution choice


@dataclass(frozen=True)
class ResolutionOption:
    """One entry of the Training resolution dropdown.

    key          stable identifier persisted in presets / last-used params
    factor       1, 2, 4, 8 ... relative to the source frames
    via_folder   True  -> emitted as `<dataparser> --downscale-factor N`
                 False -> emitted as `--pipeline.datamanager.camera-res-scale-factor 1/N`
                          (dataset has no images_<N>; rescaled when loaded)
    """

    key: str
    label: str
    factor: int
    width: int
    height: int
    via_folder: bool

    @property
    def is_auto(self) -> bool:
        return self.key == "auto"

    def dataparser_flags(self) -> list[str]:
        """Flags that must follow the dataparser subcommand."""
        if self.is_auto or self.factor == 1 or not self.via_folder:
            return []
        return ["--downscale-factor", str(self.factor)]

    def pipeline_flags(self) -> list[str]:
        """Flags that go before the dataparser subcommand."""
        if self.is_auto or self.factor == 1 or self.via_folder:
            return []
        return ["--pipeline.datamanager.camera-res-scale-factor", repr(1.0 / self.factor)]


def resolution_options(processed_dir: Path | str) -> list[ResolutionOption]:
    """Build the dropdown entries for a dataset. Order: Auto, Full, then the
    halving steps down to 1/8 (or further, if folders exist). Entries backed
    by an existing images_<N> folder are labelled with the folder; the rest
    are implemented by rescaling at load time."""
    levels = dataset_image_levels(processed_dir)
    if 1 not in levels:
        return []
    full = levels[1]
    auto_df = nerfstudio_auto_factor(levels)
    auto_lv = levels[auto_df]
    auto_note = (
        " — nerfstudio's ≤1600 px rule"
        if max(auto_lv.width, auto_lv.height) <= MAX_AUTO_RESOLUTION
        else " — still >1600 px: no smaller images_N copy exists"
    )
    opts = [
        ResolutionOption(
            key="auto",
            label=f"Auto · {auto_lv.width} × {auto_lv.height}"
            + ("" if auto_df == 1 else f" (images_{auto_df})")
            + auto_note,
            factor=auto_df,
            width=auto_lv.width,
            height=auto_lv.height,
            via_folder=True,
        ),
        ResolutionOption(
            key="full",
            label=f"Full · {full.width} × {full.height}",
            factor=1,
            width=full.width,
            height=full.height,
            via_folder=True,
        ),
    ]
    factors = sorted({2, 4, 8} | {f for f in levels if f > 1})
    for f in factors:
        if f in levels:
            lv = levels[f]
            opts.append(
                ResolutionOption(
                    key=f"down:{f}",
                    label=f"{_FRACTION_LABEL.get(f, f'1/{f}')} · {lv.width} × {lv.height} (images_{f})",
                    factor=f,
                    width=lv.width,
                    height=lv.height,
                    via_folder=True,
                )
            )
        else:
            w, h = int(full.width / f), int(full.height / f)
            opts.append(
                ResolutionOption(
                    key=f"scale:{f}",
                    label=f"{_FRACTION_LABEL.get(f, f'1/{f}')} · {w} × {h} (rescaled at load — no images_{f} folder)",
                    factor=f,
                    width=w,
                    height=h,
                    via_folder=False,
                )
            )
    return opts


def find_option(options: list[ResolutionOption], key: str | None) -> ResolutionOption | None:
    for o in options:
        if o.key == key:
            return o
    return None


def missing_levels(processed_dir: Path | str, up_to: int = 8) -> list[int]:
    """Downscale factors (2, 4, 8) that the dataset lacks as folders."""
    levels = dataset_image_levels(processed_dir)
    f, out = 2, []
    while f <= up_to:
        if f not in levels:
            out.append(f)
        f *= 2
    return out


# ----------------------------------------------------------------- image cache


def estimate_cache_gib(frame_count: int, width: int, height: int) -> float:
    """Size of the uint8 image cache nerfstudio keeps for training frames."""
    return frame_count * width * height * 3 / 1024**3


def choose_cache_images(mode: str, frame_count: int, width: int, height: int) -> str | None:
    """Resolve the Image cache control to the flag value to emit.

    mode  "auto" | "gpu" | "cpu"
    Returns "gpu", "cpu", or None meaning "do not emit the flag; keep
    nerfstudio's default".
    """
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        return "gpu"
    return "cpu" if estimate_cache_gib(frame_count, width, height) > CACHE_CPU_THRESHOLD_GIB else None


# ---------------------------------------------------------------- pose recovery


@dataclass(frozen=True)
class PoseRecovery:
    registered: int | None
    extracted: int | None
    source: str  # "manifest" | "files" | "none"

    @property
    def ratio(self) -> float | None:
        if not self.registered or not self.extracted:
            return None
        return min(1.0, self.registered / self.extracted)

    @property
    def level(self) -> str:
        """'success' | 'warning' | 'error' | 'neutral' for pill colouring."""
        r = self.ratio
        if r is None:
            return "neutral"
        if r < POSE_RECOVERY_BLOCK_RATIO:
            return "error"
        if r < POSE_RECOVERY_WARN_RATIO:
            return "warning"
        return "success"


def count_transforms_frames(processed_dir: Path | str) -> int | None:
    t = Path(processed_dir) / "transforms.json"
    if not t.is_file():
        return None
    try:
        return len(json.loads(t.read_text()).get("frames", []))
    except (OSError, ValueError):
        return None


def pose_recovery(processed_dir: Path | str) -> PoseRecovery:
    """How many extracted frames received a camera pose.

    Reads the directory first — `transforms.json` frames (one per registered
    image) and files in `images/` — because that is what training will load;
    the manifest's metrics are used only for whatever is not on disk. A
    vizer-written manifest once recorded 209 extracted frames while 267 sat
    in `images/` (written before the run had finished), which is exactly the
    case this ordering avoids.
    """
    root = Path(processed_dir)
    registered = count_transforms_frames(root)
    extracted = _count_images(root / "images") if (root / "images").is_dir() else None
    source = "files" if registered is not None else "none"
    if registered is None or extracted is None:
        m = root / "manifest.json"
        if m.is_file():
            try:
                metrics = json.loads(m.read_text()).get("metrics") or {}
                if registered is None and metrics.get("registered_frame_count") is not None:
                    registered = int(metrics["registered_frame_count"])
                    source = "manifest"
                if extracted is None and metrics.get("extracted_frame_count") is not None:
                    extracted = int(metrics["extracted_frame_count"])
            except (OSError, ValueError, TypeError):
                pass
    return PoseRecovery(registered=registered, extracted=extracted, source=source)


# ------------------------------------------------------------- argv assembly


def build_train_argv(
    *,
    method: str,
    data: str,
    output_dir: str,
    experiment_name: str,
    max_num_iterations: int,
    steps_per_save: int,
    vis: str,
    quit_on_train_completion: bool,
    steps_per_eval_image: int | None,
    steps_per_eval_all_images: int | None,
    cache_images: str | None,
    rasterize_mode: str | None,
    resolution: ResolutionOption | None,
    extra_flags: list[str],
    project_dataparser: str,
    colmap_path: str | None,
) -> list[str]:
    """Assemble the `ns-train` argv exactly as the Train tab runs it.

    Order matters for tyro: method/pipeline flags first, the user's raw flags
    next, and the dataparser subcommand (with its own flags) last. The default
    dataparser (nerfstudio-data) is only spelled out when it needs a flag.
    `colmap_path` is the project's relative COLMAP model path; splatfacto-w
    requires it (the caller enforces that before getting here).
    """
    argv = [
        "ns-train", method,
        "--experiment-name", experiment_name,
        "--data", data,
        "--output-dir", output_dir,
        "--max-num-iterations", str(max_num_iterations),
        "--steps-per-save", str(steps_per_save),
        "--vis", vis,
    ]
    if "viewer" in vis:
        argv += [
            "--viewer.websocket-host", "0.0.0.0",
            "--viewer.quit-on-train-completion", "True" if quit_on_train_completion else "False",
        ]
    if steps_per_eval_image is not None:
        argv += ["--steps-per-eval-image", str(steps_per_eval_image)]
    if steps_per_eval_all_images is not None:
        argv += ["--steps-per-eval-all-images", str(steps_per_eval_all_images)]
    if cache_images is not None:
        argv += ["--pipeline.datamanager.cache-images", cache_images]
    if rasterize_mode is not None:
        argv += ["--pipeline.model.rasterize-mode", rasterize_mode]
    if resolution is not None:
        argv += resolution.pipeline_flags()
    argv += list(extra_flags)

    dp_flags = resolution.dataparser_flags() if resolution is not None else []
    if method == "splatfacto-w":
        argv += ["splatfactow_dataparser", "--colmap-path", colmap_path or "colmap/sparse/0", *dp_flags]
    elif project_dataparser == "colmap":
        argv += ["colmap", "--colmap-path", colmap_path or "colmap/sparse/0", *dp_flags]
    elif project_dataparser == "nerfstudio-data":
        if dp_flags:
            argv += ["nerfstudio-data", *dp_flags]
    else:
        argv += [project_dataparser, *dp_flags]
    return argv

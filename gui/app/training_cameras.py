"""
Training cameras for SuperSplat — `<stem>.cameras.json` beside an exported PLY.

nerfstudio's viewer (viser, on :7007) draws every training image as a camera
frustum with the photo on its far face, at the pose the splat was trained
from. This script writes those same cameras to a file SuperSplat can load, so
the editor can draw them on the exported splat the same way:

* the poses are the dataparser's own (`get_dataparser_outputs`), in the
  dataparser frame — the frame `ns-export gaussian-splat` writes the PLY in,
  so a camera lands on the part of the splat its photo shows;
* the intrinsics and images are the ones splatfacto trained against: the
  full-image datamanager undistorts every image and replaces fx/fy/cx/cy and
  the size with the undistorted values (full_images_datamanager.py), and this
  script calls the same `_undistort_image` on the same images;
* held-out (eval) cameras are included and labelled, so the editor can show or
  hide them.

It runs INSIDE the splat-ns-v2 container (nerfstudio, numpy, PIL, cv2). The
GUI never imports nerfstudio: it passes this file's text to `python3 -c`
(`script_source()`), so only the standard library is imported at module
level and everything heavy is imported inside `main()`.

    python3 -c "$(cat training_cameras.py)" --config <config.yml> \
        --ply <export.ply> --out <export>.cameras.json [--lineage '<json>']

The last stdout line is `SPLAT_TRAINING_CAMERAS_OK {...}` or
`SPLAT_TRAINING_CAMERAS_FAILED <reason>` (exit 1); nerfstudio's own console
output may precede it.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

FORMAT = "splat-training-cameras"
VERSION = 1
OK_SENTINEL = "SPLAT_TRAINING_CAMERAS_OK"
FAIL_SENTINEL = "SPLAT_TRAINING_CAMERAS_FAILED"

# Thumbnail long side (px) and JPEG quality. 384 px q82 measured at ~26 KB per
# 16:9 frame: ~9 MB for a 266-frame dataset. Above LARGE_SET cameras the long
# side drops so the file stays reasonable.
THUMB_LONG_SIDE = 384
THUMB_LONG_SIDE_LARGE = 256
LARGE_SET = 1000
JPEG_QUALITY = 82

# The dataparser re-derives orientation and scale from the dataset on disk;
# if that no longer matches what the run recorded at training time, the
# cameras would not line up with the splat, so refuse instead.
TRANSFORM_TOLERANCE = 1e-4

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}

# nerfstudio CameraType values (cameras.py) -> names written to the file
_CAMERA_TYPES = {1: "perspective", 2: "fisheye", 3: "equirectangular", 4: "omnidirectional",
                 5: "vr180_l", 6: "vr180_r", 7: "orthophoto", 8: "fisheye624"}


# ---------------------------------------------------------------- pure helpers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="splat-training-cameras", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="the run's config.yml (container path)")
    p.add_argument("--ply", required=True, help="the exported splat .ply (container path)")
    p.add_argument("--out", required=True, help="where to write <stem>.cameras.json")
    p.add_argument("--lineage", default="{}", help="JSON object of run ids etc. copied into the file")
    p.add_argument("--thumb", type=int, default=0, help="thumbnail long side in px (0 = automatic)")
    p.add_argument("--quality", type=int, default=JPEG_QUALITY, help="thumbnail JPEG quality")
    p.add_argument("--workers", type=int, default=4, help="threads for undistort + encode")
    args = p.parse_args(argv)
    json.loads(args.lineage)  # fail early on a malformed lineage argument
    return args


def thumb_long_side(num_cameras: int, requested: int = 0) -> int:
    if requested > 0:
        return requested
    return THUMB_LONG_SIDE_LARGE if num_cameras > LARGE_SET else THUMB_LONG_SIDE


def thumb_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Size of a thumbnail whose long side is `long_side`; never upscales."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image size {width}x{height}")
    scale = min(1.0, long_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def ply_header_info(path: str | Path) -> dict[str, Any]:
    """Vertex count, property list (name, PLY type) and the byte offset of the
    data of a binary little-endian PLY."""
    with open(path, "rb") as f:
        first = f.readline().strip()
        if first != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        count = None
        props: list[tuple[str, str]] = []
        fmt = None
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: header has no end_header")
            line = raw.decode("ascii", "replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices are not supported")
                props.append((parts[2], parts[1]))
        offset = f.tell()
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: format {fmt!r} is not binary_little_endian")
    if count is None:
        raise ValueError(f"{path}: no vertex element")
    for name, kind in props:
        if kind not in _PLY_TYPES:
            raise ValueError(f"{path}: property {name} has unknown type {kind}")
    return {"count": count, "properties": props, "offset": offset}


def transforms_match(a: list[list[float]], b: list[list[float]], tol: float = TRANSFORM_TOLERANCE) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        if any(abs(x - y) > tol for x, y in zip(ra, rb)):
            return False
    return True


def scales_match(a: float, b: float, tol: float = TRANSFORM_TOLERANCE) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def jpeg_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def build_document(*, splat: dict[str, Any], lineage: dict[str, Any], cameras: list[dict[str, Any]],
                   intrinsics: str, pose_refinement: str, generator: str,
                   created_at: str | None = None) -> dict[str, Any]:
    """The file SuperSplat reads. Coordinates are the exported PLY's own
    (`frame: splat-local`); each `c2w` is a camera-to-world 3x4 in the OpenGL
    convention nerfstudio and PlayCanvas share: the camera looks down -Z with
    +Y up and +X right. Intrinsics are pixels of an image `width` x `height`."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "generator": generator,
        "createdAt": created_at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "convention": "opengl",
        "frame": "splat-local",
        "intrinsics": intrinsics,
        "poseRefinement": pose_refinement,
        "splat": splat,
        "lineage": lineage,
        "cameras": cameras,
    }


def atomic_write_json(path: str | Path, document: dict[str, Any]) -> int:
    """Write next to the target as `<name>.tmp`, then rename: a reader (or a
    manifest scraper) never sees half a file. Returns the size in bytes."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(data)


def script_source() -> str:
    """This file's text, for `python3 -c` inside the container."""
    return Path(__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------- container side


def _splat_info(ply: Path) -> dict[str, Any]:
    import numpy as np

    info = ply_header_info(ply)
    dtype = np.dtype([(name, _PLY_TYPES[kind]) for name, kind in info["properties"]])
    data = np.memmap(ply, dtype=dtype, mode="r", offset=info["offset"], shape=(info["count"],))
    bounds = None
    if info["count"] > 0 and all(k in dtype.names for k in ("x", "y", "z")):
        lo = [float(data[k].min()) for k in ("x", "y", "z")]
        hi = [float(data[k].max()) for k in ("x", "y", "z")]
        bounds = {"min": lo, "max": hi}
    return {"filename": ply.name, "gaussians": int(info["count"]), "bounds": bounds}


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import yaml
    from PIL import Image

    import nerfstudio
    import nerfstudio.configs.method_configs  # noqa: F401 — registers every method's yaml tags
    from nerfstudio.data.datasets.base_dataset import InputDataset
    from nerfstudio.data.utils.dataloaders import _undistort_image

    try:
        from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
    except ImportError:  # pragma: no cover — older nerfstudio
        FullImageDatamanager = None

    config_path = Path(args.config)
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    dm_cfg = config.pipeline.datamanager
    dp_cfg = dm_cfg.dataparser
    # what ns-train does: --data reaches the datamanager, which hands it on
    data = getattr(config, "data", None) or getattr(dm_cfg, "data", None)
    if data is not None:
        dp_cfg.data = Path(data)
    if hasattr(dp_cfg, "load_3D_points"):
        dp_cfg.load_3D_points = False  # cameras only: no point cloud, never a prompt
    dataparser = dp_cfg.setup()

    splits: list[tuple[str, Any]] = [("train", dataparser.get_dataparser_outputs(split="train"))]
    eval_note = None
    if getattr(dp_cfg, "eval_mode", None) == "all":
        eval_note = "eval_mode=all: every image trained, none held out"
    else:
        try:
            splits.append(("eval", dataparser.get_dataparser_outputs(split="val")))
        except Exception as e:  # noqa: BLE001 — a dataset without a held-out split
            eval_note = f"no held-out split ({type(e).__name__}: {e})"

    # The run recorded the transform it trained with; the dataparser just
    # re-derived one from the dataset as it is now.
    train_out = splits[0][1]
    recorded = config_path.parent / "dataparser_transforms.json"
    transform_check = "unchecked (no dataparser_transforms.json)"
    if recorded.is_file():
        rec = json.loads(recorded.read_text())
        now_t = train_out.dataparser_transform.numpy().tolist()
        now_s = float(train_out.dataparser_scale)
        if not (transforms_match(now_t, rec["transform"]) and scales_match(now_s, float(rec["scale"]))):
            raise RuntimeError(
                "the dataset changed since training: its transform/scale no longer match "
                f"{recorded} — the cameras would not line up with the splat"
            )
        transform_check = "matches dataparser_transforms.json"

    # Only the full-image datamanagers (every splatfacto variant) undistort.
    dm_target = getattr(dm_cfg, "_target", None)
    undistorts = bool(FullImageDatamanager and isinstance(dm_target, type) and issubclass(dm_target, FullImageDatamanager))

    total = sum(len(out.image_filenames) for _, out in splits)
    long_side = thumb_long_side(total, args.thumb)
    data_root = Path(dp_cfg.data) if getattr(dp_cfg, "data", None) else None
    if data_root is not None and data_root.suffix == ".json":
        data_root = data_root.parent

    def one(dataset: Any, outputs: Any, split: str, idx: int) -> dict[str, Any]:
        camera = dataset.cameras[idx].reshape(())
        c2w = camera.camera_to_worlds.numpy().astype(float)
        camera_type = int(camera.camera_type.item())
        record: dict[str, Any] = {
            "name": Path(outputs.image_filenames[idx]).name,
            "split": split,
            "cameraType": _CAMERA_TYPES.get(camera_type, str(camera_type)),
        }
        if data_root is not None:
            try:
                record["file"] = Path(outputs.image_filenames[idx]).relative_to(data_root).as_posix()
            except ValueError:
                pass
        sample = dataset.get_data(idx, image_type="uint8")
        image = sample["image"].numpy()
        K = camera.get_intrinsics_matrices().numpy().astype(np.float64)
        width, height = int(camera.width.item()), int(camera.height.item())
        dist = camera.distortion_params
        if undistorts and dist is not None and bool((dist != 0).any()):
            K, image, _mask = _undistort_image(camera, dist.numpy(), sample, image, K)
            height, width = int(image.shape[0]), int(image.shape[1])
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if not all(math.isfinite(v) for v in c2w.ravel()):
            raise RuntimeError(f"{record['name']}: non-finite pose")
        record.update({
            "c2w": [[float(v) for v in row] for row in c2w.tolist()],
            "width": width, "height": height,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        })
        pil = Image.fromarray(np.ascontiguousarray(image))
        tw, th = thumb_size(pil.width, pil.height, long_side)
        if (tw, th) != (pil.width, pil.height):
            pil = pil.resize((tw, th), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=args.quality, optimize=True)
        record["image"] = jpeg_data_uri(buf.getvalue())
        return record

    # "Held out" means not trained on. Some dataparsers (splatfacto-w's
    # NerfW: train = every image, eval = a subset of them) draw the eval set
    # from the training images; those cameras are already listed as train.
    trained_files = {str(p) for p in train_out.image_filenames}
    cameras: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for split, outputs in splits:
            dataset = InputDataset(outputs, scale_factor=1.0)
            indices = range(len(dataset))
            if split == "eval":
                indices = [i for i in indices if str(outputs.image_filenames[i]) not in trained_files]
            cameras.extend(pool.map(lambda i, d=dataset, o=outputs, s=split: one(d, o, s, i), indices))

    optimizer_mode = str(getattr(getattr(config.pipeline.model, "camera_optimizer", None), "mode", "off"))
    pose_refinement = "none" if optimizer_mode == "off" else f"not-applied (camera_optimizer.mode={optimizer_mode})"
    lineage = dict(json.loads(args.lineage))
    lineage.setdefault("config", str(config_path))
    lineage["dataparser"] = type(dataparser).__name__
    lineage["nerfstudio"] = getattr(nerfstudio, "__version__", None) or _pkg_version("nerfstudio")
    lineage["transformCheck"] = transform_check
    if eval_note:
        lineage["evalNote"] = eval_note
    return build_document(
        splat=_splat_info(Path(args.ply)),
        lineage=lineage,
        cameras=cameras,
        intrinsics="undistorted" if undistorts else "as-dataparser",
        pose_refinement=pose_refinement,
        generator=f"docker_splat_v2 training_cameras.py (nerfstudio {lineage['nerfstudio']})",
    )


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        document = _collect(args)
        size = atomic_write_json(args.out, document)
        counts = {"train": 0, "eval": 0}
        for cam in document["cameras"]:
            counts[cam["split"]] = counts.get(cam["split"], 0) + 1
        summary = {"out": str(args.out), "bytes": size, "cameras": len(document["cameras"]), **counts,
                   "intrinsics": document["intrinsics"]}
        print(f"\n{OK_SENTINEL} {json.dumps(summary)}", flush=True)
        return 0
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — one line the caller can show
        print(f"\n{FAIL_SENTINEL} {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

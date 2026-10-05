# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/

docker_splat_v2 overlay of KevinXu02/splatfacto-w's ``nerfw_dataparser.py``
(upstream commit 119a3bf), installed over the pip package by the Dockerfile so
``splatfacto-w`` works on standard COLMAP / ns-process-data layouts, not just
the phototourism ``dense/`` tree it was written for. Differences from upstream:

* ``colmap_path`` is honoured (upstream hardcoded ``dense/sparse``) and the
  image directory is auto-detected (``dense/images`` or ``images``).
* Cameras are read through nerfstudio's own ``parse_colmap_camera_params``, so
  every perspective-family COLMAP model is accepted (SIMPLE_PINHOLE, PINHOLE,
  SIMPLE_RADIAL, RADIAL, OPENCV) and the real image width/height plus the
  distortion coefficients are passed on. Upstream built ``Cameras`` without a
  size, which nerfstudio derives as ``int(2*cx) x int(2*cy)`` -- wrong as soon
  as COLMAP has refined the principal point -- and dropped distortion, so
  images were never undistorted. Fisheye models are rejected with a clear
  error instead of being mis-rendered.
* Image resolution follows the ``nerfstudio-data`` rule: an explicit
  ``--downscale-factor``, else the smallest existing ``images_<2^k>`` folder
  that brings the longest side to <= 1600 px (upstream always read full-res).
* The ``<data_name>.tsv`` train/test split is optional; without it a
  deterministic 10 % eval split is drawn (``eval_split_fraction``).
* Images are iterated per image (upstream iterated per camera model, which
  reads one row for a shared-intrinsics dataset).
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
)
from nerfstudio.utils.rich_utils import CONSOLE


# Same auto-downscale target nerfstudio-data uses (nerfstudio_dataparser.MAX_AUTO_RESOLUTION).
_MAX_AUTO_RESOLUTION = 1600

_IMAGE_DIR_CANDIDATES = ("dense/images", "images")


def _find_images_dir(data: Path) -> Path:
    """Return the first existing image directory under *data*."""
    for rel in _IMAGE_DIR_CANDIDATES:
        d = data / rel
        if d.is_dir():
            return d
    raise FileNotFoundError(
        f"Could not find an image directory in {data}. "
        f"Tried: {', '.join(_IMAGE_DIR_CANDIDATES)}"
    )


@dataclass
class NerfWDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: NerfW)
    """target class to instantiate"""
    data: Path = Path("data/brandenburg-gate")
    """Directory specifying location of data."""
    data_name: Literal["brandenburg", "trevi", "sacre"] = "brandenburg"
    """Name of the dataset."""
    scale_factor: float = 3.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    colmap_path: Path = Path("dense/sparse")
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""
    eval_split_fraction: float = 0.1
    """Fraction of images to hold out for eval when no TSV split file is present."""


@dataclass
class NerfW(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: NerfWDataParserConfig

    def __init__(self, config: NerfWDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.i_eval = None

    def _generate_dataparser_outputs(self, split="train"):
        image_filenames = []
        poses = []
        colmap_path = self.data / self.config.colmap_path
        full_res_images_dir = _find_images_dir(self.data)
        with CONSOLE.status(
            f"[bold green]Reading phototourism images and poses for {split} split..."
        ) as _:
            cams = read_cameras_binary(colmap_path / "cameras.bin")
            imgs = read_images_binary(colmap_path / "images.bin")

        images_dir, downscale = self._resolve_images_dir(full_res_images_dir, cams, imgs)
        self._downscale_factor = downscale

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        widths = []
        heights = []
        distort = []
        intrinsics = {}
        image_filenames = []

        flip = torch.eye(3)
        flip[0, 0] = -1.0
        flip = flip.double()

        # Load the TSV file for train/eval split if it exists (phototourism
        # layout).  Otherwise fall back to a deterministic fraction-based split.
        split_file = self.data / f"{self.config.data_name}.tsv"
        has_tsv = split_file.is_file()
        if has_tsv:
            split_data = pd.read_csv(split_file, sep="\t")
            split_data = split_data.dropna()

        for _id, img in imgs.items():
            cam = cams[img.camera_id]
            if has_tsv and img.name not in split_data[:]["filename"].values:
                continue
            if img.camera_id not in intrinsics:
                # Raises NotImplementedError for models nerfstudio cannot represent.
                parsed = parse_colmap_camera_params(cam)
                if parsed["camera_model"] != "OPENCV":
                    raise ValueError(
                        "The splatfacto-w dataparser supports perspective COLMAP cameras only "
                        "(SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV); "
                        f"got {cam.model!r}."
                    )
                intrinsics[img.camera_id] = parsed
            p = intrinsics[img.camera_id]

            pose = torch.cat(
                [torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))],
                dim=1,
            )
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(p["fl_x"]))
            fys.append(torch.tensor(p["fl_y"]))
            cxs.append(torch.tensor(p["cx"]))
            cys.append(torch.tensor(p["cy"]))
            widths.append(int(p["w"]))
            heights.append(int(p["h"]))
            distort.append(
                camera_utils.get_distortion_params(k1=p["k1"], k2=p["k2"], p1=p["p1"], p2=p["p2"])
            )

            image_filenames.append(images_dir / img.name)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()

        all_indices = torch.arange(len(image_filenames))
        # Create a mapping from image filenames to indices
        filename_to_index = {name: idx for idx, name in enumerate(image_filenames)}

        if has_tsv:
            eval_indices = [
                filename_to_index[images_dir / name]
                for name in split_data[split_data["split"] == "test"]["filename"].values
            ]
        else:
            n = len(image_filenames)
            n_eval = max(1, int(n * self.config.eval_split_fraction))
            rng = np.random.default_rng(42)
            eval_indices = sorted(rng.choice(n, size=n_eval, replace=False).tolist())

        eval_indices = torch.tensor(
            eval_indices,
            dtype=torch.long,
        )

        self.i_eval = eval_indices.tolist()
        # Print eval indices and corresponding filenames
        print(f"eval_indices: {eval_indices}")
        eval_filenames = [image_filenames[i] for i in eval_indices]
        print(f"eval_filenames: {eval_filenames}")

        if split == "train":
            indices = all_indices
        elif split == "val" or split == "test":
            indices = eval_indices
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            height=torch.tensor(heights, dtype=torch.int64),
            width=torch.tensor(widths, dtype=torch.int64),
            distortion_params=torch.stack(distort, dim=0),
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras.rescale_output_resolution(scaling_factor=1.0 / downscale)
        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]
        metadata = {}
        if split == "train":
            metadata.update(
                self._load_3D_points(colmap_path, transform_matrix, scale_factor)
            )
        assert len(cameras) == len(image_filenames)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

        return dataparser_outputs

    def _resolve_images_dir(self, images_dir: Path, cams, imgs):
        """Pick the image folder the way nerfstudio-data does: an explicit
        --downscale-factor, else the smallest existing images_<2^k> folder that
        brings the longest side to <= _MAX_AUTO_RESOLUTION px."""
        first = next(iter(imgs.values()))
        cam = cams[first.camera_id]

        def folder(df: int) -> Path:
            return images_dir if df == 1 else images_dir.parent / f"{images_dir.name}_{df}"

        if self.config.downscale_factor is not None:
            df = int(self.config.downscale_factor)
        else:
            df = 1
            while (
                max(cam.width, cam.height) / df > _MAX_AUTO_RESOLUTION
                and (folder(df * 2) / first.name).exists()
            ):
                df *= 2
        if not (folder(df) / first.name).exists():
            raise FileNotFoundError(
                f"Downscale factor {df} requested but {folder(df) / first.name} does not exist."
            )
        if df > 1:
            CONSOLE.log(f"splatfacto-w dataparser: image downscale factor of {df}")
        return folder(df), df

    def _load_3D_points(
        self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float
    ):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(
                colmap_path / "points3D.bin"
            )
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(
                colmap_path / "points3D.txt"
            )
        else:
            raise ValueError(
                f"Could not find points3D.txt or points3D.bin in {colmap_path}"
            )
        points3D = torch.from_numpy(
            np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
        )
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(
            np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
        )
        points3D_num_points = torch.tensor(
            [len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64
        )
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(
                np.array([p.error for p in colmap_points.values()], dtype=np.float32)
            ),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(
                    colmap_path / "images.txt"
                )
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(
                    colmap_path / "images.bin"
                )
            else:
                raise ValueError(
                    f"Could not find images.txt or images.bin in {colmap_path}"
                )
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(
                    max_num_points, self.config.max_2D_matches_per_3D_point
                )
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[
                        : self.config.max_2D_matches_per_3D_point
                    ]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [
                    im_id_to_image[im_id].xys[pt_idx]
                    for im_id, pt_idx in zip(nids, nxy_ids)
                ]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat(
                        (
                            nids,
                            torch.full(
                                (max_num_points - len(nids),), -1, dtype=torch.int64
                            ),
                        )
                    )
                )
                points3D_image_xy.append(
                    torch.cat(
                        (
                            nxy,
                            torch.full(
                                (max_num_points - len(nxy), nxy.shape[-1]),
                                0,
                                dtype=torch.float32,
                            ),
                        )
                    )
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def check_in_eval(self, idx):
        return idx in self.i_eval

    def find_eval_idx(self, idx):
        return self.i_eval[idx]


splatfactow_dataparser=DataParserSpecification(config=NerfWDataParserConfig())

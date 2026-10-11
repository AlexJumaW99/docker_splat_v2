"""Unit tests for app.training_cameras (host-side parts) and the warehouse's
training-cameras helpers (no Qt, no container).

Run from gui/:  .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import ast
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from app import training_cameras as tc
from app import warehouse as wh


def _write_ply(path: Path, rows: list[tuple[float, float, float, int]], fmt: str = "binary_little_endian") -> int:
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "comment Vertical Axis: z\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<fffB", *r) for r in rows)
    path.write_bytes(header + body)
    return len(header)


class ScriptTests(unittest.TestCase):
    def test_script_fits_in_one_argument_and_compiles(self):
        src = tc.script_source()
        # it rides as one argv element (Linux caps one argument at 128 KiB)
        self.assertLess(len(src.encode("utf-8")), 100 * 1024)
        compile(src, "training_cameras.py", "exec")

    def test_module_level_imports_are_stdlib_only(self):
        # the GUI imports this module and the container runs it with
        # `python3 -c`: nothing beyond the standard library at module level
        tree = ast.parse(tc.script_source())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names.add(node.module.split(".")[0])
        names.discard("__future__")
        self.assertTrue(names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names))

    def test_parse_args(self):
        a = tc.parse_args(["--config", "c.yml", "--ply", "s.ply", "--out", "s.cameras.json"])
        self.assertEqual((a.thumb, a.quality, a.workers, json.loads(a.lineage)), (0, tc.JPEG_QUALITY, 4, {}))
        with self.assertRaises(json.JSONDecodeError):
            tc.parse_args(["--config", "c", "--ply", "p", "--out", "o", "--lineage", "{not json"])


class ThumbnailTests(unittest.TestCase):
    def test_long_side(self):
        self.assertEqual(tc.thumb_long_side(266), tc.THUMB_LONG_SIDE)
        self.assertEqual(tc.thumb_long_side(tc.LARGE_SET + 1), tc.THUMB_LONG_SIDE_LARGE)
        self.assertEqual(tc.thumb_long_side(266, 512), 512)

    def test_size_keeps_aspect_and_never_upscales(self):
        self.assertEqual(tc.thumb_size(960, 540, 384), (384, 216))
        self.assertEqual(tc.thumb_size(2159, 3839, 384), (216, 384))
        self.assertEqual(tc.thumb_size(240, 240, 384), (240, 240))
        with self.assertRaises(ValueError):
            tc.thumb_size(0, 10, 384)

    def test_data_uri(self):
        self.assertEqual(tc.jpeg_data_uri(b"\xff\xd8\xff"), "data:image/jpeg;base64,/9j/")


class PlyHeaderTests(unittest.TestCase):
    def test_reads_count_properties_and_offset(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.ply"
            n = _write_ply(p, [(0, 1, 2, 9), (3, 4, 5, 8), (-1, -2, -3, 7)])
            info = tc.ply_header_info(p)
            self.assertEqual(info["count"], 3)
            self.assertEqual(info["offset"], n)
            self.assertEqual(info["properties"], [("x", "float"), ("y", "float"), ("z", "float"), ("red", "uchar")])

    def test_rejects_ascii_and_non_ply(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.ply"
            _write_ply(p, [(0, 0, 0, 0)], fmt="ascii")
            with self.assertRaises(ValueError):
                tc.ply_header_info(p)
            q = Path(d) / "b.ply"
            q.write_bytes(b"not a ply\n")
            with self.assertRaises(ValueError):
                tc.ply_header_info(q)


class DocumentTests(unittest.TestCase):
    def test_transform_and_scale_tolerance(self):
        a = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        b = [[1.0, 0.0, 0.0, 0.5 + 5e-5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        c = [[1.0, 0.0, 0.0, 0.5 + 5e-3], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        self.assertTrue(tc.transforms_match(a, b))
        self.assertFalse(tc.transforms_match(a, c))
        self.assertFalse(tc.transforms_match(a, a[:2]))
        self.assertTrue(tc.scales_match(0.2369862505955303, 0.23698625))
        self.assertFalse(tc.scales_match(0.2369, 0.2469))

    def test_build_and_atomic_write(self):
        doc = tc.build_document(
            splat={"filename": "s.ply", "gaussians": 3, "bounds": None}, lineage={"export": "X"},
            cameras=[{"name": "f.png", "split": "train"}], intrinsics="undistorted",
            pose_refinement="none", generator="test", created_at="2026-10-11T00:00:00+00:00",
        )
        self.assertEqual((doc["format"], doc["version"], doc["convention"], doc["frame"]),
                         (tc.FORMAT, tc.VERSION, "opengl", "splat-local"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s.cameras.json"
            size = tc.atomic_write_json(out, doc)
            self.assertEqual(size, out.stat().st_size)
            self.assertEqual(json.loads(out.read_text()), doc)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["s.cameras.json"])


class WarehouseHelperTests(unittest.TestCase):
    def test_cameras_path(self):
        self.assertEqual(wh.training_cameras_path(Path("/w/exports/splat/X/fort.ply")),
                         Path("/w/exports/splat/X/fort.cameras.json"))

    def test_outputs_replace_sidecar_entry_and_stay_pure(self):
        rel = "exports/splat/X/fort.cameras.json"
        outputs = {"splat_ply": "exports/splat/X/fort.ply", "sidecar:fort.cameras.json": rel,
                   "sidecar:job.log": "exports/splat/X/job.log"}
        checksums = {"exports/splat/X/fort.ply": "sha256:aa"}
        new_o, new_c = wh.training_camera_outputs(outputs, checksums, rel, "sha256:bb")
        self.assertEqual(new_o, {"splat_ply": "exports/splat/X/fort.ply",
                                 "sidecar:job.log": "exports/splat/X/job.log",
                                 "training_cameras": rel})
        self.assertEqual(new_c, {"exports/splat/X/fort.ply": "sha256:aa", rel: "sha256:bb"})
        self.assertIn("sidecar:fort.cameras.json", outputs)  # inputs untouched
        self.assertEqual(checksums, {"exports/splat/X/fort.ply": "sha256:aa"})

    def test_outputs_replace_a_previous_cameras_file(self):
        old, new = "exports/splat/X/a.cameras.json", "exports/splat/X/b.cameras.json"
        o, c = wh.training_camera_outputs({"training_cameras": old}, {old: "sha256:1"}, new, "sha256:2")
        self.assertEqual((o, c), ({"training_cameras": new}, {new: "sha256:2"}))

    def test_command_arguments(self):
        args = wh.training_cameras_command(config="/warehouse/c.yml", ply="/warehouse/s.ply",
                                           out="/warehouse/s.cameras.json", lineage={"export": "X"})
        self.assertEqual(args[:6], ["--config", "/warehouse/c.yml", "--ply", "/warehouse/s.ply",
                                    "--out", "/warehouse/s.cameras.json"])
        self.assertEqual(json.loads(args[args.index("--lineage") + 1]), {"export": "X"})
        # and the script accepts exactly these
        a = tc.parse_args(args)
        self.assertEqual(a.out, "/warehouse/s.cameras.json")


if __name__ == "__main__":
    unittest.main()

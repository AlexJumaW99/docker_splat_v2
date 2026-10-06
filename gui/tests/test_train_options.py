"""Unit tests for app.train_options (no Qt, no container).

Run from gui/:  .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import train_options as to


def _make_dataset(root: Path, size: tuple[int, int], levels: tuple[int, ...], frames: int = 3,
                  transforms_frames: int | None = None, manifest_metrics: dict | None = None) -> None:
    """Write a fake processed dataset: images/ at `size`, images_<N>/ for each
    N in `levels`, optional transforms.json and manifest.json."""
    for f in (1, *levels):
        folder = root / ("images" if f == 1 else f"images_{f}")
        folder.mkdir(parents=True)
        w, h = size[0] // f, size[1] // f
        for i in range(frames):
            Image.new("RGB", (w, h)).save(folder / f"frame_{i:05d}.png")
    if transforms_frames is not None:
        (root / "transforms.json").write_text(json.dumps({
            "w": size[0], "h": size[1],
            "frames": [{"file_path": f"images/frame_{i:05d}.png"} for i in range(transforms_frames)],
        }))
    if manifest_metrics is not None:
        (root / "manifest.json").write_text(json.dumps({"stage": "process", "metrics": manifest_metrics}))


class ImageLevelsTests(unittest.TestCase):
    def test_levels_and_auto_factor_4k_with_three_levels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (3840, 2160), (2, 4, 8))
            levels = to.dataset_image_levels(root)
            self.assertEqual(sorted(levels), [1, 2, 4, 8])
            self.assertEqual((levels[4].width, levels[4].height, levels[4].count), (960, 540, 3))
            # 3840 > 1600, 1920 > 1600, 960 <= 1600  -> factor 4, exactly nerfstudio's choice
            self.assertEqual(to.nerfstudio_auto_factor(levels), 4)

    def test_auto_factor_stops_at_last_existing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (3840, 2160), (2,))
            self.assertEqual(to.nerfstudio_auto_factor(to.dataset_image_levels(root)), 2)

    def test_auto_factor_is_1_without_folders_even_at_4k(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (3840, 2160), ())
            self.assertEqual(to.nerfstudio_auto_factor(to.dataset_image_levels(root)), 1)

    def test_small_source_never_downscales(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (1280, 720), (2, 4))
            self.assertEqual(to.nerfstudio_auto_factor(to.dataset_image_levels(root)), 1)

    def test_ignores_empty_level_folders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (1920, 1080), ())
            (root / "images_2").mkdir()
            self.assertEqual(sorted(to.dataset_image_levels(root)), [1])


class ResolutionOptionTests(unittest.TestCase):
    def test_options_for_dataset_with_levels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (3840, 2160), (2, 4, 8))
            opts = to.resolution_options(root)
            self.assertEqual([o.key for o in opts], ["auto", "full", "down:2", "down:4", "down:8"])
            auto = opts[0]
            self.assertEqual((auto.factor, auto.width, auto.height), (4, 960, 540))
            self.assertIn("960 × 540", auto.label)
            self.assertEqual(auto.dataparser_flags(), [])          # auto emits nothing
            self.assertEqual(auto.pipeline_flags(), [])
            half = to.find_option(opts, "down:2")
            self.assertEqual(half.dataparser_flags(), ["--downscale-factor", "2"])
            self.assertEqual(half.pipeline_flags(), [])
            full = to.find_option(opts, "full")
            self.assertEqual((full.dataparser_flags(), full.pipeline_flags()), ([], []))

    def test_options_for_dataset_without_levels_use_rescale(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _make_dataset(root, (3840, 2160), ())
            opts = to.resolution_options(root)
            self.assertEqual([o.key for o in opts], ["auto", "full", "scale:2", "scale:4", "scale:8"])
            self.assertEqual(opts[0].factor, 1)                      # auto == full here
            half = to.find_option(opts, "scale:2")
            self.assertFalse(half.via_folder)
            self.assertEqual(half.dataparser_flags(), [])
            self.assertEqual(half.pipeline_flags(), ["--pipeline.datamanager.camera-res-scale-factor", "0.5"])
            self.assertEqual((half.width, half.height), (1920, 1080))
            self.assertEqual(to.missing_levels(root), [2, 4, 8])

    def test_no_images_dir_gives_no_options(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(to.resolution_options(Path(d)), [])


class CacheTests(unittest.TestCase):
    def test_estimate_matches_measured_4k_cache(self):
        # 240 train frames at 3840x2160 uint8 measured as ~5.97 GiB of GPU cache.
        self.assertAlmostEqual(to.estimate_cache_gib(240, 3840, 2160), 5.56, places=1)

    def test_auto_switches_to_cpu_only_above_threshold(self):
        self.assertIsNone(to.choose_cache_images("auto", 261, 960, 540))       # 0.38 GiB
        self.assertEqual(to.choose_cache_images("auto", 266, 3840, 2160), "cpu")  # 6.2 GiB
        self.assertEqual(to.choose_cache_images("gpu", 266, 3840, 2160), "gpu")
        self.assertEqual(to.choose_cache_images("cpu", 10, 100, 100), "cpu")


class PoseRecoveryTests(unittest.TestCase):
    def test_disk_wins_over_a_stale_manifest(self):
        # vizer once recorded 209 extracted frames while 267 were on disk; the
        # readout must follow the directory, which is what training loads.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, (640, 480), (), frames=4, transforms_frames=4,
                          manifest_metrics={"extracted_frame_count": 75.0, "registered_frame_count": 2.0})
            pr = to.pose_recovery(root)
            self.assertEqual((pr.registered, pr.extracted, pr.source), (4, 4, "files"))
            self.assertEqual(pr.level, "success")

    def test_manifest_used_when_files_are_gone(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text(json.dumps({"stage": "process", "metrics": {"extracted_frame_count": 75.0, "registered_frame_count": 2.0}}))
            pr = to.pose_recovery(root)
            self.assertEqual((pr.registered, pr.extracted, pr.source), (2, 75, "manifest"))
            self.assertAlmostEqual(pr.ratio, 2 / 75)
            self.assertEqual(pr.level, "error")

    def test_falls_back_to_transforms_and_image_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, (640, 480), (), frames=10, transforms_frames=9,
                          manifest_metrics={"extracted_frame_count": 10.0})
            pr = to.pose_recovery(root)
            self.assertEqual((pr.registered, pr.extracted, pr.source), (9, 10, "files"))
            self.assertEqual(pr.level, "success")

    def test_warning_band(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, (640, 480), (), frames=10, transforms_frames=7)
            self.assertEqual(to.pose_recovery(root).level, "warning")

    def test_unknown_when_nothing_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            pr = to.pose_recovery(Path(d))
            self.assertIsNone(pr.ratio)
            self.assertEqual(pr.level, "neutral")


class ArgvTests(unittest.TestCase):
    def _base(self, **over):
        kw = dict(method="splatfacto", data="/warehouse/processed/X", output_dir="/warehouse/trained/Y",
                  experiment_name="proj", max_num_iterations=7000, steps_per_save=2000,
                  vis="viewer+tensorboard", quit_on_train_completion=False,
                  steps_per_eval_image=1000, steps_per_eval_all_images=1000,
                  cache_images=None, rasterize_mode=None, max_gs_num=None, resolution=None, extra_flags=[],
                  project_dataparser="nerfstudio-data", colmap_path="colmap/sparse/0")
        kw.update(over)
        return to.build_train_argv(**kw)

    def test_default_nerfstudio_data_has_no_subcommand(self):
        argv = self._base()
        self.assertEqual(argv[:2], ["ns-train", "splatfacto"])
        self.assertNotIn("nerfstudio-data", argv)
        self.assertEqual(argv[argv.index("--steps-per-eval-image") + 1], "1000")
        self.assertIn("--viewer.quit-on-train-completion", argv)

    def test_downscale_adds_subcommand_last(self):
        half = to.ResolutionOption("down:2", "x", 2, 1920, 1080, True)
        argv = self._base(resolution=half, extra_flags=["--pipeline.model.cull-alpha-thresh", "0.005"])
        self.assertEqual(argv[-3:], ["nerfstudio-data", "--downscale-factor", "2"])
        # raw flags stay before the subcommand
        self.assertLess(argv.index("--pipeline.model.cull-alpha-thresh"), argv.index("nerfstudio-data"))

    def test_rescale_option_goes_before_subcommand(self):
        q = to.ResolutionOption("scale:4", "x", 4, 960, 540, False)
        argv = self._base(resolution=q, project_dataparser="colmap", colmap_path="sparse/0")
        i = argv.index("--pipeline.datamanager.camera-res-scale-factor")
        self.assertEqual(argv[i + 1], "0.25")
        self.assertEqual(argv[-2:], ["--colmap-path", "sparse/0"])
        self.assertLess(i, argv.index("colmap"))

    def test_splatfacto_w_routing(self):
        half = to.ResolutionOption("down:2", "x", 2, 1920, 1080, True)
        argv = self._base(method="splatfacto-w", resolution=half, cache_images="cpu", rasterize_mode="classic")
        self.assertEqual(argv[-5:], ["splatfactow_dataparser", "--colmap-path", "colmap/sparse/0", "--downscale-factor", "2"])
        self.assertEqual(argv[argv.index("--pipeline.datamanager.cache-images") + 1], "cpu")
        self.assertEqual(argv[argv.index("--pipeline.model.rasterize-mode") + 1], "classic")

    def test_splatfacto_w_light_uses_standard_dataparser(self):
        argv = self._base(method="splatfacto-w-light")
        self.assertNotIn("splatfactow_dataparser", argv)
        argv = self._base(method="splatfacto-w-light", project_dataparser="colmap")
        self.assertEqual(argv[-3:], ["colmap", "--colmap-path", "colmap/sparse/0"])

    def test_headless_vis_has_no_viewer_flags(self):
        argv = self._base(vis="tensorboard")
        self.assertNotIn("--viewer.websocket-host", argv)

    def test_max_gs_num_absent_by_default(self):
        self.assertNotIn(to.MAX_GS_NUM_FLAG, self._base(method="splatfacto-mcmc"))

    def test_max_gs_num_goes_before_subcommand(self):
        half = to.ResolutionOption("down:2", "x", 2, 1920, 1080, True)
        argv = self._base(method="splatfacto-mcmc", max_gs_num=2_000_000, resolution=half)
        i = argv.index("--pipeline.model.max-gs-num")
        self.assertEqual(argv[i + 1], "2000000")
        self.assertLess(i, argv.index("nerfstudio-data"))


class McmcBudgetTests(unittest.TestCase):
    def test_only_mcmc_above_default_emits(self):
        self.assertEqual(to.choose_max_gs_num("splatfacto-mcmc", 2_000_000), 2_000_000)
        self.assertIsNone(to.choose_max_gs_num("splatfacto-mcmc", to.MAX_GS_NUM_DEFAULT))
        self.assertIsNone(to.choose_max_gs_num("splatfacto-mcmc", None))
        # splatfacto / -big ignore the flag; splatfacto-w / -w-light reject it
        for method in ("splatfacto", "splatfacto-big", "splatfacto-w", "splatfacto-w-light"):
            self.assertIsNone(to.choose_max_gs_num(method, 4_000_000))

    def test_raw_cli_flag_wins_over_dropdown(self):
        for raw in (["--pipeline.model.max-gs-num", "3000000"],
                    ["--pipeline.model.max-gs-num=3000000"],
                    ["--pipeline.model.max_gs_num", "3000000"]):
            self.assertIsNone(to.choose_max_gs_num("splatfacto-mcmc", 2_000_000, raw))
        other = ["--pipeline.model.cull-alpha-thresh", "0.005"]
        self.assertEqual(to.choose_max_gs_num("splatfacto-mcmc", 2_000_000, other), 2_000_000)

    def test_choices_start_at_the_default(self):
        self.assertEqual(to.MAX_GS_NUM_CHOICES[0], to.MAX_GS_NUM_DEFAULT)
        self.assertEqual(to.max_gs_num_label(1_000_000), "1 M · nerfstudio default")
        self.assertEqual(to.max_gs_num_label(2_000_000), "2 M")

    def test_cost_measured_interpolated_and_out_of_range(self):
        two = to.mcmc_budget_cost(2_000_000)
        self.assertTrue(two.measured)
        self.assertAlmostEqual(two.peak_vram_gib, 4.229)
        three = to.mcmc_budget_cost(3_000_000)
        self.assertFalse(three.measured)
        self.assertIsNone(three.psnr)
        self.assertAlmostEqual(three.peak_vram_gib, (4.229 + 7.551) / 2)
        self.assertAlmostEqual(three.train_minutes, (16.3 + 28.3) / 2)
        self.assertIsNone(to.mcmc_budget_cost(6_000_000))
        self.assertIsNone(to.mcmc_budget_cost(500_000))

    def test_every_choice_has_a_caption(self):
        for n in to.MAX_GS_NUM_CHOICES:
            self.assertIsNotNone(to.mcmc_budget_cost(n))

    def test_caption_wording(self):
        self.assertEqual(
            to.max_gs_num_caption(1_000_000, 960),
            "measured at 960 px, 30k steps: 2.6 GiB peak VRAM · 10 min · 238 MB PLY",
        )
        self.assertIn("+0.39 dB PSNR vs 1 M", to.max_gs_num_caption(2_000_000, 960))
        self.assertIn("+0.71 dB PSNR vs 1 M", to.max_gs_num_caption(4_000_000, None))
        three = to.max_gs_num_caption(3_000_000, 960)
        self.assertTrue(three.startswith("≈ 5.9 GiB peak VRAM"))
        self.assertIn("interpolated", three)
        self.assertNotIn("dB", three)

    def test_caption_warns_above_measured_resolution(self):
        self.assertNotIn("⚠", to.max_gs_num_caption(2_000_000, 960))
        self.assertNotIn("⚠", to.max_gs_num_caption(2_000_000, 480))
        self.assertIn("VRAM at 1920 px is unknown", to.max_gs_num_caption(2_000_000, 1920))


if __name__ == "__main__":
    unittest.main()

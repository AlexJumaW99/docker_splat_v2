# Improvement plan

Prioritized backlog for `docker_splat_v2`, written 2026-10-05 after a full audit of the
five training methods, the matching paths and the camera-type handling. Every number
below was measured on this machine (RTX 5070 Ti 16 GB, image `a4f876fc`, nerfstudio
`50e0e3c`, gsplat 1.5.3, COLMAP `ffb5404e`) unless marked *not verified*. Benchmark
details are in the appendix; keep this file current as items land.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (date noted; see `git log` for the commit).

---

## Tier 1 — fix what is broken today (landed 2026-10-05, image v4.7)

### 1.1 Route only `splatfacto-w` through `splatfactow_dataparser` `[x]`
- **Problem.** `gui/app/views/train.py` sends both `splatfacto-w` and `splatfacto-w-light`
  through the NerfW dataparser (since `83606a2`). `splatfacto-w-light`'s upstream default is
  `nerfstudio-data`; the upstream author recommends exactly that for non-phototourism
  captures (splatfacto-w README, issue #12). The NerfW parser also puts *every* image in
  the train split, so -light's eval metrics were measured on training images
  (261 train / 26 eval instead of 235 / 26).
- **Change.** One condition: only `splatfacto-w` gets the `splatfactow_dataparser
  --colmap-path` subcommand; `-light` falls through to the same dataparser logic as the
  built-in methods. Pre-flight: refuse to start `splatfacto-w` on a project without a
  COLMAP sparse model instead of letting it crash on `dense/sparse/cameras.bin`.
- **Validation.** -light with `nerfstudio-data`: 7,000 iterations, 250,083 gaussians,
  PSNR 30.8, PLY export, `ns-viewer`, `ns-render` all OK. -light with
  `colmap --colmap-path`: 2,000 iterations + export OK.

### 1.2 NerfW dataparser v2 (`image_bin/nerfw_dataparser.py`) `[x]`
- **Problem.** The patched parser built `Cameras(...)` without width/height, so nerfstudio
  derived them as `int(2·cx) × int(2·cy)`. nerfstudio's `run_colmap` refines the
  principal point, so 13 of 14 OPENCV warehouse datasets mismatched their real image
  size (e.g. 3838×2161 vs 3840×2160). Symptoms: -light asserted *"The size of image …
  does not match the camera parameters"*; splatfacto-w skipped every frame recursively
  and, on 4K data, died with a misleading **CUDA out of memory** (386 skipped frames,
  9.6 GiB). The parser also accepted only PINHOLE/OPENCV (all 6 SIMPLE_PINHOLE / hloc
  datasets asserted), dropped OPENCV distortion, and always read full-resolution
  `images/`. Net effect: 19 of the 20 processed datasets with a COLMAP model could not
  start either w-method from the GUI.
- **Change.** Reuse nerfstudio's own `parse_colmap_camera_params`; pass real width/height
  and distortion; accept the perspective family (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL,
  RADIAL, OPENCV) and reject fisheye models with a clear message; pick `images_<N>` the
  way `nerfstudio-data` does (longest side ≤ 1600 px, or `--downscale-factor`).
- **Validation.** Produces identical intrinsics/undistortion to `nerfstudio-data` on
  the same dataset (959×539 after undistortion, same k1/k2/p1/p2). splatfacto-w 7,000
  iterations on an off-centre OPENCV dataset (253,578 gaussians, PSNR 29.4, export OK) and
  on a SIMPLE_PINHOLE dataset (1920×1080 via `images_2`); 4K dataset loads; `ns-viewer`
  serves; `ns-render` wrote a 480-frame 4K video.
- *Not verified:* the original phototourism layout (`dense/sparse`, `dense/images`) — the
  code path is unchanged by design but no such dataset is on disk; fisheye datasets.

### 1.3 splatfacto-w colour-cache guard (`image_bin/splatfactow_model_cache.patch`) `[x]`
- **Problem.** Upstream `splatfactow_model.py` reuses `cached_colors` in eval whenever the
  eval camera index equals the last training camera index, without checking that the
  gaussian count is unchanged. A densify/cull in between hands gsplat a stale
  `[N, 16, 3]` tensor → `AssertionError: torch.Size([…, 16, 3])` (upstream issue #14, open
  since 2024-08). Intermittent: 1 of 2 identical 7,000-iteration runs crashed at step
  ~2,300; with 12 training images it crashed at step 1,590.
- **Change.** Three lines: only eval renders populate `cached_colors` / `cached_bg_sh`,
  and a cached tensor must match the current gaussian count. Applied with
  `patch --forward --batch` so the build fails loudly if upstream drifts.
- **Validation.** 12-image stress run completes 7,000 iterations; deterministic
  checkpoint test (train render → cull → eval render on the same camera) passes.
  -light is not exposed (its eval cameras carry no `cam_idx`).

### 1.4 Bake hloc into the image `[x]`
- **Problem.** hloc 1.5, LightGlue, kornia and pycolmap existed only in the running
  container's writable layer (`docker diff`), not in the Dockerfile. Any rebuild or
  container recreation silently removed the `--sfm-tool hloc` route used for the
  DRC datasets.
- **Change.** Dockerfile step pinned to hloc `c13273b` (SuperGlue submodule only),
  LightGlue `eb42fee`, `pycolmap==3.13.0`, `kornia==0.8.3`, `kornia_rs==0.1.14`; verified
  in a throwaway container from the current image (imports OK, `pip check` clean,
  torch/gsplat/numpy unchanged).
- **Known limits.** Of nerfstudio's hloc options, `r2d2`, `d2net-ss` and `adalam` do not
  import in this install (submodules not checked out / kornia API); NetVLAD weights are
  downloaded on the first non-exhaustive hloc run; SuperPoint/SuperGlue weights are
  licensed for non-commercial research use only (`disk+lightglue` is the permissive
  alternative, untested here).

### 1.5 Smoke test + docs for the above `[x]`
- `smoke_test.sh` §17 (parser v2 marker, cache-patch anchors, radii fix, camera size/distortion probe) and §18 (hloc imports from `/opt/hloc`, module imports, SuperPoint weights): 76/76 pass on the rebuilt container.
- README / USER_GUIDE: stack table, fixes table, splatfacto-w guidance, hloc usage, troubleshooting entries.

### Post-rebuild validation (image `4bc97967`, 2026-10-05)
Build took 27 s (all heavy layers cached). In the recreated container: splatfacto-w with the GUI's exact arguments trained on the off-centre OPENCV dataset (1,500 its, export + eval OK) and on the SIMPLE_PINHOLE dataset; the deterministic colour-cache check passed on the installed package; the 12-image stress run completed 3,000 iterations; splatfacto-w-light with the corrected routing trained and exported; built-in splatfacto unchanged; `--sfm-tool hloc --feature-type superpoint_max --matcher-type superpoint+lightglue` registered 87/87 frames in 80 s; vocab_tree 87/87 in 34 s. Rollback image: `splat-nerfstudio-v2:pre-v4.7`.

---

## Tier 2 — VRAM behaviour and silent failures (all measured, not yet implemented)

### 2.1 Raise `--steps-per-eval-image` / `--steps-per-eval-all-images` `[ ]`
- Per-call tracing of `Trainer.train_iteration` / `eval_iteration`: the VRAM high-water
  mark of every run is the **first eval render at step 100** (nerfstudio default, GUI does
  not override), drawn at full eval resolution while the gaussians are still the large
  initial SfM points. 3.7 GiB at 960 px, 11.1 GiB at 1920 px (two datasets), 5.4 GiB at
  4K, against training steps of 0.6 / 1.8 / 0.8 GiB. With eval disabled the 1920 px run
  never exceeded 1.79 GiB. The same cadence wrote 1.56 GB of TensorBoard images for an
  8,500-step 4K run. Proposed default: 1000 for both flags (eval at step 1000 measured
  2.45 GiB at 1920 px).

### 2.2 Training-resolution control in the Train tab `[ ]`
- nerfstudio's auto rule trains 4K sources at 960 px (`images_4`). A splat is only good
  at the resolution it was trained at (splatfacto, 30k, `buddha_v2`):

  | Trained at | PSNR viewed @960 | PSNR viewed @1920 | Wall time |
  |---|---|---|---|
  | 960 px, classic (today's default) | 33.93 | 28.20 | 370 s |
  | 960 px, antialiased | 34.85 | 30.24 | 383 s |
  | 1920 px, classic (`--downscale-factor 2`) | 29.11 | 32.46 | 892 s |

- Implementation: `nerfstudio-data --downscale-factor N` (also `colmap`) when `images_N`
  exists; `--pipeline.datamanager.camera-res-scale-factor` for datasets processed with 0
  downscale levels (verified: 216 s / 6,500 its at 1920 px on `fort_garry`); an
  "Add downscale levels" action using nerfstudio's `downscale_images()` (167 s for
  266 4K frames). Required for splatfacto-w, which peaked at 9.5 GiB at 1080p.
- `antialiased` mode is a quality option for in-app viewing/rendering; nerfstudio's own
  docstring warns the exported PLY is not compatible with classic-mode web viewers
  (*not verified* against SuperSplat).

### 2.3 Auto-select `--pipeline.datamanager.cache-images cpu` for large frame sets `[ ]`
- splatfacto-big at native 4K (`fort_garry`, 266 frames, 6,500 its): GPU cache peak
  11.64 GiB / 581 s; CPU cache peak 5.63 GiB / 781 s; same 1.97 M gaussians.
  *Not verified:* whether a full 30k native-4K splatfacto-big run then completes.

### 2.4 Pose-recovery gate `[ ]`
- Record `registered_frame_count = len(transforms.json frames)` in the ProcessManifest
  (no log parsing needed) and warn/block in the Train tab when the ratio to extracted
  frames is low. Three warehouse datasets have 2/16, 2/75 and 23/253 posed frames; test
  runs on a 61-frame set exited 0 with 2–3 frames registered; the existing splinter
  check only fires with ≥ 2 submaps.

### 2.5 Persist job stdout per run `[ ]`
- `JobManager` has no file sink: 6 of 7 unfinalized splatfacto-big runs left nothing to
  diagnose. Write `<run>/job.log` next to `pending.json`; keep it on failure. Also warn
  when `--steps-per-save` exceeds a few thousand (yesterday's 4K run lost 8,500 steps
  with `steps_per_save 10000`).

### 2.6 docker-compose: add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` `[ ]`
- `PYTORCH_ALLOC_CONF` alone is inert in the NGC 25.11 torch build (allocator snapshot
  `is_expandable=False`); setting both enables it without the deprecation warning.
  PyTorch documents expandable segments for "a job changing allocation sizes
  frequently". 4K run: peak reserved 12.34 → 11.92 GiB; wall time 581 → 667 s on
  single runs (first run's reserve was pinned at a test cap, so inconclusive).

---

## Tier 3 — defaults and hygiene

- `[ ]` Default `Max iterations` 30000 (upstream presets' schedules assume it); keep 7k as
  a "preview" preset. Measured: 7k → 30k = PSNR 30.99 → 33.93 (87 s → 370 s) at 960 px.
- `[ ]` `smoke_test.sh`: 300-iteration `ns-train` smoke per method + the deterministic
  cache check; the current suite passed 62/62 while two methods could not start.
- `[x]` Dockerfile/README: gsplat 1.5.3 does *not* expose 2DGS/3DGUT/fisheye through
  nerfstudio's splatfacto (0 references in `splatfacto.py`); corrected wording.
- `[ ]` `quit-on-train-completion` default or a VRAM-held warning: a finished viewer held
  10.7 GB for 15 h.
- `[ ]` Manifest: record the effective dataparser for splatfacto-w runs
  (`splatfactow_dataparser`), and typed SfM fields (`sfm_tool`, `feature_type`,
  `matcher_type`) instead of free-text `extra_flags`; note that under hloc
  `sequential`/`vocab_tree` both mean NetVLAD top-50 retrieval.
- `[ ]` Investigate one vizer-processed manifest recording 209 extracted frames while
  267 exist on disk.
- `[ ]` Optional COLMAP upgrade to a tagged release (4.2.1 as of 2026-09-29): release
  notes list mapper speedups and bug fixes but nothing about the submap issue the shim
  handles; *not tested here*, and the flag-rename patch, shim and vocab tree would need
  re-validation.

## Do not change

nerfstudio pin (`50e0e3c` equals upstream main), gsplat 1.5.3 (latest release; 1.6.0
unreleased), splatfacto-w pin (equals HEAD, repo idle since 2024-11), the vocab tree
(87/87 frames registered), the COLMAP shim, the equirectangular path (336/336 crops
registered). splatfacto / -big / -mcmc train, export and evaluate cleanly.

---

## Appendix — benchmark log (2026-10-05)

| Run | Setup | Result |
|---|---|---|
| splatfacto 3,500 its | buddha_v2 @960 | 232,059 gaussians, PSNR 26.5, export OK |
| splatfacto-big 3,500 its | buddha_v2 @960 | 449,436 gaussians, PSNR 27.1, export OK |
| splatfacto-mcmc 3,500 its | buddha_v2 @960 | 541,117 gaussians, PSNR 27.1, export OK |
| splatfacto 7k / 30k | buddha_v2 @960 | PSNR 30.99 / 33.93, 87 s / 370 s |
| splatfacto-w-light, GUI routing (old) | OPENCV dataset | AssertionError (image size vs camera) |
| splatfacto-w, GUI routing (old) | OPENCV dataset, 4K | 386 skipped frames → CUDA OOM at 9.6 GiB |
| splatfacto-w / -light, GUI routing (old) | SIMPLE_PINHOLE dataset | AssertionError (camera model) |
| splatfacto-w-light, default parser | buddha_v2 @960, 7k | 250,083 gaussians, PSNR 30.8, export OK |
| splatfacto-w, parser v2 + cache guard | buddha_v2 @960, 7k | 253,578 gaussians, PSNR 29.4, export/viewer/render OK |
| splatfacto-w, parser v2 + cache guard | drc SIMPLE_PINHOLE @1920, 7k | 204,946 gaussians, peak 9.49 GiB |
| splatfacto-w, parser v2, installed model | 12-image subset | crash step 1,590 (`torch.Size([95371, 16, 3])`) |
| splatfacto-w, parser v2 + cache guard | 12-image subset | 7,000 its OK |
| vocab_tree / sequential | 87-frame buddha subset | 87/87 registered, 45 s / 24 s |
| exhaustive / sequential / vocab_tree | 61-frame long-focal subset | 2–3/61 registered for all three (dataset, not matcher) |
| equirectangular video, sequential | 42 frames → 336 crops | 336/336 registered, 321 s |
| splatfacto-big native 4K, GPU cache | fort_garry, 6,500 its | 1.96 M gaussians, peak 11.64 GiB, 581 s |
| same, expandable segments on | | peak reserved 11.92 vs 12.34 GiB, 667 s |
| same, `cache-images cpu` | | peak 5.63 GiB, 781 s |
| same, `camera-res-scale-factor 0.5` | | 1.02 M gaussians, 216 s |
| VRAM trace @960 / @1920 / @4K | per-call peaks | eval@100: 3.72 / 11.12 / 5.43 GiB; train ≤ 0.59 / 1.83 / 0.80 GiB |

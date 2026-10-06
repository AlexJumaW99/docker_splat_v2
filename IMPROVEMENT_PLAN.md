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

## Tier 2 — VRAM behaviour and silent failures (landed 2026-10-06, image v4.8)

### 2.1 Eval cadence `[x]`
- Per-call tracing of `Trainer.train_iteration` / `eval_iteration` showed the VRAM
  high-water mark of every run was the **first eval image**: 3.7 GiB at 960 px, 11.1 GiB at
  1920 px (two datasets), 5.4 GiB at 4K, against training steps of 0.6 / 1.8 / 0.8 GiB.
  **Correction (2026-10-06):** moving the first eval to step 1000 left the peak unchanged
  (11.15 GiB) — the spike belongs to the first eval, not to step 100; its cause and fix are
  item 2.7. What cadence does control is the other cost of evaluating every 100 steps:
  one full-resolution render + PSNR/SSIM/LPIPS per 100 steps and ~180 MB of TensorBoard
  images per 1000 steps at 4K (1.56 GB for an 8,500-step run). The GUI now emits
  `--steps-per-eval-image 1000 --steps-per-eval-all-images 1000` (Advanced → Eval cadence)
  and records both in `TrainParams`.

### 2.2 Training-resolution control in the Train tab `[x]`
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

### 2.3 Auto-select `--pipeline.datamanager.cache-images cpu` for large frame sets `[x]`
- splatfacto-big at native 4K (`fort_garry`, 266 frames, 6,500 its): GPU cache peak
  11.64 GiB / 581 s; CPU cache peak 5.63 GiB / 781 s; same 1.97 M gaussians.
  *Not verified:* whether a full 30k native-4K splatfacto-big run then completes.

### 2.4 Pose-recovery gate `[x]`
- Record `registered_frame_count = len(transforms.json frames)` in the ProcessManifest
  (no log parsing needed) and warn/block in the Train tab when the ratio to extracted
  frames is low. Three warehouse datasets have 2/16, 2/75 and 23/253 posed frames; test
  runs on a 61-frame set exited 0 with 2–3 frames registered; the existing splinter
  check only fires with ≥ 2 submaps.

### 2.5 Persist job stdout per run `[x]`
- `JobManager` has no file sink: 6 of 7 unfinalized splatfacto-big runs left nothing to
  diagnose. Write `<run>/job.log` next to `pending.json`; keep it on failure. Also warn
  when `--steps-per-save` exceeds a few thousand (yesterday's 4K run lost 8,500 steps
  with `steps_per_save 10000`).

### 2.6 docker-compose: add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` `[x]`
- `PYTORCH_ALLOC_CONF` alone is inert in the NGC 25.11 torch build (allocator snapshot
  `is_expandable=False`); setting both enables it without the deprecation warning.
  PyTorch documents expandable segments for "a job changing allocation sizes
  frequently". 4K run: peak reserved 12.34 → 11.92 GiB; wall time 581 → 667 s on
  single runs (first run's reserve was pinned at a test cap, so inconclusive).

---

### 2.7 cuDNN autotuning off in `ns-train` (found while validating 2.1) `[x]`
- Isolated probe (torchmetrics LPIPS + pytorch_msssim SSIM + PSNR on a 1920×1080 pair):
  with `cudnn.benchmark=True` — what `nerfstudio/scripts/train.py:71` sets — the **first**
  LPIPS call costs +12.40 GiB (+3.15 GiB at 960×540); the second call, or any call with
  autotuning off, costs +0.50 GiB. SSIM +0.27 GiB, PSNR +0.05 GiB. `CUDNN_CONV_WSCAP_DBG`
  had no effect in this cuDNN build.
- Whole runs, autotuning off vs on: 1920 px / 1200 its peak **2.42 vs 11.15 GiB**, 25 s both;
  960 px / 3500 its peak **0.90 vs 3.72 GiB**, 34 s both; train-iteration 4.0 ms both.
- Image v4.8 patches the line to `… = os.environ.get("NS_CUDNN_BENCHMARK", "0") == "1"`
  (default off, upstream behaviour one env var away); `smoke_test.sh` §20 asserts anchor
  and effective value. Verified through the real `ns-train` on v4.8: 2.42 GiB default,
  11.16 GiB with the override.

### Tier 2 validation (2026-10-06)
- 20 unit tests for `gui/app/train_options.py` (`python -m unittest discover -s tests`).
- Train form instantiated offscreen against three real datasets: option labels, pose
  pills, cache estimate, resolved values, preset round-trip.
- `JobManager` log sink verified end to end (header, both streams, exit footer).
- GUI-generated commands run on the GPU: `--downscale-factor 2` via `nerfstudio-data`,
  `colmap` and `splatfactow_dataparser` (no interactive prompt); `camera-res-scale-factor
  0.5` on a 0-downscale dataset; auto CPU cache at native 4K; `antialiased`; the exact
  *Add downscale levels* command (16 frames → images_2/4/8 in 6 s).
- Container recreated with both allocator variables (`smoke_test.sh` §19 reads
  `is_expandable=True`); image rebuilt to v4.8 (`smoke_test.sh` 83/83). Rollback tags:
  `splat-nerfstudio-v2:pre-v4.7`, `:pre-v4.8`.
- All existing 50 warehouse manifests still validate against the extended schema.

## Tier 2 — implementation plan (written 2026-10-05 before coding)

**Integration constraints found in the code**
- `TrainParams` / `ProcessParams` use `extra="forbid"`; `manifest_schema.json` is generated by `python -m app.manifest`. New knobs are added as *optional fields with defaults* under schema 1.0 (additive: every existing manifest still validates; `load_manifest` keeps accepting 1.0). `ProcessManifest.metrics` is a free `dict[str, float]`, so `registered_frame_count` needs no schema change.
- `JobSpec` has no log sink; `JobManager._drain_stdout` is the single place every line passes through. A `log_path` on the spec, opened in `start()` and closed in every terminal handler, persists stdout without touching the views' signal wiring.
- The Train tab builds `argv` in `_Form._on_run`; dataparser flags must follow the subcommand (`nerfstudio-data` / `colmap` / `splatfactow_dataparser` all accept `--downscale-factor`). `FullImageDatamanagerConfig` and `SplatfactoWDatamanagerConfig` both expose `camera_res_scale_factor` and `cache_images`.
- Theme vocabulary to reuse: `card()`, `section_title()`, `labeled(text, help_title, help_body)`, `pill()/set_pill()` with kinds `neutral | success | warning | error`, button roles `primary | secondary | ghost`, label roles `caption | readout`, `QFrame[role="subcard"]`.
- Logic that decides options and argv goes into a Qt-free module (`gui/app/train_options.py`) so it can be unit-tested with `python -m unittest` (new `gui/tests/`).

**2.1 Eval cadence** — new Advanced controls *Eval image every N steps* (default **1000**; nerfstudio's 100 is the measured VRAM spike) and *Eval all images every N steps* (default 1000, nerfstudio's own). Always emitted as `--steps-per-eval-image` / `--steps-per-eval-all-images`; recorded in `TrainParams`.

**2.2 Training resolution** — a *Training resolution* dropdown in the PARAMETERS card, populated from the dataset: *Auto* (what nerfstudio would pick, shown with its pixel size), *Full*, and one entry per existing `images_<N>` folder (½ · ¼ · ⅛ with pixel sizes). Datasets processed with 0 downscale levels get the same ½ / ¼ / ⅛ entries implemented through `--pipeline.datamanager.camera-res-scale-factor` (labelled "rescaled at load"), plus an **Add downscale levels** action that runs nerfstudio's own `downscale_images()` in the container as a job and refreshes the list. Emitted as `<dataparser> --downscale-factor N` when a folder exists. Advanced: *Rasterize mode* (`classic` default, `antialiased` with the PLY-compatibility caveat in its help text). Both recorded in `TrainParams`.

**2.3 Image cache** — Advanced control *Image cache*: Auto / GPU / CPU. Auto estimates frames × W × H × 3 bytes at the chosen training resolution and picks CPU above 3 GiB (the failed 4K runs carried 6.2 GiB); the estimate is shown live under the control. Emitted as `--pipeline.datamanager.cache-images`; recorded in `TrainParams`.

**2.4 Pose-recovery gate** — `collect_process_metrics` records `registered_frame_count` from `transforms.json`. Process tab shows *Poses recovered: N of M (x %)* after a run, with a warning panel below 50 %. Train tab's DATASET card shows the same figure (falls back to counting `transforms.json` frames for manifests written elsewhere, e.g. vizer) with a success / warning / error pill, and asks for confirmation before training below 50 %.

**2.5 Job logs** — `JobSpec.log_path`; Process, Train, Export and Render jobs write `job.log` into their ULID directory (header with the command and start time, every line, footer with exit status). The *Steps per save* help text gains the measured cost of a late save; values above 5000 are flagged inline.

**2.6 Allocator** — `docker-compose.yml` sets `PYTORCH_CUDA_ALLOC_CONF` alongside `PYTORCH_ALLOC_CONF`; the Dashboard's copyable `docker create` and the USER_GUIDE flag table follow; `smoke_test.sh` gains an allocator probe (`torch.cuda.memory._snapshot()` → `is_expandable`).

**Validation** — unit tests for the pure module; `py_compile` + offscreen import of every edited view; the exact GUI-generated commands run on real datasets (½ resolution, Auto cache, eval 1000; a 0-downscale dataset through the rescale path; *Add downscale levels* on a scratch copy); container recreated for the env change; `smoke_test.sh` all green.

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
| eval every 1000 instead of 100 | buddha_v2 @1920, 1200 its | peak still 11.15 GiB (spike = first eval, not step 100) |
| LPIPS first call, cudnn.benchmark on / off | 1920×1080 pair | +12.40 GiB / +0.50 GiB (960×540: +3.15 / +0.12) |
| ns-train with cuDNN autotuning off (v4.8) | buddha_v2 @1920 1200 its · @960 3500 its | peak 2.42 GiB · 0.90 GiB, same wall time; override restores 11.16 GiB |
| GUI Tier-2 commands | 6 runs (see Tier 2 validation) | all exit 0 |

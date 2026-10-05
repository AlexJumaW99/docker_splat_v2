<div align="center">

# User Guide

### Complete walkthrough for `docker_splat_v2`

From host setup to your first exported Gaussian Splat PLY.

---

</div>

## Table of contents

1. [Host prerequisites](#host-prerequisites)
2. [Building the image](#building-the-image)
3. [Container lifecycle](#container-lifecycle)
4. [Desktop GUI (optional)](#desktop-gui-optional)
5. [Recording videos for splatfacto](#recording-videos-for-splatfacto)
6. [Processing a video](#processing-a-video)
7. [Processing images (instead of video)](#processing-images-instead-of-video)
8. [Training splatfacto](#training-splatfacto)
9. [Training on a pre-processed dataset](#training-on-a-pre-processed-dataset)
10. [Viewing in viser](#viewing-in-viser)
11. [Real-time monitoring with TensorBoard](#real-time-monitoring-with-tensorboard)
12. [Exporting a PLY file](#exporting-a-ply-file)
13. [Rendering a camera-path video](#rendering-a-camera-path-video)
14. [Troubleshooting](#troubleshooting)
15. [Appendix: full command reference](#appendix-full-command-reference)

---

## Host prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | Any NVIDIA Blackwell (sm_120) — RTX 5070 Ti, 5080, 5090 | RTX 5080 or 5090 for larger scenes |
| **VRAM** | 12 GB | 16 GB+ |
| **System RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB free for image + datasets | 200 GB+ for serious work |

### Software

**NVIDIA driver 580.126.18 or later** (required for CUDA 13.0 support):

```bash
nvidia-smi | head -3
```

Expected output should show `Driver Version: 580.x` or newer.

**Docker 28+ from apt (NOT snap):**

```bash
which docker && docker --version
```

Expected: path `/usr/bin/docker` and version `Docker version 28.x.x`. If you see `/snap/bin/docker`, uninstall the snap version and install via apt — NVIDIA Container Toolkit doesn't play nicely with snap Docker.

**NVIDIA Container Toolkit** configured as Docker's default runtime:

```bash
nvidia-ctk --version
cat /etc/docker/daemon.json
```

The `daemon.json` should include an `nvidia` runtime entry. If missing:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**GPU passthrough smoke test:**

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Should show your RTX 50-series GPU. If this fails, fix it before proceeding — nothing in this repo will work without functional GPU passthrough.

---

## Building the image

### Full build (first time)

```bash
git clone https://github.com/AlexJumaW99/docker_splat_v2.git
cd docker_splat_v2
docker build --target nerfstudio -t splat-nerfstudio-v2 .
```

**Expected duration:** 25–40 minutes on first run. Breakdown:

| Stage | Time | Cacheable |
|---|---|---|
| Stage 1–2 (apt deps) | 3–5 min | Yes (until apt list changes) |
| Stage 3 (COLMAP compile) | 10–15 min | Yes (until `COLMAP_COMMIT` is bumped) |
| Stage 4 (gsplat compile) | 2–4 min | Yes |
| Stage 5 (nerfstudio pip install) | 5–10 min | Partially (pip caches wheels) |

### Incremental build — use `--target` for development

If you're iterating on the Dockerfile, building to an earlier stage is faster. Stages cache aggressively, so rebuilding downstream stages is typically sub-minute.

```bash
# Test just the system deps
docker build --target system-deps -t splat:sysdeps .

# Test up through COLMAP
docker build --target colmap -t splat:colmap .

# Full build (default target)
docker build --target nerfstudio -t splat-nerfstudio-v2 .
```

### Verify the built image

```bash
docker images | grep splat-nerfstudio-v2
```

Expected: one line showing the image with size around **26 GB**. The size is large because NGC PyTorch is ~20 GB on its own; COLMAP, nerfstudio, and dependencies add the rest.

---

## Container lifecycle

> **Shortcut.** A [`Makefile`](Makefile) at the repo root wraps the commands in
> this section: `make up` / `make shell` / `make down` for daily use, and
> `make fresh` for a full rebuild-from-scratch (down → build → up → smoke
> test) if you lose the image or container. `make help` lists everything.
> The raw `docker` / `docker compose` commands below remain the ground
> truth — the Makefile just saves keystrokes.

### Create the persistent container (once)

```bash
mkdir -p ~/docker-stacks/docker_splat_v2/data \
         ~/docker-stacks/docker_splat_v2/data_v2

docker create \
  --name splat-ns-v2 \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/docker-stacks/docker_splat_v2/data:/workspace \
  -v ~/docker-stacks/docker_splat_v2/data_v2:/warehouse \
  -p 7007:7007 \
  -p 6006:6006 \
  -it \
  splat-nerfstudio-v2
```

**What each flag does:**

| Flag | Purpose |
|---|---|
| `--name splat-ns-v2` | Names the container for easy reference |
| `--gpus all` | GPU passthrough |
| `--ipc=host` | Shared memory for PyTorch multi-worker dataloaders |
| `--ulimit memlock=-1 --ulimit stack=67108864` | Required by NGC PyTorch for proper memory management |
| `--shm-size=16g` | Shared memory size for large tensors |
| `-e PYTORCH_ALLOC_CONF=expandable_segments:True` | Memory allocator config; reduces fragmentation on long training runs |
| `-v ~/docker-stacks/docker_splat_v2/data:/workspace` | Legacy artifact tree — still used by raw CLI workflows in this guide |
| `-v ~/docker-stacks/docker_splat_v2/data_v2:/warehouse` | Manifest-driven, ULID-keyed warehouse — where the GUI writes new sources, processed datasets, and training runs (see [Desktop GUI: lineage & manifests](#lineage--manifests)) |
| `-p 7007:7007` | Viser web viewer port |
| `-p 6006:6006` | TensorBoard web UI port |
| `-it` | Interactive TTY |

### Daily use

```bash
# Start a previously created container
docker start splat-ns-v2

# Attach a shell
docker exec -it splat-ns-v2 bash

# Exit the shell (container keeps running)
exit

# Stop the container (state preserved)
docker stop splat-ns-v2
```

### When you need a clean slate

```bash
docker stop splat-ns-v2
docker rm splat-ns-v2
# then re-run the `docker create` block above
```

Bind-mounted data at `~/docker-stacks/docker_splat_v2/data` is unaffected — only the container's writable layer is destroyed. Note: this will invalidate the gsplat JIT kernel cache at `/root/.cache/torch_extensions/`, so the first training in the new container will pause 60-120 seconds to recompile kernels.

### File ownership housekeeping

The container runs as root, so files it writes to `/workspace/` will be root-owned on the host. Every so often:

```bash
sudo chown -R $USER:$USER ~/docker-stacks/docker_splat_v2/data
```

---

## Desktop GUI (optional)

A PySide6 app in [`gui/`](gui/) wraps the same `ns-process-data` / `ns-train` commands you'd run by hand. It's a thin, read-mostly front end over `docker exec` — never builds, creates, or removes containers.

### Launch

```bash
cd gui
./run.sh
```

First launch creates `.venv/` and installs PySide6, pydantic, and pillow-heif (~90 MB). Subsequent launches are instant.

### What each tab does

| Tab       | State | Behavior                                                                                                                                      |
|-----------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Dashboard | live  | Polls Docker + `splat-ns-v2` + GPU. If the container is missing, shows the `docker build` / `docker create` commands with copy buttons — never runs them. |
| Process   | live  | Form-driven `ns-process-data` (video or images). Per-project presets, advanced raw-flag passthrough, live ffprobe readout for video, live image count for image folders. Auto-converts HEIC/HEIF photos (iPhone) to maximum-quality JPEG with EXIF preserved before processing; originals are stashed in a `heic_originals/` subfolder. |
| Train     | live  | Form-driven `ns-train`. Method picker (`splatfacto`, `splatfacto-big`, `splatfacto-mcmc`, `splatfacto-w`, `splatfacto-w-light`), iteration slider, viser + TensorBoard toggle. Detects when viser goes live and deep-links `http://localhost:7007`. |
| Exports   | live  | Form-driven PLY export and `ns-render camera-path` (video from viser keyframe JSON). Each artifact lands in its own ULID directory — `data_v2/exports/splat/<EXPORT_ULID>/<name>.ply` and `data_v2/renders/<RENDER_ULID>/<name>.mp4` — alongside a `manifest.json` whose `parent.run_id` is the train ULID; collisions are impossible by construction. Auto-routes to `ns-export-splatfactow` for `splatfacto-w` / `splatfacto-w-light` runs (upstream `ns-export gaussian-splat` doesn't handle them) and to `ns-export gaussian-splat` for everything else. |
| Viewers   | live  | Live port-status cards for viser (port 7007) and TensorBoard (port 6006). Run picker scoped to the active project — launch `ns-viewer` against a finished checkpoint, start/stop TensorBoard, and open either in the browser. Shows per-run TensorBoard-event availability. |
| Runs      | stub  | Placeholder for a training-run browser over `data/outputs/`.                                                                                  |

### Projects

A project bundles a dataset with its processing/training runs. **+ New project** offers three modes:

- **From video** — point at a video file inside `~/docker-stacks/docker_splat_v2/data/`. Goes Process → Train like the CLI workflow.
- **From images** — point at a folder of photos inside `~/docker-stacks/docker_splat_v2/data/input_images/`. Supports JPEG, PNG, TIFF, CR2, and HEIC/HEIF (auto-converted to JPEG). Goes Process → Train.
- **Import processed dataset** — point at an already-processed folder inside `~/docker-stacks/docker_splat_v2/data/`. The app auto-detects the layout:
  - `transforms.json` at the root → Nerfstudio dataparser
  - `sparse/0/cameras.bin` or `colmap/sparse/0/cameras.bin` → COLMAP dataparser, with the relative path saved so `ns-train` gets `--colmap-path <rel>` automatically

  Imported projects skip the Process tab and land straight on Train. This is the right flow for Tanks & Temples, MipNeRF360, or anything else already reconstructed with COLMAP.

State persists to `~/.config/docker_splat_v2_gui/projects.json`. Delete that directory to reset.

### Lineage & manifests

Anything you produce through the GUI's Process or Train tab lands in
`~/docker-stacks/docker_splat_v2/data_v2/` (the warehouse), bind-mounted into
the container at `/warehouse/`. Every run is its own ULID-keyed directory
with a `manifest.json` declaring how it was produced and which upstream
run it descends from:

```
data_v2/
├── sources/<source_ULID>/                          # original video or image folder
│   └── manifest.json                               # SourceManifest, parent = null (root)
├── processed/<process_ULID>/                       # ns-process-data output
│   └── manifest.json                               # ProcessManifest, parent.run_id = <source_ULID>
├── trained/<train_ULID>/                           # ns-train output (nested per nerfstudio)
│   ├── <project>/<method>/<timestamp>/             # config.yml, ckpts, tfevents
│   └── manifest.json                               # TrainManifest, parent.run_id = <process_ULID>
├── exports/<kind>/<export_ULID>/                   # ns-export output (kind: splat | pcd | mesh)
│   ├── <name>.ply                                  # primary payload (renamed from ns-export's `splat.ply`)
│   └── manifest.json                               # ExportManifest, parent.run_id = <train_ULID>
└── renders/<render_ULID>/                          # ns-render camera-path output
    ├── <name>.mp4                                  # rendered video
    ├── camera_path.json                            # snapshot of the keyframe path used (for reproducibility)
    └── manifest.json                               # RenderManifest, parent.run_id = <train_ULID>
```

Lineage is **declared**, not inferred — to trace a `.ply` back to its
source video, read `manifest.json` for the export, follow `parent.run_id`
into `data_v2/trained/`, then into `data_v2/processed/`, then into
`data_v2/sources/`. Four file reads, no tooling required (the same walk
works for an `.mp4` via `data_v2/renders/`). The full JSON Schema for all
five manifest stages is at [`gui/app/manifest_schema.json`](gui/app/manifest_schema.json)
(Draft 2020-12, frozen at v1.0).

**The `mesh/` and `pointcloud/` export sub-folders exist by convention
but are not wired today.** `ns-export poisson`, `tsdf`, `marching-cubes`,
and `pointcloud` all require the volumetric-NeRF model interface
(`pipeline.model(ray_bundle)` returning depth-along-rays, or an SDF
field), which `SplatfactoModel` doesn't implement — see
[`nerfstudio/scripts/exporter.py`](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/exporter.py).
Mesh from a Gaussian scene currently requires retraining with 2DGS
(unlocked by the gsplat 1.5.3 upgrade) and using the 2DGS repo's TSDF
extractor, or a separate tool such as SuGaR / GS2Mesh.

**Incomplete runs are first-class.** When the Train tab starts a run it
writes a `pending.json` sidecar *before* `ns-train` launches. If the GUI
is closed, the project switched, or the run is cancelled before
completion, the sidecar stays on disk as a breadcrumb. The Train tab also
watches `ns-train`'s stdout for the `🎉 Training Finished 🎉` banner so it
can finalize the manifest *the moment training ends*, not when the viewer
process eventually exits — important because the default
`--viewer.quit-on-train-completion False` keeps `ns-train` pinned alive
after the last iteration.

**Recovering an orphan run.** If you ever find a `data_v2/trained/<ULID>/`
directory with `pending.json` but no `manifest.json` (e.g. you killed the
GUI while training was still alive), reconstruct the manifest from the
on-disk `config.yml` and pending sidecar:

```bash
cd gui
.venv/bin/python -m app.warehouse backfill-train ../data_v2/trained/<ULID>
```

The CLI writes a `manifest.json` with `provenance: "backfilled"` and
removes the pending sidecar. The lineage chain closes; the run becomes
indistinguishable from a live-finalized one for any downstream consumer
walking the warehouse.

---

## Recording videos for splatfacto

The quality of your splat is 80% determined by the quality of your input video. Five rules:

### 1. Pick the right subject

**Good:** textured objects with surface detail — plants, books, toys, ceramic items, small sculptures, bikes, musical instruments, figurines.

**Bad:** transparent glass, mirrors, perfectly shiny metal, plain white/black surfaces, objects without texture.

### 2. Orbit, don't pan

Walk **around** the subject in a full 360° circle at a constant distance. Keep the subject roughly centered in frame throughout. Do not zoom.

A second orbit from a different height (e.g., 20° higher) significantly improves coverage.

### 3. Move slowly

Target ~30 seconds per full orbit. Moving too fast introduces motion blur, which destroys COLMAP feature matching.

### 4. Control lighting

Bright, diffuse, consistent. Avoid:

- Harsh directional sunlight that creates moving shadows as you orbit
- Dim lighting that causes your camera's ISO to spike (noise destroys features)
- Auto-exposure drift as you rotate through different light angles

### 5. Static scene

Nothing should move during capture. No pets, no people walking through, no wind blowing leaves. Even small motion produces "floaters" in the final splat.

### Target specs

| Setting | Recommended |
|---|---|
| **Resolution** | 1080p or higher |
| **Framerate** | 30 fps |
| **Duration** | 30–90 seconds |
| **Format** | MP4/H.264 (phone default) |
| **File size** | < 500 MB |

Phone video straight out of the camera is fine. No editing needed.

---

## Processing a video

Place your video at `~/docker-stacks/docker_splat_v2/data/yourvideo.mp4` on the host. This becomes `/workspace/yourvideo.mp4` inside the container.

Inside the container:

```bash
ns-process-data video \
  --data /workspace/yourvideo.mp4 \
  --output-dir /workspace/processed/yourvideo_splat \
  --num-frames-target 250 \
  --matching-method sequential \
  --num-downscales 3 \
  --verbose
```

### Tuning `--num-frames-target`

Rule of thumb:

| Video length | Target frames | Effective fps |
|---|---|---|
| 15–20 s | 100–150 | ~7 fps |
| 30–45 s | 200 | ~5 fps |
| 60–90 s | 250–300 | ~3–5 fps |
| 90–120 s | 300–400 | ~3 fps |

More is not better — over ~400 frames, COLMAP bundle adjustment slows down significantly with diminishing quality returns.

### Expected timeline

On an RTX 5070 Ti with a 30-second 1080p video at 250 frames:

| Phase | Time |
|---|---|
| ffmpeg frame extraction | 30–60 s |
| COLMAP feature extraction (GPU) | 1–3 min |
| COLMAP sequential matching | 2–5 min |
| COLMAP sparse reconstruction (CPU-bound) | 5–15 min |
| Intrinsics refinement | 1–3 min |
| Downscaled image generation | 30–60 s |
| **Total** | **~10–25 min** |

### Output structure

```
/workspace/processed/yourvideo_splat/
├── images/              # Full-resolution frames (N images)
├── images_2/            # Half resolution
├── images_4/            # Quarter resolution
├── images_8/            # 1/8 resolution
├── colmap/
│   ├── sparse/0/        # COLMAP sparse model that ns-process-data exported from
│   ├── sparse/1/        # Present iff COLMAP splintered; preserved for inspection
│   └── snapshots/       # Incremental mapper snapshots (viser-watchable)
├── sparse_pc.ply        # Sparse point cloud (seed for splatfacto)
└── transforms.json      # Nerfstudio camera poses (the key file)
```

> On longer videos, COLMAP 4.1's `incremental_pipeline` sometimes seeds a 2-image bootstrap as `sparse/0` and grows the real model into `sparse/1`. The image ships a `colmap` PATH-shim (`/opt/colmap-shim/colmap`) that detects this after the mapper exits, probes each numeric submap with `colmap model_analyzer`, and name-swaps the largest into `sparse/0` (the original `sparse/0` is preserved under its old index). The result: `ns-process-data` always reads the largest reconstruction, never the bootstrap stub. You'll see `colmap-shim: promoting sparse/N (M registered images) -> sparse/0` on stderr when promotion happens. The shim is transparent and only touches `colmap mapper` — every other subcommand passes straight through.

### Success criteria

At the end, you should see:

```
🎉 🎉 🎉 All DONE 🎉 🎉 🎉
Starting with N video frames
We extracted M images with prefix 'frame_'
Colmap matched K images
COLMAP found poses for XX.X% of the images.
```

**Pose recovery percentage interpretation:**

| Result | Meaning |
|---|---|
| > 95% | Excellent. Proceed with confidence. |
| 85–95% | Good. Some unregistered frames are normal. |
| 70–85% | Marginal. Consider reshooting with better lighting or slower motion. |
| < 70% | Poor. Your video likely has motion blur, poor texture, or scene dynamics. Reshoot. |

---

## Processing images (instead of video)

You can also process a folder of still photos instead of a video. Place your images in a subfolder under `~/docker-stacks/docker_splat_v2/data/input_images/` on the host. This becomes `/workspace/input_images/<folder>` inside the container.

### iPhone HEIC photos

Modern iPhones (and some Android phones) save photos in HEIC/HEIF format. The GUI's Process tab auto-converts HEIC/HEIF files to maximum-quality JPEG (quality 100, chroma subsampling 4:4:4) with full EXIF metadata preserved — including focal length and camera model, which COLMAP needs for accurate intrinsics. After conversion, originals are moved to a `heic_originals/` subfolder so the input folder stays clean.

If processing via the CLI, convert HEIC files to JPEG beforehand (e.g. with `heif-convert` or any tool that preserves EXIF).

### CLI recipe

```bash
ns-process-data images \
  --data /workspace/input_images/your_scene \
  --output-dir /workspace/processed/your_scene_splat \
  --matching-method exhaustive \
  --num-downscales 1 \
  --verbose
```

### Matching method guidance for images

| Method | Best for | Notes |
|---|---|---|
| `exhaustive` | < 100 images | Checks all N*(N-1)/2 pairs. Gold standard for small sets. |
| `vocab_tree` | 100–500 images | Vocabulary-tree approximation. Faster but can miss matches on textureless or repetitive scenes. |
| `sequential` | Ordered sequences (e.g. video frames) | Only matches nearby frames. Not suitable for unordered photo sets. |

For still photos, **use `exhaustive` unless you have more than ~100 images.** With 31 photos, exhaustive takes seconds; vocab_tree can miss critical matches in scenes with repetitive geometry.

### Learned features with hloc (`--sfm-tool hloc`)

Image v4.7 bakes the [hloc](https://github.com/cvg/Hierarchical-Localization) toolbox into the container, so COLMAP's SIFT can be swapped for learned features when SIFT struggles (long focal lengths, low texture, strong exposure changes). In the GUI, put the flags in the Process tab's **Raw CLI flags** box; on the CLI, append them to `ns-process-data`:

```bash
ns-process-data images --data /workspace/photos --output-dir /workspace/processed/photos_splat \
  --matching-method exhaustive \
  --sfm-tool hloc --feature-type superpoint_max --matcher-type superpoint+lightglue
```

What works in this image (verified 2026-10-05): feature types `sift`, `superpoint_aachen`, `superpoint_max`, `superpoint_inloc`, `sosnet`, `disk`; matcher types `superglue`, `superglue-fast`, `NN-superpoint`, `NN-ratio`, `NN-mutual`, `superpoint+lightglue`, `disk+lightglue`. `r2d2`, `d2net-ss` and `adalam` are accepted by the CLI but fail to import here (hloc submodules not checked out; kornia API), so don't use them.

Things to know:

- Under hloc, `--matching-method exhaustive` means all pairs; `sequential` **and** `vocab_tree` both mean NetVLAD image retrieval of the 50 most similar images per frame (nerfstudio's `hloc_utils.py`). The NetVLAD weights are downloaded on the first such run.
- The crop and circle-mask options are rejected with hloc (`Cannot use a mask with hloc`).
- hloc runs SfM through its own `pycolmap 3.13.0`, not the system COLMAP binary, so the submap-promotion shim does not apply (hloc keeps the largest model itself). The camera model follows `--camera-type` as usual (`OPENCV` by default); focal length and distortion are refined but the principal point stays at the image centre (verified: `cx, cy = 480.0, 270.0` on a 960×540 run).
- Licence: the SuperPoint / SuperGlue weights (`superpoint_*`, `superglue*`, `superpoint+lightglue`) are released by Magic Leap for academic or non-profit non-commercial research use only. `disk+lightglue` (DISK by EPFL, LightGlue by ETH, both Apache-2.0) is the permissively licensed combination.

### Tips for good photo captures

The same principles as video apply — see [Recording videos for splatfacto](#recording-videos-for-splatfacto) — with these additions:

1. **60–80% overlap between adjacent shots.** Each photo should share most of its frame with its neighbors.
2. **Avoid textureless close-ups.** Shots of flat white walls, ceiling tiles, or uniform surfaces give COLMAP nothing to match on.
3. **Keep the camera at a consistent height** and pointed roughly horizontal.
4. **50–100 photos** for a room-sized scene. More overlap is always better than more coverage.

---

## Training splatfacto

### Basic command (viser only)

```bash
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False
```

### With live TensorBoard metrics (recommended)

Add `--vis viewer+tensorboard` to enable simultaneous 3D preview and metric logging:

```bash
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False
```

See the [Real-time monitoring section](#real-time-monitoring-with-tensorboard) for launching TensorBoard alongside training.

### Choosing `--max-num-iterations`

| Iterations | Time (RTX 5070 Ti) | Use case |
|---|---|---|
| 2000 | ~3–5 min | Quick "does it work" check |
| 7000 | ~10–15 min | Good-quality preview |
| 30000 (default) | ~45–60 min | Publication quality |
| 50000+ | > 1 hr | Diminishing returns |

### First run: gsplat JIT compilation

On the **first** ns-train invocation inside a container, gsplat JIT-compiles its CUDA kernels for sm_120. This takes 60–120 seconds during which the terminal looks stuck. Do not Ctrl+C.

Subsequent runs are instant — the compiled kernels cache at `/root/.cache/torch_extensions/` and persist across `docker stop`/`docker start`, but are lost if you `docker rm` and recreate the container.

### What healthy training output looks like

```
Step (% Done)
--------------------
1000 (14.29%)   12.4 ms    1 min 15 s    14.2 M
2000 (28.57%)   11.8 ms    53 s          28.6 M
3000 (42.86%)   11.5 ms    39 s          45.1 M
...
```

Columns: step, iteration time, remaining time estimate, gaussian count. The gaussian count should grow steadily — doubling roughly every 2000 iterations in the densification phase.

### Splatfacto variants

| Method | When to use |
|---|---|
| `splatfacto` | Default — balanced quality and speed. Good starting point for most videos. |
| `splatfacto-big` | Larger gaussian capacity. Use for detailed scenes where the default caps out. Slower and hungrier on VRAM. |
| `splatfacto-mcmc` | MCMC-based densification. More stable gaussian counts; sometimes cleaner results on sparse-view scenes. Requires the git-pinned nerfstudio from Stage 5. |
| `splatfacto-w` | ["In the wild" splatfacto](https://github.com/KevinXu02/splatfacto-w) — per-image appearance embedding, a learned background model and transient masking (from step 6000). Good for outdoor or crowd-tourism-style datasets. Needs the dataset's **COLMAP sparse model** with **perspective cameras** (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV; fisheye is refused). The image ships a v2 `nerfw_dataparser.py` overlay (real image size + distortion, auto-picks `images_<N>` so the longest side is ≤ 1600 px, honours `--downscale-factor`) and a colour-cache patch for an upstream eval crash. Upstream's schedule assumes 55,000 iterations; eval metrics are computed on the right half of held-out images (left half is trained on), so they are not comparable with the other methods. Heavier than -light: 9.5 GiB peak at 1920×1080 in testing. |
| `splatfacto-w-light` | Lighter-weight variant of `splatfacto-w` (smaller appearance MLP, no background model, no transient masking). Faster and less VRAM-hungry. Uses nerfstudio's standard `nerfstudio-data` / `colmap` dataparsers exactly like the built-in methods, so it runs on any processed dataset and its eval split is a true hold-out. The upstream author recommends it for non-phototourism captures. |

Swap the first positional argument of `ns-train` to pick one (e.g. `ns-train splatfacto-big --data …`).

### Final output location

When training completes from the **raw CLI** with `--output-dir /workspace/outputs`, your model lands at:

```
/workspace/outputs/<dataset_name>/splatfacto/<YYYY-MM-DD_HHMMSS>/
├── config.yml                          # Training config (needed for ns-viewer and ns-export)
├── events.out.tfevents.*               # TensorBoard logs (if --vis included tensorboard)
└── nerfstudio_models/
    └── step-NNNNNNNNN.ckpt             # Model checkpoint
```

Note the timestamp directory — you'll reference it for viewing and exporting.

When training is started from the **GUI Train tab**, output goes under the warehouse instead:

```
/warehouse/trained/<train_ULID>/
├── manifest.json                        # TrainManifest, parent.run_id → ProcessManifest
└── <project>/<method>/<YYYY-MM-DD_HHMMSS>/
    ├── config.yml
    ├── events.out.tfevents.*
    └── nerfstudio_models/step-NNNNNNNNN.ckpt
```

Both layouts are functional — `ns-viewer` and `ns-export` only need the path to `config.yml`. See [Lineage & manifests](#lineage--manifests) for why the GUI nests an extra ULID level.

---

## Training on a pre-processed dataset

If you already have a reconstructed scene (Tanks & Temples, MipNeRF360, your own prior `ns-process-data` output), you can skip processing entirely and train on it directly.

### Place the dataset under the bind mount

The container only sees `~/docker-stacks/docker_splat_v2/data/` (it's mounted at `/workspace`). So move or symlink the dataset there:

```bash
mv /path/to/Ballroom ~/docker-stacks/docker_splat_v2/data/processed/ballroom_splat
```

The `_splat` suffix is convention — not required, but matches what `ns-process-data` produces.

### Pick the right dataparser

Nerfstudio's `ns-train` takes a dataparser as a tyro subcommand. Which one depends on the layout of your folder:

| Layout present | Dataparser | Notes |
|---|---|---|
| `transforms.json` at the root | `nerfstudio-data` (default) | What `ns-process-data` writes. No extra flags needed. |
| `sparse/0/cameras.bin` | `colmap` with `--colmap-path sparse/0` | Plain COLMAP layout (Tanks & Temples, hand-run COLMAP). |
| `colmap/sparse/0/cameras.bin` | `colmap` (defaults work) | Nerfstudio's own COLMAP layout. `ColmapDataParserConfig.colmap_path` defaults to `colmap/sparse/0`. |

### CLI recipe — plain COLMAP folder

For a Tanks & Temples-style folder (`sparse/0/...` at the root):

```bash
ns-train splatfacto \
  --data /workspace/processed/ballroom_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False \
  colmap \
  --colmap-path sparse/0
```

The `colmap` subcommand and its `--colmap-path` flag must come **after** all the `ns-train` method flags. Argument order matters to tyro.

### CLI recipe — splatfacto-w with a COLMAP folder

`splatfacto-w` uses its own `splatfactow_dataparser` (not the built-in `colmap` dataparser), so the subcommand name changes:

```bash
ns-train splatfacto-w \
  --data /workspace/processed/ballroom_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False \
  splatfactow_dataparser \
  --colmap-path colmap/sparse/0
```

Note the subcommand is `splatfactow_dataparser` (not `colmap`). The `--colmap-path` flag is the same — a relative path under `--data` pointing at the COLMAP sparse model directory. The GUI's Train tab handles this automatically and refuses to start `splatfacto-w` on a project that has no COLMAP sparse model. `splatfacto-w-light` does **not** need this — it uses the standard nerfstudio dataparser, exactly like `splatfacto`.

What the v2 `splatfactow_dataparser` accepts and does (image v4.7 and later):

- COLMAP camera models `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`. Distortion is undistorted at load time, as with `nerfstudio-data`. Fisheye models are rejected with an explicit error.
- Image resolution follows the same rule as `nerfstudio-data`: the smallest existing `images_<N>` folder that brings the longest side to ≤ 1600 px, or an explicit `--downscale-factor N` after the subcommand (`splatfactow_dataparser --colmap-path colmap/sparse/0 --downscale-factor 2`). A dataset processed with **Downscale levels = 0** therefore trains at full resolution; at 4K that is far beyond 16 GB for splatfacto-w, so add downscale levels first.
- With no `<data_name>.tsv` split file, a deterministic 10 % eval split is drawn; those images still train on their left half (upstream NeRF-W protocol).

### Via the GUI

Use **+ New project → Import processed dataset** and point at the folder. The app auto-detects the format, records the COLMAP path on the project, and emits the right `ns-train` invocation at train time — no need to remember the subcommand ordering.

---

## Viewing in viser

### Live view during training

As soon as training starts, viser prints a line like:

```
Viewer running locally at: http://localhost:7007 (listening on 0.0.0.0)
```

Open that URL in any browser on your host. The splat renders live as training progresses.

**Viewer controls:**

| Action | Control |
|---|---|
| Orbit | Left-click + drag |
| Pan | Right-click + drag (or middle-click) |
| Zoom | Scroll wheel |
| Reset view | `R` key |
| Wireframe toggle | In the sidebar |

### Important: the `quit-on-train-completion` flag

When training finishes, viser's behavior depends on this flag:

| Flag value | Behavior |
|---|---|
| `True` | Viser shuts down immediately when training ends; localhost:7007 returns "site can't be reached" |
| `False` | Viser stays alive after training; you can navigate the finished scene until you Ctrl+C the training process |

**Recommendation:** use `--viewer.quit-on-train-completion False` for interactive sessions where you want to explore the result. Only use `True` for automated/scripted runs.

### Post-training view with ns-viewer

To re-open a completed model in a fresh session:

```bash
ns-viewer \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --viewer.websocket-host 0.0.0.0
```

---

## Real-time monitoring with TensorBoard

TensorBoard gives you live training metrics (loss, PSNR, SSIM, gaussian count) and eval rendered images in a browser dashboard. It runs as a separate process alongside `ns-train` and reads the event files that nerfstudio writes to disk.

### The two-shell workflow

You need two shells into the container: one for training, one for TensorBoard.

**Shell 1 (TensorBoard) — start this first:**

From a host terminal:

```bash
docker exec -it splat-ns-v2 bash
```

Inside the container:

```bash
tensorboard --logdir /workspace/outputs --bind_all --port 6006
```

Expected output (ignore the deprecation warning — it's harmless noise from the library's internals):

```
TensorBoard 2.20.0 at http://4dfa087b79d2:6006/ (Press CTRL+C to quit)
```

The URL in the output uses the container's internal hostname; ignore it. Open `http://localhost:6006` in your browser instead — Docker's port forwarding routes this to the container.

**Shell 2 (Training) — open in a new host terminal:**

```bash
docker exec -it splat-ns-v2 bash
```

Inside the container, kick off training with `--vis viewer+tensorboard`:

```bash
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False
```

### What to watch in each tab

**Browser tab 1: `http://localhost:7007` (viser)**
- Live 3D splat rendering as it trains
- Camera frustums showing training image positions
- Densification happening in real time

**Browser tab 2: `http://localhost:6006` (TensorBoard)**
- **Scalars tab:** loss curves, PSNR, SSIM, gaussian count over time
- **Images tab:** rendered eval images (appears at eval steps — default every 1000 iterations)
- **Histograms tab:** distribution of gaussian parameters (scales, opacities)

TensorBoard refreshes automatically as new event data is written. If it feels stale, click the circular arrow icon in the top-right to force a reload.

### Known tradeoff: viser may stutter during eval

The nerfstudio docs note that using the viewer alongside wandb or tensorboard may cause stuttering during evaluation steps. On an RTX 5070 Ti with a moderate scene, this typically manifests as 2–3 seconds of laggy viewport every ~1000 iterations when an eval pass runs. Training metrics are unaffected.

If the stutter bothers you more than the metrics help:

- Switch to `--vis tensorboard` alone and use `ns-viewer` on checkpoints for 3D preview
- Or switch to `--vis viewer` alone and post-analyze TensorBoard after training ends (event files still exist)

### Harmless warnings you'll see in TensorBoard's console

```
pkg_resources is deprecated as an API...
TensorFlow installation not found - running with reduced feature set.
NOTE: Using experimental fast data loading logic...
```

All three are noise from tensorboard's import chain. **We don't need or want TensorFlow** — nerfstudio uses PyTorch. TensorBoard's "reduced feature set" without TensorFlow still includes everything we care about (scalars, images, histograms). Ignore all three.

### After training ends

TensorBoard keeps running until you Ctrl+C it in Shell 1. Event files are persistent on disk, so you can re-launch TensorBoard anytime to review past runs:

```bash
tensorboard --logdir /workspace/outputs --bind_all --port 6006
```

Multiple training runs appear as separate entries in TensorBoard's left sidebar, making it easy to compare hyperparameter variations.

---

## Exporting a PLY file

Once trained, export to a portable PLY that can be opened in any Gaussian Splatting viewer or imported into Blender (with a plugin like [3DGS Render](https://github.com/ReshotAI/gaussian-splatting-blender-addon)).

### Via the GUI Exports tab (recommended)

Open the **Exports** tab, pick a project + run from the dropdown, and click **Export .ply**. The GUI mints an `EXPORT_ULID`, writes the PLY at `data_v2/exports/splat/<EXPORT_ULID>/<name>.ply` (defaulting `<name>` to `<display_name>_<method>_<timestamp>` so the file is still self-describing if copied out), and drops a `manifest.json` next to it whose `parent.run_id` is the train ULID — so every export carries full lineage back to the source video without filename parsing. Each export lives in its own ULID directory, so two clicks on the same run never overwrite each other. The right exporter is picked automatically based on the run's training method, so you never need to think about `splatfacto-w` vs. built-in splatfacto. The run picker is sourced from `data_v2/trained/*/manifest.json` filtered by the project's processed ULID; legacy projects (whose `processed_dir` predates the warehouse) see a gated empty state asking you to reprocess + retrain via the warehouse-aware tabs first.

### Via the CLI — built-in splatfacto / -big / -mcmc

```bash
ns-export gaussian-splat \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --output-dir /workspace/outputs/<dataset>/export
```

Output: `splat.ply` in the specified directory. File size is typically 100–500 MB for a 7000-iteration model with ~80M gaussians. Note: running the raw CLI twice into the same `--output-dir` overwrites `splat.ply`; the GUI's Exports tab avoids this.

### Via the CLI — `splatfacto-w` / `splatfacto-w-light`

Upstream `ns-export gaussian-splat` hard-asserts `isinstance(pipeline.model, SplatfactoModel)` and crashes on splatfacto-w runs (its model inherits from `Model` directly). The image ships a splatfacto-w-aware exporter that bakes a chosen appearance embedding into per-gaussian SH coefficients and writes the same inria-format PLY:

```bash
ns-export-splatfactow \
  --load-config /workspace/outputs/<dataset>/splatfacto-w-light/<timestamp>/config.yml \
  --output-dir /workspace/exports/splat
```

Key flag: `--appearance-mode` controls which appearance embedding is baked in.

| Value | Behavior |
|---|---|
| `avg` (default) | Mean of all training-camera embeddings. Most stable for arbitrary viewpoints. |
| `first` | Training-camera index 0. |
| `camera` | Training camera at `--camera-idx N`. Use when you want to reproduce a specific training image's lighting. |

Output: `splat.ply` in the specified directory; same 62-property layout (`x y z · nx ny nz · f_dc_{0..2} · f_rest_{0..44} · opacity · scale_{0..2} · rot_{0..3}`) as `ns-export gaussian-splat`, so downstream 3DGS viewers (SuperSplat, Blender 3DGS Render, etc.) open it identically.

### Fixing host-side ownership

The PLY will be written as root inside the container. To open it on the host:

```bash
sudo chown -R $USER:$USER ~/docker-stacks/docker_splat_v2/data/outputs
```

### Recommended post-training editor: SuperSplat

For cleaning up stray gaussians, cropping the scene, applying post-processing effects, or publishing to the web:

**[superspl.at/editor](https://superspl.at/editor)** — free, open-source, browser-based editor from PlayCanvas. Drag-and-drop your exported PLY to load it. Features: selection tools (rect/brush/lasso), cropping, timeline animations, WebXR export, and more.

---

## Rendering a camera-path video

Viser lets you record a camera path by placing keyframes along a flythrough of your trained scene, then exporting the path as a JSON file (lands under `processed/<project>/camera_paths/<timestamp>.json`). Turn that JSON into an MP4 with `ns-render`.

### Via the GUI Exports tab

Same tab as PLY export. Below the PLY card is a **Render camera-path video** card:

1. Pick the project + run (same dropdown as PLY export).
2. Pick a camera-path JSON from the project's `camera_paths/` folder (auto-populated).
3. Click **Render video**.

Output lands at `data_v2/renders/<RENDER_ULID>/<name>.mp4` alongside (a) a snapshot of the `camera_path.json` actually used — so the render stays reproducible even if the original keyframe file is later edited — and (b) a `manifest.json` whose `parent.run_id` is the train ULID. Each render lives in its own ULID directory, so collisions are impossible.

### Via the CLI

```bash
ns-render camera-path \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --camera-path-filename /workspace/processed/<project>/camera_paths/<timestamp>.json \
  --output-path /workspace/renders/<project>/<camera-path-stem>.mp4
```

Rendering time scales with video length and scene gaussian count. A 10-second 1080p path from an 80M-gaussian scene takes a few minutes on an RTX 5070 Ti.

---

## Troubleshooting

### "Could NOT find Qt6Svg" during build

Cause: COLMAP 4.1.0.dev0 requires `qt6-svg-dev`, which older documentation doesn't mention.

Fix: already in the v2+ Dockerfile. If you see this, you're building from an outdated source.

### `ns-process-data` fails with "unrecognised option '--SiftExtraction.use_gpu'"

Cause: COLMAP renamed this flag to `--FeatureExtraction.use_gpu`; nerfstudio 1.1.5 still uses the old name.

Fix: already patched in Stage 5 of the v3 Dockerfile. If you hit this, your image was built from a pre-v3 Dockerfile. Rebuild with `docker build --target nerfstudio -t splat-nerfstudio-v2 .`

### Training appears to hang for 1–2 minutes on first run

Cause: gsplat is JIT-compiling CUDA kernels for your sm_120 GPU.

Fix: wait. Do not Ctrl+C. Subsequent runs will be instant.

### "Failed to retrieve file url" when running `ns-download-data`

Cause: [long-running upstream nerfstudio bug](https://github.com/nerfstudio-project/nerfstudio/issues/3134) — Google Drive rate-limits gdown.

Fix: not a fix for this repo to make. Use your own video data.

### Browser can't connect to `localhost:7007` or `localhost:6006`

Checklist:

1. Container was created with both `-p 7007:7007` AND `-p 6006:6006` flags (verify with `docker port splat-ns-v2` on host)
2. For viser: training was started with `--viewer.websocket-host 0.0.0.0` (otherwise viser binds to 127.0.0.1 inside the container and Docker can't route to it)
3. For viser: training is currently running OR was started with `--viewer.quit-on-train-completion False` (viser dies when training ends if that flag is True)
4. For TensorBoard: `tensorboard` process is still running (Shell 1 wasn't closed or Ctrl+C'd)
5. For TensorBoard: command used `--bind_all` flag (otherwise binds to 127.0.0.1 only)
6. No other process on the host is using the port (`sudo lsof -i :7007` or `sudo lsof -i :6006`)
7. The container has been detached from its bridge network — symptoms: `docker port splat-ns-v2` prints nothing despite the container being `Up`, and `docker inspect splat-ns-v2 --format '{{.NetworkSettings.Networks}}'` returns `{}`. This can happen after an interrupted `docker compose down` or a docker-daemon hiccup; the container keeps running (and any in-progress `ns-train` keeps training, since it doesn't need the network) but every host-side connection is refused. Fix it without restarting:

   ```bash
   make heal-net
   # or, equivalently:
   docker network connect docker_splat_v2_default splat-ns-v2
   ```

   `docker-proxy` re-establishes itself on reconnect, so `localhost:7007` / `localhost:6006` come back online immediately and the running training is undisturbed.

### `docker build` fails on Stage 3 (COLMAP)

Most common causes:

- Out of disk space — COLMAP compile needs ~5 GB of build artifacts temporarily
- apt package missing from Stage 2 (if you modified the apt list)
- GitHub rate-limiting the `git clone` of colmap/colmap.git (retry in a few minutes)
- Invalid `COLMAP_COMMIT` SHA — Stage 3 errors at `git checkout` with `fatal: reference is not a tree`. Verify the SHA in [`Dockerfile`](Dockerfile) (`ARG COLMAP_COMMIT=...`) is reachable on `colmap/colmap` upstream

### CUDA "no kernel image is available for execution on the device"

Cause: you're running the image on a GPU that isn't Blackwell (sm_120). This image is compiled exclusively for `sm_120`.

Fix: rebuild with a different `TORCH_CUDA_ARCH_LIST` value in the Dockerfile:

| GPU family | Value |
|---|---|
| Ampere (RTX 30xx) | `8.6` |
| Ada (RTX 40xx) | `8.9` |
| Hopper (H100) | `9.0` |
| Blackwell (RTX 50xx) | `12.0` |

Multiple architectures can be built simultaneously by separating with semicolons (at the cost of image size and build time).

### `transforms.json` only contains 2 frames despite a long input video

Cause: COLMAP 4.1's `incremental_pipeline` splintered the reconstruction — it seeded a 2-image bootstrap as `sparse/0` and grew the real model into `sparse/1`, `sparse/2`, etc. Nerfstudio's converter hardcodes `colmap_model_path = Path("colmap/sparse/0")`, so it exports the bootstrap stub and leaves the real reconstruction unused on disk.

Fix: already handled by the `colmap-shim` baked into the image (`/opt/colmap-shim/colmap`, prepended to `PATH`). After the mapper exits, the shim probes every numeric submap with `colmap model_analyzer`, name-swaps the largest into `sparse/0`, and preserves the original `sparse/0` under its old index. You'll see `colmap-shim: promoting sparse/N (M registered images) -> sparse/0` on stderr when it kicks in.

If you still see a 2-frame `transforms.json`:

1. Confirm the shim is active: `docker exec splat-ns-v2 which colmap` should print `/opt/colmap-shim/colmap`. If it prints `/usr/local/bin/colmap`, your container was built from a pre-shim image — rebuild with `docker build --target nerfstudio -t splat-nerfstudio-v2 .` and recreate the container.
2. Confirm the shim wasn't disabled: unset `COLMAP_SHIM_PROMOTE_LARGEST` (or set it to `1`) and re-run.
3. Inspect `/workspace/processed/<run>/colmap/sparse/` — you should see multiple numeric subfolders. `colmap model_analyzer --path /workspace/processed/<run>/colmap/sparse/0` should report a large `Registered images` count.
4. As a last resort, rerun `ns-process-data` with `--skip-colmap --colmap-model-path sparse/<N>` pointing at the real submap.

### `splatfacto-w` fails with `FileNotFoundError: dense/sparse/cameras.bin`

Cause: you're running an image without the NerfW dataparser patch. Upstream `splatfactow/nerfw_dataparser.py` hardcodes phototourism-specific paths (`dense/sparse/`, `dense/images/`), asserts PINHOLE cameras only, and requires a TSV split file — none of which exist in a standard `ns-process-data` output.

Fix: already patched in the v4.4 Dockerfile. If you see this, your image was built from a pre-v4.4 Dockerfile. Rebuild with `docker build --target nerfstudio -t splat-nerfstudio-v2 .` and recreate the container. Also ensure the GUI's Train tab (or your CLI invocation) uses the `splatfactow_dataparser --colmap-path <rel>` subcommand, not `--pipeline.datamanager.dataparser.colmap-path`.

### `splatfacto-w-light` fails with `AssertionError: The size of image (3840, 2160) loaded does not match the camera parameters ((3838, 2161))`

Cause: an image older than v4.7, where the Train tab routed `splatfacto-w-light` through the NerfW dataparser and that parser derived the image size from the principal point. Fix: rebuild (`make build && make up`), then retrain — -light now uses the standard dataparser. The same symptom from `splatfacto-w` looks different: hundreds of `Image shape … does not match camera shape` lines followed by `CUDA out of memory`. That OOM is not a VRAM problem; it is the same stale-size bug, fixed by the v4.7 dataparser overlay.

### `splatfacto-w` stops at a random step with `AssertionError: torch.Size([112061, 16, 3])`

Cause: upstream colour-cache bug (splatfacto-w issue #14) in images older than v4.7 — the eval render reused colours cached before a densify/cull step. Fix: rebuild to v4.7; `smoke_test.sh` §17 confirms the patch is present.

### `ns-process-data … --sfm-tool hloc` fails with `you must install hloc toolbox!!`

Cause: the container was created from an image older than v4.7, where hloc was only ever installed by hand into a container's writable layer. Fix: rebuild to v4.7; `smoke_test.sh` §18 confirms the toolbox imports.

### "Aborted (core dumped)" at the end of training

Observed occasionally on first-ever training run in a fresh container, specifically at Python process shutdown after the `🎉 Training Finished 🎉` banner. Training completed successfully and all data was written to disk before the crash — checkpoint and TensorBoard logs are intact. Does not reproduce on subsequent runs in the same container. Likely related to first-run JIT compilation timing. Not a blocker; if it happens on a run, simply re-run — subsequent runs are clean.

---

## Appendix: full command reference

### Host-side (on your Pop!_OS / Ubuntu host)

```bash
# Clone and build
git clone https://github.com/AlexJumaW99/docker_splat_v2.git
cd docker_splat_v2
docker build --target nerfstudio -t splat-nerfstudio-v2 .

# Verify
docker images | grep splat-nerfstudio-v2

# Create persistent container (with both viser + tensorboard ports)
docker create --name splat-ns-v2 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/docker-stacks/docker_splat_v2/data:/workspace \
  -p 7007:7007 -p 6006:6006 -it splat-nerfstudio-v2

# Daily lifecycle
docker start splat-ns-v2
docker exec -it splat-ns-v2 bash
docker stop splat-ns-v2

# Clean slate (keeps data)
docker rm splat-ns-v2

# Verify port mappings
docker port splat-ns-v2

# Fix host ownership
sudo chown -R $USER:$USER ~/docker-stacks/docker_splat_v2/data
```

### Container-side (inside `docker exec -it splat-ns-v2 bash`)

```bash
# Process video into dataset
ns-process-data video \
  --data /workspace/yourvideo.mp4 \
  --output-dir /workspace/processed/yourvideo_splat \
  --num-frames-target 250 \
  --matching-method sequential

# Process image folder into dataset (use exhaustive for < 100 images)
ns-process-data images \
  --data /workspace/input_images/your_scene \
  --output-dir /workspace/processed/your_scene_splat \
  --matching-method exhaustive \
  --num-downscales 1 \
  --verbose

# Train splatfacto with viser + TensorBoard monitoring
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False

# Train splatfacto-w (uses its own dataparser subcommand)
ns-train splatfacto-w \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False \
  splatfactow_dataparser \
  --colmap-path colmap/sparse/0

# Launch TensorBoard in a second shell
tensorboard --logdir /workspace/outputs --bind_all --port 6006

# View a completed model
ns-viewer \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --viewer.websocket-host 0.0.0.0

# Export to PLY (built-in splatfacto / -big / -mcmc)
ns-export gaussian-splat \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --output-dir /workspace/outputs/<dataset>/export

# Export to PLY (splatfacto-w / splatfacto-w-light)
ns-export-splatfactow \
  --load-config /workspace/outputs/<dataset>/splatfacto-w-light/<timestamp>/config.yml \
  --output-dir /workspace/exports/splat

# Render a viser camera-path JSON to MP4
ns-render camera-path \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --camera-path-filename /workspace/processed/<project>/camera_paths/<timestamp>.json \
  --output-path /workspace/renders/<project>/<camera-path-stem>.mp4
```

### Browser URLs (on host)

| URL | Purpose | Required flags |
|---|---|---|
| `http://localhost:7007` | Viser 3D viewer | `--viewer.websocket-host 0.0.0.0` on `ns-train` |
| `http://localhost:6006` | TensorBoard metrics | `--bind_all` on `tensorboard` |

---

<div align="center">

**Need help?** Open an issue in this repo, or consult the [troubleshooting section](#troubleshooting) above.

</div>

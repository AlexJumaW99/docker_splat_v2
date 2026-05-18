<div align="center">

# docker_splat_v2

### Blackwell-native Docker pipeline for 3D Gaussian Splatting

**Turn videos into 3D Gaussian Splats on RTX 50-series GPUs.**
A fully reproducible Docker stack with COLMAP, gsplat, and Nerfstudio — configured from scratch for `sm_120` (Blackwell) compute capability.

---

![Linux](https://img.shields.io/badge/OS-Pop!__OS%2024.04-48B9C7?style=flat-square&logo=ubuntu&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat-square&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10_NGC-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-28.2-2496ED?style=flat-square&logo=docker&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RTX_5070_Ti_(sm__120)-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Verified_working-success?style=flat-square)

</div>

---

## Why this exists

The official Nerfstudio Docker image ships CUDA 11.8 and doesn't know what to do with a Blackwell GPU. The official COLMAP binary doesn't include Blackwell kernels. Gsplat's PyPI wheel JIT-compiles against whatever CUDA toolchain it finds — which breaks if you try to mix toolchains.

This repo solves all three problems in a single reproducible Docker build:

- **Blackwell `sm_120` kernels** compiled from source for both COLMAP and gsplat
- **Pinned against NGC PyTorch 25.11** (CUDA 13.0, Python 3.12, sm_120 pre-compiled)
- **Multi-stage Dockerfile** with `--target` support for incremental rebuilds
- **Compatibility patch included** for the COLMAP `SiftExtraction` → `FeatureExtraction` flag rename that breaks Nerfstudio 1.1.5 out of the box
- **COLMAP multi-submap auto-fix** via a PATH-shim that promotes the largest submap into `sparse/0` so `ns-process-data` doesn't silently export a 2-frame bootstrap stub
- **Real-time training monitoring** via viser (3D preview) + TensorBoard (metrics/eval images) simultaneously

End-to-end tested on an 80-second 720p orbit video: 98.12% pose recovery, ~84M gaussians at 7000 iterations, ~15 minutes on an RTX 5070 Ti.

---

## Stack

| Component | Version | Source | Purpose |
|---|---|---|---|
| **Base image** | `nvcr.io/nvidia/pytorch:25.11-py3` | NGC | CUDA 13.0 + PyTorch 2.10 + sm_120 |
| **COLMAP** | `4.1.0.dev0` (commit `ffb5404`) | Built from source, pinned commit | SfM + GPU SIFT |
| **gsplat** | `1.5.3` | PyPI (JIT-compiled at runtime) | Gaussian rasterization — unlocks 2DGS, fused-SSIM, 3DGUT, Fisheye-GS, F-Theta inside splatfacto via config flags |
| **Nerfstudio** | `1.1.5` (git `50e0e3c`) | git main, pinned | Training CLI + viewer |
| **splatfacto-w** | `0.1.5` (git `119a3bf`) | [KevinXu02/splatfacto-w](https://github.com/KevinXu02/splatfacto-w) | `splatfacto-w` / `splatfacto-w-light` methods |
| **Viser** | `1.0.0` | Pinned by Nerfstudio (`viser==1.0.0`) | Web-based 3D viewer (port 7007) |
| **TensorBoard** | `2.20.0` | Bundled with NGC PyTorch | Metrics + eval image logging (port 6006) |
| **ffmpeg** | `6.1.1` | apt | Video frame extraction |

---

## Quick start

**Prerequisites:** Linux host with NVIDIA driver ≥ 580, Docker 28+, NVIDIA Container Toolkit, and a Blackwell GPU (RTX 5070 Ti, 5080, 5090, etc.). See [USER_GUIDE.md](USER_GUIDE.md#host-prerequisites) for full setup.

### TL;DR — Makefile shortcuts

If you have `make` installed, one target covers disaster recovery end-to-end:

```bash
git clone https://github.com/AlexJumaW99/docker_splat_v2.git
cd docker_splat_v2
make fresh        # down → build → up → ./smoke_test.sh (full verification)
```

Everyday: `make up` (start), `make shell` (bash into `/workspace`), `make test` (run smoke suite), `make down` (stop + remove). Run `make help` for the list, or skip the Makefile entirely and use the raw commands in the numbered steps below — they're equivalent.

### 1. Clone and build

```bash
git clone https://github.com/AlexJumaW99/docker_splat_v2.git
cd docker_splat_v2
docker build --target nerfstudio -t splat-nerfstudio-v2 .
```

Build takes **25–40 minutes** on first run (COLMAP compile is the bottleneck, ~10 min). Subsequent rebuilds leverage Docker's layer cache for near-instant iteration.

### 2. Create and start the persistent container

```bash
mkdir -p ~/docker-stacks/docker_splat_v2/data \
         ~/docker-stacks/docker_splat_v2/data_v2

docker create --name splat-ns-v2 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/docker-stacks/docker_splat_v2/data:/workspace \
  -v ~/docker-stacks/docker_splat_v2/data_v2:/warehouse \
  -p 7007:7007 -p 6006:6006 -it splat-nerfstudio-v2

docker start splat-ns-v2
docker exec -it splat-ns-v2 bash
```

**Two ports mapped:** `7007` for the viser 3D viewer, `6006` for TensorBoard metrics. Both are optional but recommended for full visibility during training.

**Two bind mounts:** `data/` is the legacy artifact tree (still used by raw CLI workflows); `data_v2/` is the new ULID-keyed, manifest-driven warehouse where the GUI writes new sources, processed datasets, and training runs. `make up` and `docker compose up -d` set both up automatically.

### 3. Process a video → train → view

```bash
# Inside the container — drop your video at ~/docker-stacks/docker_splat_v2/data/ first
ns-process-data video --data /workspace/yourvideo.mp4 \
  --output-dir /workspace/processed/yourvideo_splat \
  --num-frames-target 250 --matching-method sequential

ns-train splatfacto --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs --max-num-iterations 7000 \
  --vis viewer+tensorboard \
  --viewer.websocket-host 0.0.0.0 --viewer.quit-on-train-completion False
```

Open `http://localhost:7007` for the 3D view. In a second shell (`docker exec -it splat-ns-v2 bash`), run `tensorboard --logdir /workspace/outputs --bind_all --port 6006` and open `http://localhost:6006` for live metrics.

Available training methods: `splatfacto`, `splatfacto-big`, `splatfacto-mcmc`, `splatfacto-w`, `splatfacto-w-light`.

For detailed workflow, troubleshooting, and export instructions, see **[USER_GUIDE.md](USER_GUIDE.md)**.

### 4. Optional: desktop GUI

A PySide6 desktop app lives in [`gui/`](gui/) and drives the same `splat-ns-v2` container without the CLI. It runs the exact `ns-process-data` / `ns-train` / `ns-export` / `ns-render` / `ns-viewer` invocations you'd run by hand, with per-project state, an "import processed dataset" flow for already-reconstructed folders like Tanks & Temples, an **Exports tab** for PLY export (auto-routed to the splatfacto-w-aware exporter when applicable) and camera-path video rendering, and a **Viewers tab** for launching ns-viewer against finished checkpoints and starting/stopping TensorBoard scoped to the active project. Process, Train, Export, and Render outputs all land in the manifest-driven `data_v2/` warehouse, ULID-keyed and lineage-linked via `parent.run_id` so any artifact traces back to its source video through file reads alone. The Process tab accepts both video files and image folders as input, with automatic HEIC/HEIF → JPEG conversion (quality 100, EXIF preserved) for iPhone photos — originals are stashed in a `heic_originals/` subfolder.

```bash
cd gui && ./run.sh
```

First launch creates `.venv/` and installs PySide6, pydantic, and pillow-heif. See [gui/README.md](gui/README.md) and the [Desktop GUI section](USER_GUIDE.md#desktop-gui-optional) of the user guide.

---

## Architecture

The Dockerfile uses a 5-stage build. Each stage is independently buildable with `docker build --target <stage>`, which dramatically speeds up iteration during development.

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: base                                              │
│  NGC PyTorch 25.11 + Blackwell env vars                     │
│  ──────────────────────────────────────────────             │
│  TORCH_CUDA_ARCH_LIST=12.0  FORCE_CUDA=1  CUDA_HOME=...     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Stage 2: system-deps                                       │
│  apt install — COLMAP + ffmpeg build dependencies           │
│  ──────────────────────────────────────────────             │
│  Qt6, Ceres, Boost, Eigen, MKL, glog, gtest, SuiteSparse…   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Stage 3: colmap                                            │
│  COLMAP from source, CMAKE_CUDA_ARCHITECTURES=120           │
│  ──────────────────────────────────────────────             │
│  SIFT on GPU via sm_120 kernels                             │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Stage 4: gsplat (pinned to 1.5.3 PyPI wheel)               │
│  Re-installed after nerfstudio's 1.4.0 downgrade in Stage 5 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Stage 5: nerfstudio                                        │
│  pip install nerfstudio (git pin) + COLMAP/torch.load       │
│  patches + splatfacto-w (KevinXu02, pinned commit)          │
│  ──────────────────────────────────────────────             │
│  ns-train, ns-process-data, ns-viewer, ns-export, ns-render │
└─────────────────────────────────────────────────────────────┘
```

---

## What's been fixed

Seven non-obvious issues you'd hit trying to build this yourself:

| Issue | Root cause | Fix |
|---|---|---|
| COLMAP CMake fails with `Could NOT find Qt6Svg` | Undocumented new Qt6 Svg dependency in COLMAP 4.1.0.dev0 | `qt6-svg-dev` added to `apt install` list |
| Nerfstudio `ns-process-data video` fails with `unrecognised option '--SiftExtraction.use_gpu'` | COLMAP renamed SIFT-specific GPU flags to generic `FeatureExtraction.use_gpu` when ALIKED support was added | `sed` patch applied to nerfstudio's `colmap_utils.py` in Stage 5 |
| `PYTORCH_CUDA_ALLOC_CONF` warning on every invocation | Env var renamed to `PYTORCH_ALLOC_CONF` in PyTorch 2.10 | Use the new name in the `docker run` / `docker create` command |
| `ns-train` fails to resume from checkpoints with `_pickle.UnpicklingError: Weights only load failed` | PyTorch 2.6 flipped `torch.load`'s `weights_only` default to `True`; nerfstudio's checkpoints include non-tensor state and fail the new safety check | `sed` patch in Stage 5 pins `weights_only=False` in nerfstudio's loader |
| `ns-export gaussian-splat` crashes with `AssertionError` on splatfacto-w runs | Upstream exporter hard-asserts `isinstance(pipeline.model, SplatfactoModel)`, but `SplatfactoWModel` inherits from `Model` directly — and splatfacto-w computes SH coefficients per-frame from an appearance MLP rather than storing them as tensors | Image ships a custom `ns-export-splatfactow` CLI that bakes one appearance embedding (average of training cameras by default) into per-gaussian SH coefficients and writes an inria-format `.ply` with the same layout as `ns-export gaussian-splat`. The GUI Exports tab auto-routes to it for splatfacto-w/-light. |
| `ns-process-data` silently exports a 2-frame `transforms.json` on longer videos | COLMAP 4.1's `incremental_pipeline` seeds a 2-image bootstrap as `sparse/0` before growing the main reconstruction into `sparse/1`. Nerfstudio's converter hardcodes `colmap_model_path = Path("colmap/sparse/0")`, so it reads the bootstrap stub and the real 200+-frame model sits unused on disk. | Image ships a PATH-shim (`/opt/colmap-shim/colmap`) that transparently wraps `colmap mapper`. After a successful mapper run it probes every numeric submap with `model_analyzer`, name-swaps the largest into `sparse/0`, and preserves the original `sparse/0` under its old name for inspection. Also injects `--Mapper.snapshot_path` so viser can watch incremental progress. Non-mapper subcommands pass through untouched. |
| Two builds done a month apart could ship different COLMAP code under the same `4.1.0.dev0` version string | `git clone --depth 1` always pulled whatever was at `main` HEAD; nothing pinned the actual commit | `ARG COLMAP_COMMIT` in Stage 3 pins to a specific upstream SHA (currently `ffb5404`); partial clone (`--filter=blob:none`) checks out exactly that commit. Resolved SHA is recorded at `/etc/colmap-commit.sha` in the image, and `smoke_test.sh` asserts it matches the Dockerfile pin. Bumping the SHA + rerunning smoke tests is the upgrade ritual, matching the convention already in place for `NERFSTUDIO_COMMIT` / `SPLATFACTOW_COMMIT` / `VOCAB_TREE_COMMIT`. |
| `splatfacto-w` training crashes with `FileNotFoundError: dense/sparse/cameras.bin` on non-phototourism datasets | Upstream `nerfw_dataparser.py` hardcodes phototourism-specific paths (`dense/sparse/`, `dense/images/`), asserts only `PINHOLE` cameras, iterates camera models instead of images (reads 1 row for a shared-intrinsics dataset instead of N), and requires a TSV train/test split file that only phototourism ships | `COPY` overlay replaces `nerfw_dataparser.py` in Stage 5: uses `config.colmap_path`, auto-detects `dense/images/` or `images/`, accepts `OPENCV` cameras, iterates images (not camera models), and falls back to a deterministic fraction-based eval split when no TSV file is present. The GUI's Train tab emits the `splatfactow_dataparser --colmap-path <rel>` tyro subcommand so the patched parser receives the project's COLMAP path. |
| `splatfacto-w` training crashes with `IndexError` on gsplat 1.5.3 (`mask shape [200206] is not broadcastable to [100103, 2]`) | gsplat 1.5.3 changed `info["radii"]` from shape `[C, N]` (scalar per-gaussian visibility radius) to `[C, N, 2]` (per-axis radii for anisotropic gaussians). `splatfactow_model.py` sliced `info["radii"][0]` to get `[N]`, but now gets `[N, 2]`; downstream boolean masks built from the scalar `.flatten()` doubled in length | `sed` patch in Stage 5 changes `self.radii = info["radii"][0]` to `self.radii = info["radii"][0].amax(dim=-1)`, collapsing the per-axis `[N, 2]` back to a scalar `[N]` via the conservative (max) axis. |

All nine are baked into the Dockerfile and the documented `docker create` command.

---

## Verification status

| Check | Result |
|---|---|
| NGC base image pull and tag detection | ✅ Verified |
| apt dependency install (Stage 2) | ✅ Verified |
| COLMAP source build with sm_120 | ✅ Verified — binary reports "with CUDA" |
| COLMAP commit pin (Dockerfile vs image) | ✅ `smoke_test.sh` asserts `/etc/colmap-commit.sha` matches `ARG COLMAP_COMMIT` |
| gsplat rasterization on `cuda:0` | ✅ Verified — `RASTERIZATION SUCCESS` smoke test |
| Nerfstudio CLI available | ✅ Built-in methods including splatfacto, splatfacto-big, splatfacto-mcmc |
| External method: splatfacto-w | ✅ Registers `splatfacto-w` + `splatfacto-w-light` via nerfstudio entry points |
| `.ply` export for splatfacto-w / -light | ✅ Custom `ns-export-splatfactow` CLI baked into the image; GUI routes automatically |
| `ns-process-data video` end-to-end | ✅ 98.12% pose recovery on test video |
| COLMAP multi-submap promotion shim | ✅ Largest submap swapped into `sparse/0` automatically on splintered runs |
| `ns-train splatfacto` full training | ✅ 7000 iterations, ~84M gaussians, ~15 min |
| Viser viewer reachable from host | ✅ Port 7007 bridged, scene renders live |
| TensorBoard reachable from host | ✅ Port 6006 bridged, scalars + events logged |
| Viewer split-mode patch (gl_z_buf_depth) | ✅ `smoke_test.sh` verifies patched anchors present + depth compositing intact |
| `--vis viewer+tensorboard` simultaneous | ✅ Verified — both work in parallel, minor eval stutter |
| `splatfacto-w` training (non-phototourism) | ✅ 50-iteration smoke test with OPENCV cameras, 4694 images, 100k 3D points |
| NerfW dataparser patch (`nerfw_dataparser.py`) | ✅ `COPY` overlay in Stage 5; `smoke_test.sh` verifies splatfactow config imports |
| gsplat 1.5.3 radii shape fix (`splatfactow_model.py`) | ✅ `sed` patch in Stage 5; densification logic operates on scalar `[N]` radii |

---

## Known limitations

- **Blackwell-only by design.** `TORCH_CUDA_ARCH_LIST=12.0` targets `sm_120` exclusively. To run on an Ada or Ampere card, change to `8.9` or `8.6` respectively in the Dockerfile (and rebuild).
- **16 GB VRAM assumption.** Training large scenes (> 500 images at 4K) may OOM on a 5070 Ti. The `PYTORCH_ALLOC_CONF=expandable_segments:True` env var helps, but is not magic. For heavier scenes, use a 5080 (16GB) or 5090 (32GB).
- **Container runs as root.** Files written to `/workspace/outputs/` will be `root:root` on the host. Run `sudo chown -R $USER:$USER ~/docker-stacks/docker_splat_v2/data` periodically.
- **`ns-download-data` is broken upstream.** Nerfstudio's bundled dataset downloader fails due to Google Drive rate-limiting — this is not a bug in this repo, it's a [long-running upstream issue](https://github.com/nerfstudio-project/nerfstudio/issues/3134). Use your own video data.
- **Viser may stutter briefly during eval steps** when used alongside TensorBoard. Documented upstream; typically a few seconds of laggy viewport every ~1000 iterations. Training metrics are unaffected.

---

## Project history

`v3-working` (commit `349e43e`) is the tagged milestone — the first Dockerfile that produced a working pipeline from scratch. Subsequent versioned commits (v3.1, v3.2) extend it without retagging.

Commit history (`git log --oneline`, most recent first):

```
Fix splatfacto-w training on non-phototourism datasets (NerfW dataparser patch + gsplat 1.5.3 radii fix + GUI Train tab splatfactow_dataparser subcommand)
Migrate GUI Exports tab to the manifest-driven warehouse: splat PLYs land under data_v2/exports/splat/<EXPORT_ULID>/ and rendered videos under data_v2/renders/<RENDER_ULID>/, each with a manifest.json whose parent.run_id closes the lineage chain back through train → process → source. Run discovery rewired from legacy data/outputs/ to data_v2/trained/*/manifest.json filtered by the project's processed ULID; scan_catalog.py bumped to schema v2 with warehouse_sources / warehouse_processed / warehouse_train / warehouse_exports / warehouse_renders tables joined by manifest ULID rather than filename parsing. mesh/ and pointcloud/ sub-folders reserved by the layout but not wired (ns-export poisson / tsdf / marching-cubes / pointcloud all require the volumetric-NeRF model interface SplatfactoModel doesn't implement).
Migrate GUI Process + Train tabs to a manifest-driven ULID warehouse under data_v2/: SourceManifest → ProcessManifest → TrainManifest chain with declared parent.run_id at every hop, frozen v1.0 JSON Schema artifact, pending.json write-ahead-log sidecar so a closed GUI never strands a successful run, "🎉 Training Finished" stdout banner detection so manifests finalize when training ends (not when the viewer process exits), and a `python -m app.warehouse backfill-train` CLI that reconstructs manifests for orphan runs from config.yml + the pending sidecar
Upgrade gsplat 1.4.0 -> 1.5.3 to unlock 2DGS, fused-SSIM, 3DGUT, Fisheye-GS
Add scan_catalog.py: rebuild catalog.db index over data/ tree
Add vizer live point-cloud viewer button to Process tab + project management enhancements
Add HEIC/HEIF auto-conversion to JPEG in the Process tab GUI (pillow-heif, quality 100, EXIF preserved)
Add Viewers tab with ns-viewer launch, TensorBoard start/stop, and live port status (GUI)
Document network-detachment recovery + add make heal-net
Add image-folder source mode to the Process tab GUI
Pin COLMAP commit + smoke-test SHA against Dockerfile pin
Add Makefile wrapping compose + smoke-test for disaster-recovery shortcuts
Fix viewer split-mode crash by filtering gl_z_buf_depth from dropdowns
Relocate repo from ~/Desktop to ~/docker-stacks; pin PySide6==6.11.0
Capture runtime contract in docker-compose.yml + trim build context
Correct viser version in stack table (0.2.7 → 1.0.0)
Auto-promote largest COLMAP submap into sparse/0 via PATH-shim
Warn on Process tab when COLMAP splinters into multiple submaps
Pre-place faiss-format vocab tree for vocab_tree matcher
Add help-badge tooltips across GUI and retheme to light palette
Add advanced controls to Process tab and drop broken depth-debug flag
Label splatfacto-mcmc runs distinctly in Exports tab
Add splatfacto-w .ply exporter and route GUI Exports tab to it
Bake splatfacto-w into image and expose in Train dropdown (GUI)
Add Exports tab with .ply export and camera-path video render (GUI)
Fork naming to v2 and pin nerfstudio to git main (enables splatfacto-mcmc)
Add "import processed dataset" flow to skip straight to training (GUI)
Add Train tab and viser-port safety helpers (GUI)
Add smoke_test.sh for container verification
v3.2 — patch nerfstudio torch.load for PyTorch 2.6+ weights_only default
Add GUI application source
v3.1 — add TensorBoard monitoring (EXPOSE 6006) + docs
v3   — patch nerfstudio for COLMAP SiftExtraction→FeatureExtraction flag rename
v2.1 — add trailing newline + expand Stage 4 explanatory comment
v2   — multi-stage refactor + add qt6-svg-dev to fix COLMAP build
v1   — single-stage build, fails at COLMAP/Qt6Svg (baseline)
```

---

## Acknowledgments

Built on top of excellent open-source work:

- [**Nerfstudio**](https://github.com/nerfstudio-project/nerfstudio) — the training framework and CLI
- [**gsplat**](https://github.com/nerfstudio-project/gsplat) — the CUDA Gaussian rasterizer
- [**splatfacto-w**](https://github.com/KevinXu02/splatfacto-w) — in-the-wild splatfacto variants (KevinXu02)
- [**COLMAP**](https://github.com/colmap/colmap) — Structure-from-Motion for camera pose recovery
- [**NVIDIA NGC**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) — PyTorch base image with Blackwell support
- [**Viser**](https://github.com/nerfstudio-project/viser) — the web-based 3D viewer
- [**TensorBoard**](https://www.tensorflow.org/tensorboard) — training metric visualization

---

<div align="center">

**Questions or issues?** Open an issue in this repo.

</div>

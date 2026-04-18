<div align="center">

# docker_splat

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

End-to-end tested on an 80-second 720p orbit video: 98.12% pose recovery, ~84M gaussians at 7000 iterations, ~15 minutes on an RTX 5070 Ti.

---

## Stack

| Component | Version | Source | Purpose |
|---|---|---|---|
| **Base image** | `nvcr.io/nvidia/pytorch:25.11-py3` | NGC | CUDA 13.0 + PyTorch 2.10 + sm_120 |
| **COLMAP** | `4.1.0.dev0` | Built from source (`main` branch) | SfM + GPU SIFT |
| **gsplat** | `1.4.0` | PyPI (JIT-compiled at runtime) | Gaussian rasterization |
| **Nerfstudio** | `1.1.5` | PyPI | Training CLI + viewer |
| **Viser** | `0.2.7` | Pinned by Nerfstudio | Web-based 3D viewer |
| **ffmpeg** | `6.1.1` | apt | Video frame extraction |

---

## Quick start

**Prerequisites:** Linux host with NVIDIA driver ≥ 580, Docker 28+, NVIDIA Container Toolkit, and a Blackwell GPU (RTX 5070 Ti, 5080, 5090, etc.). See [USER_GUIDE.md](USER_GUIDE.md#host-prerequisites) for full setup.

### 1. Clone and build

```bash
git clone https://github.com/AlexJumaW99/docker_splat.git
cd docker_splat
docker build --target nerfstudio -t splat-nerfstudio .
```

Build takes **25–40 minutes** on first run (COLMAP compile is the bottleneck, ~10 min). Subsequent rebuilds leverage Docker's layer cache for near-instant iteration.

### 2. Create and start the persistent container

```bash
mkdir -p ~/Desktop/docker_splat/data

docker create --name splat-ns --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/Desktop/docker_splat/data:/workspace \
  -p 7007:7007 -it splat-nerfstudio

docker start splat-ns
docker exec -it splat-ns bash
```

### 3. Process a video → train → view

```bash
# Inside the container — drop your video at ~/Desktop/docker_splat/data/ first
ns-process-data video --data /workspace/yourvideo.mp4 \
  --output-dir /workspace/processed/yourvideo_splat \
  --num-frames-target 250 --matching-method sequential

ns-train splatfacto --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs --max-num-iterations 7000 \
  --viewer.websocket-host 0.0.0.0 --viewer.quit-on-train-completion False
```

Open `http://localhost:7007` in your browser to watch training live.

For detailed workflow, troubleshooting, and export instructions, see **[USER_GUIDE.md](USER_GUIDE.md)**.

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
│  Stage 4: gsplat (cache accelerator)                        │
│  gsplat from git, replaced by 1.4.0 wheel in next stage     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Stage 5: nerfstudio                                        │
│  pip install nerfstudio + COLMAP compatibility patch        │
│  ──────────────────────────────────────────────             │
│  ns-train, ns-process-data, ns-viewer, ns-export            │
└─────────────────────────────────────────────────────────────┘
```

---

## What's been fixed

Three non-obvious issues you'd hit trying to build this yourself:

| Issue | Root cause | Fix |
|---|---|---|
| COLMAP CMake fails with `Could NOT find Qt6Svg` | Undocumented new Qt6 Svg dependency in COLMAP 4.1.0.dev0 | `qt6-svg-dev` added to `apt install` list |
| Nerfstudio `ns-process-data video` fails with `unrecognised option '--SiftExtraction.use_gpu'` | COLMAP renamed SIFT-specific GPU flags to generic `FeatureExtraction.use_gpu` when ALIKED support was added | `sed` patch applied to nerfstudio's `colmap_utils.py` in Stage 5 |
| `PYTORCH_CUDA_ALLOC_CONF` warning on every invocation | Env var renamed to `PYTORCH_ALLOC_CONF` in PyTorch 2.10 | Use the new name in the `docker run` / `docker create` command |

All three are baked into the v3 Dockerfile and the documented `docker create` command.

---

## Verification status

| Check | Result |
|---|---|
| NGC base image pull and tag detection | ✅ Verified |
| apt dependency install (Stage 2) | ✅ Verified |
| COLMAP source build with sm_120 | ✅ Verified — binary reports "with CUDA" |
| gsplat rasterization on `cuda:0` | ✅ Verified — `RASTERIZATION SUCCESS` smoke test |
| Nerfstudio CLI available | ✅ 17 built-in methods including splatfacto |
| `ns-process-data video` end-to-end | ✅ 98.12% pose recovery on test video |
| `ns-train splatfacto` full training | ✅ 7000 iterations, ~84M gaussians, ~15 min |
| Viser viewer reachable from host | ✅ Port 7007 bridged, scene renders live |

---

## Known limitations

- **Blackwell-only by design.** `TORCH_CUDA_ARCH_LIST=12.0` targets `sm_120` exclusively. To run on an Ada or Ampere card, change to `8.9` or `8.6` respectively in the Dockerfile (and rebuild).
- **16 GB VRAM assumption.** Training large scenes (> 500 images at 4K) may OOM on a 5070 Ti. The `PYTORCH_ALLOC_CONF=expandable_segments:True` env var helps, but is not magic. For heavier scenes, use a 5080 (16GB) or 5090 (32GB).
- **Container runs as root.** Files written to `/workspace/outputs/` will be `root:root` on the host. Run `sudo chown -R $USER:$USER ~/Desktop/docker_splat/data` periodically.
- **`ns-download-data` is broken upstream.** Nerfstudio's bundled dataset downloader fails due to Google Drive rate-limiting — this is not a bug in this repo, it's a [long-running upstream issue](https://github.com/nerfstudio-project/nerfstudio/issues/3134). Use your own video data.

---

## Project history

Tagged versions in git mark verified milestones:

| Tag | Commit | Description |
|---|---|---|
| `v3-working` | `349e43e` | First Dockerfile producing a working pipeline from scratch |

Commit history (`git log --oneline`):

```
v3  — patch nerfstudio for COLMAP SiftExtraction→FeatureExtraction flag rename
v2.1 — add trailing newline + expand Stage 4 explanatory comment
v2   — multi-stage refactor + add qt6-svg-dev to fix COLMAP build
v1   — single-stage build, fails at COLMAP/Qt6Svg (baseline)
```

---

## Acknowledgments

Built on top of excellent open-source work:

- [**Nerfstudio**](https://github.com/nerfstudio-project/nerfstudio) — the training framework and CLI
- [**gsplat**](https://github.com/nerfstudio-project/gsplat) — the CUDA Gaussian rasterizer
- [**COLMAP**](https://github.com/colmap/colmap) — Structure-from-Motion for camera pose recovery
- [**NVIDIA NGC**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) — PyTorch base image with Blackwell support
- [**Viser**](https://github.com/nerfstudio-project/viser) — the web-based 3D viewer

---

<div align="center">

**Questions or issues?** Open an issue in this repo.

</div>

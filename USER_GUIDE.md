<div align="center">

# User Guide

### Complete walkthrough for `docker_splat`

From host setup to your first exported Gaussian Splat PLY.

---

</div>

## Table of contents

1. [Host prerequisites](#host-prerequisites)
2. [Building the image](#building-the-image)
3. [Container lifecycle](#container-lifecycle)
4. [Recording videos for splatfacto](#recording-videos-for-splatfacto)
5. [Processing a video](#processing-a-video)
6. [Training splatfacto](#training-splatfacto)
7. [Viewing in viser](#viewing-in-viser)
8. [Exporting a PLY file](#exporting-a-ply-file)
9. [Troubleshooting](#troubleshooting)
10. [Appendix: full command reference](#appendix-full-command-reference)

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
git clone https://github.com/AlexJumaW99/docker_splat.git
cd docker_splat
docker build --target nerfstudio -t splat-nerfstudio .
```

**Expected duration:** 25–40 minutes on first run. Breakdown:

| Stage | Time | Cacheable |
|---|---|---|
| Stage 1–2 (apt deps) | 3–5 min | Yes (until apt list changes) |
| Stage 3 (COLMAP compile) | 10–15 min | Yes (until COLMAP source changes) |
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
docker build --target nerfstudio -t splat-nerfstudio .
```

### Verify the built image

```bash
docker images | grep splat-nerfstudio
```

Expected: one line showing the image with size around **26 GB**. The size is large because NGC PyTorch is ~20 GB on its own; COLMAP, nerfstudio, and dependencies add the rest.

---

## Container lifecycle

### Create the persistent container (once)

```bash
mkdir -p ~/Desktop/docker_splat/data

docker create \
  --name splat-ns \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/Desktop/docker_splat/data:/workspace \
  -p 7007:7007 \
  -it \
  splat-nerfstudio
```

**What each flag does:**

| Flag | Purpose |
|---|---|
| `--name splat-ns` | Names the container for easy reference |
| `--gpus all` | GPU passthrough |
| `--ipc=host` | Shared memory for PyTorch multi-worker dataloaders |
| `--ulimit memlock=-1 --ulimit stack=67108864` | Required by NGC PyTorch for proper memory management |
| `--shm-size=16g` | Shared memory size for large tensors |
| `-e PYTORCH_ALLOC_CONF=expandable_segments:True` | Memory allocator config; reduces fragmentation on long training runs |
| `-v ~/Desktop/docker_splat/data:/workspace` | Bind mount — data persists on host outside the container |
| `-p 7007:7007` | Viser web viewer port |
| `-it` | Interactive TTY |

### Daily use

```bash
# Start a previously created container
docker start splat-ns

# Attach a shell
docker exec -it splat-ns bash

# Exit the shell (container keeps running)
exit

# Stop the container (state preserved)
docker stop splat-ns
```

### When you need a clean slate

```bash
docker stop splat-ns
docker rm splat-ns
# then re-run the `docker create` block above
```

Bind-mounted data at `~/Desktop/docker_splat/data` is unaffected — only the container's writable layer is destroyed.

### File ownership housekeeping

The container runs as root, so files it writes to `/workspace/` will be root-owned on the host. Every so often:

```bash
sudo chown -R $USER:$USER ~/Desktop/docker_splat/data
```

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

Place your video at `~/Desktop/docker_splat/data/yourvideo.mp4` on the host. This becomes `/workspace/yourvideo.mp4` inside the container.

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
│   └── sparse/0/        # COLMAP sparse model (binary format)
├── sparse_pc.ply        # Sparse point cloud (seed for splatfacto)
└── transforms.json      # Nerfstudio camera poses (the key file)
```

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

## Training splatfacto

```bash
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False
```

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

### Final output location

When training completes, your model lands at:

```
/workspace/outputs/<dataset_name>/splatfacto/<YYYY-MM-DD_HHMMSS>/
├── config.yml                  # Training config (needed for ns-viewer and ns-export)
└── nerfstudio_models/
    └── step-NNNNNNNNN.ckpt     # Model checkpoint
```

Note the timestamp directory — you'll reference it for viewing and exporting.

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

### Post-training view with ns-viewer

If `--viewer.quit-on-train-completion False` was set, viser stays alive after training ends — just keep the terminal open.

To re-open a completed model in a fresh session:

```bash
ns-viewer \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --viewer.websocket-host 0.0.0.0
```

---

## Exporting a PLY file

Once trained, export to a portable PLY that can be opened in any Gaussian Splatting viewer or imported into Blender (with a plugin like [3DGS Render](https://github.com/ReshotAI/gaussian-splatting-blender-addon)).

```bash
ns-export gaussian-splat \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --output-dir /workspace/outputs/<dataset>/export
```

Output: `splat.ply` in the specified directory. File size is typically 100–500 MB for a 7000-iteration model with ~80M gaussians.

### Fixing host-side ownership

The PLY will be written as root inside the container. To open it on the host:

```bash
sudo chown -R $USER:$USER ~/Desktop/docker_splat/data/outputs
```

---

## Troubleshooting

### "Could NOT find Qt6Svg" during build

Cause: COLMAP 4.1.0.dev0 requires `qt6-svg-dev`, which older documentation doesn't mention.

Fix: already in the v2+ Dockerfile. If you see this, you're building from an outdated source.

### `ns-process-data` fails with "unrecognised option '--SiftExtraction.use_gpu'"

Cause: COLMAP renamed this flag to `--FeatureExtraction.use_gpu`; nerfstudio 1.1.5 still uses the old name.

Fix: already patched in Stage 5 of the v3 Dockerfile. If you hit this, your image was built from a pre-v3 Dockerfile. Rebuild with `docker build --target nerfstudio -t splat-nerfstudio .`

### Training appears to hang for 1–2 minutes on first run

Cause: gsplat is JIT-compiling CUDA kernels for your sm_120 GPU.

Fix: wait. Do not Ctrl+C. Subsequent runs will be instant.

### "Failed to retrieve file url" when running `ns-download-data`

Cause: [long-running upstream nerfstudio bug](https://github.com/nerfstudio-project/nerfstudio/issues/3134) — Google Drive rate-limits gdown.

Fix: not a fix for this repo to make. Use your own video data.

### Browser can't connect to `localhost:7007`

Checklist:

1. Container was created with `-p 7007:7007` flag
2. Training/viewer was started with `--viewer.websocket-host 0.0.0.0`
3. No other process on the host is using port 7007 (`sudo lsof -i :7007`)
4. Firewall isn't blocking localhost (unusual, but check)

### `docker build` fails on Stage 3 (COLMAP)

Most common causes:

- Out of disk space — COLMAP compile needs ~5 GB of build artifacts temporarily
- apt package missing from Stage 2 (if you modified the apt list)
- GitHub rate-limiting the `git clone` of colmap/colmap.git (retry in a few minutes)

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

---

## Appendix: full command reference

### Host-side (on your Pop!_OS / Ubuntu host)

```bash
# Clone and build
git clone https://github.com/AlexJumaW99/docker_splat.git
cd docker_splat
docker build --target nerfstudio -t splat-nerfstudio .

# Verify
docker images | grep splat-nerfstudio

# Create persistent container
docker create --name splat-ns --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/Desktop/docker_splat/data:/workspace \
  -p 7007:7007 -it splat-nerfstudio

# Daily lifecycle
docker start splat-ns
docker exec -it splat-ns bash
docker stop splat-ns

# Clean slate (keeps data)
docker rm splat-ns

# Fix host ownership
sudo chown -R $USER:$USER ~/Desktop/docker_splat/data
```

### Container-side (inside `docker exec -it splat-ns bash`)

```bash
# Process video into dataset
ns-process-data video \
  --data /workspace/yourvideo.mp4 \
  --output-dir /workspace/processed/yourvideo_splat \
  --num-frames-target 250 \
  --matching-method sequential

# Train splatfacto
ns-train splatfacto \
  --data /workspace/processed/yourvideo_splat \
  --output-dir /workspace/outputs \
  --max-num-iterations 7000 \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.quit-on-train-completion False

# View a completed model
ns-viewer \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --viewer.websocket-host 0.0.0.0

# Export to PLY
ns-export gaussian-splat \
  --load-config /workspace/outputs/<dataset>/splatfacto/<timestamp>/config.yml \
  --output-dir /workspace/outputs/<dataset>/export
```

---

<div align="center">

**Need help?** Open an issue in this repo, or consult the [troubleshooting section](#troubleshooting) above.

</div>

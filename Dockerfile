# syntax=docker/dockerfile:1.6

# ============================================================
# Stage 1: base — NGC PyTorch with Blackwell-ready env vars
# ============================================================
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt

# ============================================================
# Stage 2: system-deps — apt packages (COLMAP + ffmpeg build deps)
# v2 fix: added qt6-svg-dev (COLMAP CMake requires Qt6 Svg component,
# not documented in COLMAP's official install docs but required by
# their CMakeLists.txt as of 4.1.0.dev0).
# ============================================================
FROM base AS system-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git cmake ninja-build build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libopenimageio-dev openimageio-tools \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
        libcgal-dev \
        libceres-dev \
        libsuitesparse-dev \
        libcurl4-openssl-dev libssl-dev \
        libmkl-full-dev \
    && mkdir -p /usr/include/opencv4 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 3: colmap — COLMAP built from source with sm_120
# ============================================================
FROM system-deps AS colmap

RUN git clone --depth 1 https://github.com/colmap/colmap.git /opt/colmap_src \
    && cd /opt/colmap_src \
    && mkdir build && cd build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DBLA_VENDOR=Intel10_64lp \
    && ninja \
    && ninja install \
    && cd / && rm -rf /opt/colmap_src

# ============================================================
# Stage 4: gsplat — CUDA rasterizer compiled against NGC torch
# Note: nerfstudio in Stage 5 pins gsplat==1.4.0 and will replace
# this install. Kept for cache efficiency during iteration; the
# final gsplat 1.4.0 wheel uses JIT compilation that picks up
# TORCH_CUDA_ARCH_LIST=12.0 at runtime for sm_120 kernels.
# ============================================================
FROM colmap AS gsplat

RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/nerfstudio-project/gsplat.git

# ============================================================
# Stage 5: nerfstudio — final image with CLI and splatfacto methods
#
# v3 fix: patch nerfstudio colmap_utils.py for COLMAP 4.1.0.dev0 flag
# rename. COLMAP moved --SiftExtraction.use_gpu to
# --FeatureExtraction.use_gpu (and same for SiftMatching) when it added
# ALIKED as a second feature extractor alongside SIFT. Nerfstudio's
# colmap_utils was written against the old flag names and fails with
# "unrecognised option" during ns-process-data without this patch.
# Verified still needed against main@50e0e3c.
#
# v3.1: expose port 6006 for TensorBoard (requires matching -p 6006:6006
# in `docker create` to be reachable from host). TensorBoard ships with
# NGC PyTorch and is used via `--vis viewer+tensorboard` in ns-train.
#
# v3.2: patch nerfstudio for PyTorch >=2.6 weights_only=True default. NGC
# PyTorch 25.11 ships torch 2.10, where torch.load defaults to a restricted
# unpickler that rejects numpy.core.multiarray.scalar (which nerfstudio
# checkpoints contain). Pass weights_only=False at the four torch.load call
# sites. Affects ns-render, ns-eval, ns-export, ns-viewer (loading existing
# models), and ns-train --load-dir/--load-checkpoint (resume). Safe because
# checkpoints are produced locally by this same install, not untrusted
# input. Verified still needed against main@50e0e3c.
#
# v4 (v2 fork): install nerfstudio from git main (pinned) instead of PyPI.
# PyPI nerfstudio 1.1.5 doesn't register splatfacto-mcmc; main does.
# Pinning to a specific commit keeps the image reproducible. Bump
# NERFSTUDIO_COMMIT to pull newer upstream changes, then re-run
# smoke_test.sh to confirm the sed patches still find their line patterns.
# ============================================================
FROM gsplat AS nerfstudio

ARG NERFSTUDIO_COMMIT=50e0e3c70c775e89333256213363badbf074f29d
RUN pip install --no-cache-dir \
        "nerfstudio @ git+https://github.com/nerfstudio-project/nerfstudio.git@${NERFSTUDIO_COMMIT}"

RUN sed -i \
        -e 's/SiftExtraction\.use_gpu/FeatureExtraction.use_gpu/g' \
        -e 's/SiftMatching\.use_gpu/FeatureMatching.use_gpu/g' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/process_data/colmap_utils.py

RUN sed -i 's|map_location="cpu")|map_location="cpu", weights_only=False)|g' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/utils/eval_utils.py \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/engine/trainer.py \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/downloads/download_data.py

# splatfacto-w (KevinXu02) is listed as an external method by nerfstudio. It
# registers `splatfacto-w` and `splatfacto-w-light` via the
# nerfstudio.method_configs entry point. We install with --no-deps because its
# setup.py pins gsplat/nerfstudio versions that we've already satisfied (and
# often exceeded) in earlier stages; its real runtime deps (pytorch_msssim,
# pandas, cv2) are already present in the NGC base + nerfstudio install.
ARG SPLATFACTOW_COMMIT=119a3bfb3aa03669278e174ff11c4dfdcbcf97d7
RUN pip install --no-cache-dir --no-deps \
        "splatfacto-w @ git+https://github.com/KevinXu02/splatfacto-w.git@${SPLATFACTOW_COMMIT}"

# v4.1: splatfacto-w-aware .ply exporter. Upstream `ns-export gaussian-splat`
# hard-asserts `isinstance(pipeline.model, SplatfactoModel)`; SplatfactoWModel
# inherits from Model directly, and splatfacto-w stores SH coefficients as an
# MLP over per-image appearance embeddings rather than as trainable tensors.
# This script bakes one chosen embedding (average of training cameras by
# default) into per-gaussian SH coefficients and writes an inria-format .ply.
COPY image_bin/ns-export-splatfactow /usr/local/bin/ns-export-splatfactow
RUN chmod +x /usr/local/bin/ns-export-splatfactow

WORKDIR /workspace
EXPOSE 7007 6006
CMD ["/bin/bash"]

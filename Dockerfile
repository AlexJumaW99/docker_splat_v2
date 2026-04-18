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
# v3 fix: patch nerfstudio 1.1.5 colmap_utils.py for COLMAP 4.1.0.dev0
# flag rename. COLMAP moved --SiftExtraction.use_gpu to
# --FeatureExtraction.use_gpu (and same for SiftMatching) when it added
# ALIKED as a second feature extractor alongside SIFT. Nerfstudio 1.1.5
# was written against the old flag names and fails with "unrecognised
# option" during ns-process-data without this patch.
# ============================================================
FROM gsplat AS nerfstudio

RUN pip install --no-cache-dir nerfstudio

RUN sed -i \
        -e 's/SiftExtraction\.use_gpu/FeatureExtraction.use_gpu/g' \
        -e 's/SiftMatching\.use_gpu/FeatureMatching.use_gpu/g' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/process_data/colmap_utils.py

WORKDIR /workspace
EXPOSE 7007
CMD ["/bin/bash"]

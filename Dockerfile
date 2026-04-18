# syntax=docker/dockerfile:1.6
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt

# ---------- Layer 1: System dependencies ----------
# COLMAP 4.1-dev build deps from the official Ubuntu 24.04 / Qt6 install guide.
# We deliberately skip the `nvidia-cuda-toolkit` apt package because the NGC
# base image already ships CUDA 13.0.2 at /usr/local/cuda -- installing the
# Ubuntu toolkit on top would shadow it.
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
        qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 \
        libcgal-dev \
        libceres-dev \
        libsuitesparse-dev \
        libcurl4-openssl-dev libssl-dev \
        libmkl-full-dev \
    && mkdir -p /usr/include/opencv4 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Layer 2: Build COLMAP from source with Blackwell (sm_120) ----------
# -DCMAKE_CUDA_ARCHITECTURES=120 targets your 5070 Ti exactly.
# This takes 15-40 minutes depending on CPU; it's the slowest layer.
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

# ---------- Layer 3: Install gsplat from source ----------
# --no-build-isolation ensures gsplat compiles against the NGC torch 2.10.0a0
# rather than pulling a mainline torch wheel into an isolated build env.
# TORCH_CUDA_ARCH_LIST and FORCE_CUDA (set as ENV above) tell the setup.py
# to emit sm_120 kernels even though `docker build` has no GPU attached.
RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/nerfstudio-project/gsplat.git

# ---------- Layer 4: Install Nerfstudio ----------
# PIP_CONSTRAINT (set as ENV above) points at NGC's constraints file, which
# pins torch to the prebuilt 2.10.0a0 so pip won't silently replace it.
RUN pip install --no-cache-dir nerfstudio

# ---------- Layer 5: Runtime config ----------
WORKDIR /workspace
EXPOSE 7007
CMD ["/bin/bash"]

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
#
# v4.5: pin COLMAP to a specific upstream commit instead of tracking main
# HEAD. The previous `git clone --depth 1` would always pull whatever was
# at main when the build ran, so two builds done a month apart could ship
# different COLMAP code under the same `4.1.0.dev0` version string.
# Bumping COLMAP_COMMIT below + re-running smoke_test.sh is the new
# upgrade ritual; this matches the convention already used by
# NERFSTUDIO_COMMIT, SPLATFACTOW_COMMIT, and VOCAB_TREE_COMMIT in Stage 5.
#
# `git clone --filter=blob:none` is a partial clone — defers blob downloads
# until checkout, so disk + network usage stays close to a shallow clone
# while still letting us check out an arbitrary SHA (which `--depth 1`
# can't do — shallow clones only contain HEAD). Partial clones are
# supported by GitHub and have been default-on in git since 2.19 (2018).
#
# The resolved commit is recorded at /etc/colmap-commit.sha so it can be
# probed at runtime (`cat /etc/colmap-commit.sha`) and asserted by
# smoke_test.sh. COLMAP's `-h` does include a 7-char abbreviated commit
# (e.g. "Commit ffb5404e on 2026-04-24"), but parsing that and matching
# against a 40-char SHA prefix is fiddly — keeping the full SHA in a
# dedicated file makes the comparison trivial and unambiguous.
# ============================================================
FROM system-deps AS colmap

ARG COLMAP_COMMIT=ffb5404e7bbfb9c718726f16718d7264582fe718

RUN git -c advice.detachedHead=false clone --filter=blob:none \
        https://github.com/colmap/colmap.git /opt/colmap_src \
    && git -C /opt/colmap_src checkout "${COLMAP_COMMIT}" \
    && cd /opt/colmap_src \
    && mkdir build && cd build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DBLA_VENDOR=Intel10_64lp \
    && ninja \
    && ninja install \
    && echo "${COLMAP_COMMIT}" > /etc/colmap-commit.sha \
    && cd / && rm -rf /opt/colmap_src

# ============================================================
# Stage 4: gsplat — pinned to 1.5.3 (PyPI wheel, JIT-compiled at runtime)
#
# v4.6: pin gsplat to 1.5.3 (was: untracked git tip overwritten by
# nerfstudio's gsplat==1.4.0 pin in Stage 5). 1.5.3 unlocks 2DGS,
# fused-SSIM (Taming-3DGS), 3DGUT, Fisheye-GS, and F-Theta camera
# support inside the existing splatfacto methods — all via config
# flags on splatfacto/-big/-mcmc, no new ns-train method names.
#
# Verified backward compatible against the existing splatfacto +
# splatfacto-w callsites: rasterization() in 1.5.3 is a strict
# superset of 1.4.0 (9 new kwargs all default-valued; return tuple
# `(Tensor, Tensor, Dict)` unchanged). gsplat.strategy and
# gsplat.cuda._wrapper.spherical_harmonics keep the same signatures
# splatfacto-w imports. Both callsites use all-kwargs invocations.
#
# Stage 5's nerfstudio install will pip-downgrade gsplat back to its
# pinned 1.4.0 — we re-install 1.5.3 explicitly after that step. This
# stage is kept for docker-layer cache warming of the pip wheel.
# ============================================================
FROM colmap AS gsplat

ARG GSPLAT_VERSION=1.5.3
RUN pip install --no-cache-dir "gsplat==${GSPLAT_VERSION}"

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

# v4.6: nerfstudio 1.1.5's pyproject.toml pins gsplat==1.4.0, so the
# install above silently downgrades the 1.5.3 we put in place in Stage 4.
# Re-install 1.5.3 explicitly to lock the upgrade in. Pip prints a
# non-fatal warning ("nerfstudio 1.1.5 requires gsplat==1.4.0, but you
# have gsplat 1.5.3 which is incompatible") — this is informational
# only and does not abort the build. Verified end-to-end by
# smoke_test.sh (62/62 passing, including #9 rasterization on sm_120)
# and by importing splatfacto, splatfacto-w, and ns-export-splatfactow
# against the upgraded rasterizer.
ARG GSPLAT_VERSION=1.5.3
RUN pip install --no-cache-dir "gsplat==${GSPLAT_VERSION}"

RUN sed -i \
        -e 's/SiftExtraction\.use_gpu/FeatureExtraction.use_gpu/g' \
        -e 's/SiftMatching\.use_gpu/FeatureMatching.use_gpu/g' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/process_data/colmap_utils.py

RUN sed -i 's|map_location="cpu")|map_location="cpu", weights_only=False)|g' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/utils/eval_utils.py \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/engine/trainer.py \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/downloads/download_data.py

# v4.4: filter the internal viewer key `gl_z_buf_depth` out of the output
# dropdowns in the viser viewer. render_state_machine.py injects
# outputs["gl_z_buf_depth"] on every render — a depth channel used *only* for
# 3D browser depth compositing (camera frustums / crop gizmo occluded by the
# gaussian render). For SplatfactoModel it is *downsampled* to 128² in the
# low_move/low_static interaction states, so it has a different shape than the
# rgb/depth/accumulation/background keys returned by the model. The viewer
# passes every key in outputs to update_output_options() and defaults the
# split dropdown to the *last* key, which happens to be gl_z_buf_depth, so
# the first time a user enables "Enable" split the viewer crashes with
#   RuntimeError: Sizes of tensors must match except in dimension 1.
#   Expected size N but got size M for tensor number 1 in the list.
# in _send_output_to_viewer's torch.cat. The documented workaround
# (nerfstudio#3255) is to uncheck "Composite depth" in the viewer — which
# works but disables 3D compositing. No upstream fix exists as of
# main@50e0e3c; grep confirms gl_z_buf_depth is only read at three sites, all
# inside render_state_machine.py (two writers + one reader at line 283 that
# ships it to viser). We keep the key in the outputs dict so depth
# compositing still works, and only exclude it from the dropdown options so
# split mode works regardless of the Composite-depth setting. Both sed
# anchors are unique in the source (verified with `grep -c` == 1 for each);
# py_compile succeeds on the patched file. Verified still needed against
# nerfstudio main@50e0e3c.
RUN sed -i \
        -e 's|output_keys = set(outputs.keys())|output_keys = set(outputs.keys()) - {"gl_z_buf_depth"}|' \
        -e 's|self.viewer.control_panel.update_output_options(list(outputs.keys()))|self.viewer.control_panel.update_output_options([k for k in outputs.keys() if k != "gl_z_buf_depth"])|' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/viewer/render_state_machine.py

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

# v4.3: colmap shim — wraps `colmap mapper` to (a) inject snapshot flags for
# live viser progress and (b) swap the largest submap into sparse/0 when
# COLMAP's incremental_pipeline (4.1.0.dev0) splinters the reconstruction.
# Nerfstudio hardcodes colmap_model_path=Path("colmap/sparse/0"); COLMAP 4.1
# seeds a 2-image bootstrap as sparse/0 before growing the main model into
# sparse/1, so ns-process-data silently exports a 2-frame transforms.json
# while the real reconstruction sits unused. Shim lives at a stable path and
# is prepended to PATH so every caller (ns-process-data, GUI, interactive
# shells) hits it transparently; non-mapper subcommands pass through. See
# shim source for env knobs (COLMAP_SHIM_SNAPSHOT_FREQ, *_PROMOTE_LARGEST).
COPY image_bin/colmap-shim /opt/colmap-shim/colmap
RUN chmod +x /opt/colmap-shim/colmap
ENV PATH=/opt/colmap-shim:$PATH

# v4.2: pre-place a faiss-format vocabulary tree for vocab_tree_matcher.
# In May 2025 COLMAP migrated its vocab_tree_matcher backend from FLANN to
# faiss (colmap/colmap#3486). The legacy file nerfstudio downloads from
# demuc.de/colmap/vocab_tree_flickr100K_words32K.bin is still FLANN-format
# and makes our sm_120 COLMAP build abort with
#   Check failed: file_version == 1 || file_version == 2
#   Failed to read faiss index. ... COLMAP switched from flann to faiss...
# at pairing.cc while indexing the first image, so ns-process-data
# --matching-method vocab_tree never completes (nerfstudio#3664, #3520).
#
# Nerfstudio's get_vocab_tree() only downloads when the target file is
# absent, so pre-placing a valid faiss-format tree at the exact expected
# path short-circuits the broken download without patching nerfstudio.
# File comes from a pinned commit of a community repo whose sole content
# is faiss-converted versions of the official COLMAP vocab trees. SHA-256
# is verified so a silent upstream swap would fail the build.
# Verified end-to-end inside a live container: COLMAP 4.1.0.dev0 loads
# the tree and completes vocab_tree matching on 253 images in ~45s.
ARG VOCAB_TREE_COMMIT=b88921d40767bef9cfd9d67126f58848e7ca59d4
ARG VOCAB_TREE_SHA256=818de625d3ed285fa9a78117700e86e956dd7b64c3efa6ab984b483060f45581
RUN mkdir -p /root/.local/share/nerfstudio \
    && curl -fsSL \
        "https://raw.githubusercontent.com/ZachMckennedyFWig/ColmapFaissVocabTrees/${VOCAB_TREE_COMMIT}/vocab_tree_flickr100K_words32K.bin" \
        -o /root/.local/share/nerfstudio/vocab_tree.fbow \
    && echo "${VOCAB_TREE_SHA256}  /root/.local/share/nerfstudio/vocab_tree.fbow" \
        | sha256sum -c -

WORKDIR /workspace
EXPOSE 7007 6006
CMD ["/bin/bash"]

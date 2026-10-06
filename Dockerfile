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
# v4.9 (2026-10-06): bumped from 4.1.0.dev0 @ffb5404e (main, 2026-04-24) to
# the 4.2.1 release tag @bd1fcf65 (2026-09-29). Validated side by side in a
# throwaway container before adoption: every flag nerfstudio's colmap_utils
# and the shim emit still exists (FeatureExtraction/FeatureMatching.use_gpu,
# Mapper.ba_global_function_tolerance, Mapper.snapshot_*, BundleAdjustment.
# refine_principal_point, VocabTreeMatching.vocab_tree_path); the faiss vocab
# tree loads; sequential / vocab_tree / exhaustive register 87/87 of the same
# 87-frame set in 16 / 28 / 21 s (4.1: 23 / 33 / 35 s) with statistically
# identical models (points within 2 %, reprojection error 0.380 vs 0.385 px);
# the 42-frame equirectangular sample registers 376/376 crops in 282 s (4.1:
# 336 of 375 in 321 s). The 2-3-image bootstrap submap in sparse/0 still
# happens on 4.2.1, so the shim below stays. Rollback: `splat-nerfstudio-v2:pre-v4.9`.
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
# (e.g. "Commit bd1fcf65 on 2026-09-29"), but parsing that and matching
# against a 40-char SHA prefix is fiddly — keeping the full SHA in a
# dedicated file makes the comparison trivial and unambiguous.
# ============================================================
FROM system-deps AS colmap

ARG COLMAP_COMMIT=bd1fcf654d2dd8fefa1466999c190a246f83f4b9

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
# nerfstudio's gsplat==1.4.0 pin in Stage 5). 1.5.3 is the latest gsplat
# release (checked 2026-10-05; 1.6.0 exists only on main). The library
# gains 2DGS, fused-SSIM, 3DGUT, Fisheye-GS and F-Theta cameras, but NOTE
# (corrected 2026-10-05): nerfstudio@50e0e3c's splatfacto never passes
# `camera_model`, `with_ut` or the 2DGS entry point to gsplat — zero
# references in models/splatfacto.py — so none of these are reachable
# through `ns-train splatfacto*` flags. What 1.5.3 buys the splatfacto
# methods here is the newer rasterizer/strategy code paths, not new
# camera models.
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

# v4.8: cuDNN autotuning off by default in ns-train. nerfstudio's train.py
# sets `torch.backends.cudnn.benchmark = True` at import. With that on, the
# FIRST LPIPS evaluation of a run (torchmetrics AlexNet convolutions at full
# eval resolution) makes cuDNN try every convolution algorithm, workspaces
# included, and that single call was the VRAM high-water mark of every run
# measured on 2026-10-05: +3.15 GiB at 960 px, +12.40 GiB at 1920 px (vs
# +0.12 / +0.50 GiB with autotuning off or on the second call). Whole runs:
# peak 11.15 -> 2.42 GiB at 1920 px, 3.72 -> 0.90 GiB at 960 px, with
# identical wall time and identical 4.0 ms train-iteration time (splatfacto's
# only hot convolution is the 11x11 SSIM window). CUDNN_CONV_WSCAP_DBG had
# no effect in this cuDNN build. The patch keeps an escape hatch:
# NS_CUDNN_BENCHMARK=1 restores upstream behaviour. `grep -c` fails the build
# if the anchor line ever changes upstream.
RUN sed -i \
        's|^torch.backends.cudnn.benchmark = True  # type: ignore$|torch.backends.cudnn.benchmark = __import__("os").environ.get("NS_CUDNN_BENCHMARK", "0") == "1"  # docker_splat_v2 v4.8: default off, see Dockerfile|' \
        /usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/train.py \
    && test "$(grep -c 'NS_CUDNN_BENCHMARK' /usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/train.py)" = "1" \
    && test "$(grep -c '^torch.backends.cudnn.benchmark = True' /usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/train.py)" = "0" \
    && python -c "import py_compile; py_compile.compile('/usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/train.py', doraise=True)"

# splatfacto-w (KevinXu02) is listed as an external method by nerfstudio. It
# registers `splatfacto-w` and `splatfacto-w-light` via the
# nerfstudio.method_configs entry point. We install with --no-deps because its
# setup.py pins gsplat/nerfstudio versions that we've already satisfied (and
# often exceeded) in earlier stages; its real runtime deps (pytorch_msssim,
# pandas, cv2) are already present in the NGC base + nerfstudio install.
ARG SPLATFACTOW_COMMIT=119a3bfb3aa03669278e174ff11c4dfdcbcf97d7
RUN pip install --no-cache-dir --no-deps \
        "splatfacto-w @ git+https://github.com/KevinXu02/splatfacto-w.git@${SPLATFACTOW_COMMIT}"

# v4.4: patched NerfW dataparser. Upstream hardcodes phototourism-only paths
# (dense/sparse, dense/images, PINHOLE assertion, mandatory TSV split file).
# Our overlay uses config.colmap_path, auto-detects images dir, accepts
# OPENCV cameras, and falls back to fraction-based eval split when no TSV
# exists — so splatfacto-w works with standard COLMAP/nerfstudio layouts.
#
# v4.7: the v4.4 overlay still built `Cameras` without width/height (nerfstudio
# then derives int(2*cx) x int(2*cy), wrong once COLMAP has refined the
# principal point — 13 of 14 OPENCV datasets in the warehouse), dropped the
# OPENCV distortion coefficients, accepted only PINHOLE/OPENCV, and always read
# full-resolution images/. Symptoms: splatfacto-w-light asserted "The size of
# image ... does not match the camera parameters"; splatfacto-w skipped every
# frame recursively and on 4K data died with a misleading CUDA OOM. v4.7 reads
# intrinsics through nerfstudio's own parse_colmap_camera_params (real size +
# distortion, perspective-family models), rejects fisheye explicitly, and
# picks images_<N> with the same <=1600 px rule as nerfstudio-data. See the
# module docstring for the full upstream delta.
COPY image_bin/nerfw_dataparser.py \
     /usr/local/lib/python3.12/dist-packages/splatfactow/nerfw_dataparser.py

# v4.4b: gsplat 1.5.3 returns radii as [C, N, 2] (per-axis); splatfactow's
# custom densification expects [N]. Collapse with amax so visible_mask
# and max_2Dsize indexing stay consistent.
RUN sed -i \
        's|self\.radii = info\["radii"\]\[0\]  # \[N\]|self.radii = info["radii"][0].amax(dim=-1)  # [N] — gsplat >=1.5 returns [N,2]|' \
        /usr/local/lib/python3.12/dist-packages/splatfactow/splatfactow_model.py

# v4.7: splatfacto-w eval colour-cache guard (upstream KevinXu02/splatfacto-w
# issue #14, open since 2024-08). SplatfactoWModel.get_outputs() reuses
# `cached_colors` in eval whenever the eval camera index equals the last
# *training* camera index, without checking that the gaussian count is still
# the same. Refinement (densify/cull) runs every 100 steps, in-training eval
# also runs every 100 steps, so whenever the randomly chosen eval image
# happens to share the index of the last training image the cache is one
# densification stale and gsplat asserts:
#   AssertionError: torch.Size([112061, 16, 3])
# Reproduced 2026-10-05: 1 of 2 identical 7k runs died at step ~2300; with 12
# training images it dies at step ~1590 every time. The patch (3 lines) lets
# only eval renders populate cached_colors / cached_bg_sh and requires the
# cached tensor to match the live gaussian count. Applied with `patch` rather
# than sed so the build FAILS if upstream drifts instead of silently
# no-op'ing; `--forward` makes a second application a hard error too.
COPY image_bin/splatfactow_model_cache.patch /opt/patches/splatfactow_model_cache.patch
RUN patch --forward --batch \
        /usr/local/lib/python3.12/dist-packages/splatfactow/splatfactow_model.py \
        /opt/patches/splatfactow_model_cache.patch \
    && test "$(grep -c 'self.cached_colors = None if self.training else colors' \
        /usr/local/lib/python3.12/dist-packages/splatfactow/splatfactow_model.py)" = "1" \
    && python -c "import py_compile; py_compile.compile('/usr/local/lib/python3.12/dist-packages/splatfactow/splatfactow_model.py', doraise=True)"

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
# COLMAP's incremental_pipeline (4.1.0.dev0, still 4.2.1) splinters the reconstruction.
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

# v4.7: bake hloc (Hierarchical-Localization) into the image so
# `ns-process-data ... --sfm-tool hloc --feature-type superpoint_max
# --matcher-type superpoint+lightglue` survives a container rebuild. Until
# this step the toolbox lived only in the running container's writable
# layer (installed by hand on 2026-09-20; visible via `docker diff`), so any
# `docker compose up --build` silently removed the learned-feature route the
# DRC datasets were processed with. Pins mirror exactly what that container
# had: hloc @ HLOC_COMMIT as an editable install (hloc resolves its model
# code relative to /opt/hloc), only the SuperGluePretrainedNetwork submodule
# (ships the SuperPoint/SuperGlue weights; r2d2 / d2net / deep-image-retrieval
# were never checked out, so nerfstudio's `r2d2` / `d2net-ss` feature types
# are NOT available here), LightGlue @ LIGHTGLUE_COMMIT, pycolmap 3.13.0 as
# hloc's SfM backend (independent of the system COLMAP binary), kornia 0.8.3
# (+ kornia_rs 0.1.14; `adalam` needs a kornia API that 0.8.3 no longer has).
# `--no-deps` for hloc because its remaining requirements (h5py, gdown,
# plotly, matplotlib, scipy, tqdm, opencv-python) are already in the NGC base
# + nerfstudio install — verified with `pip check` on 2026-10-05. Runtime
# weights (LightGlue, NetVLAD) download on first use into the torch-hub cache
# volume. Licence note: SuperPoint/SuperGlue weights are "academic or
# non-profit organization noncommercial research use only" (Magic Leap);
# `disk+lightglue` is the permissively licensed combination.
ARG HLOC_COMMIT=c13273bd0ecc2917a35910fd843712a1c6243193
ARG LIGHTGLUE_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8
RUN git -c advice.detachedHead=false clone \
        https://github.com/cvg/Hierarchical-Localization.git /opt/hloc \
    && git -C /opt/hloc checkout "${HLOC_COMMIT}" \
    && git -C /opt/hloc submodule update --init third_party/SuperGluePretrainedNetwork \
    && pip install --no-cache-dir \
        pycolmap==3.13.0 kornia==0.8.3 kornia_rs==0.1.14 \
        "lightglue @ git+https://github.com/cvg/LightGlue.git@${LIGHTGLUE_COMMIT}" \
    && pip install --no-cache-dir --no-deps -e /opt/hloc \
    && python -c "import hloc, pycolmap, lightglue, kornia; from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction; import nerfstudio.process_data.hloc_utils; print('hloc', hloc.__version__, 'pycolmap', pycolmap.__version__, 'kornia', kornia.__version__)"

WORKDIR /workspace
EXPOSE 7007 6006
CMD ["/bin/bash"]

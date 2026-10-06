#!/usr/bin/env bash
# smoke_test.sh — splat-ns-v2 container smoke tests
# Verifies every feature built by the Dockerfile: GPU passthrough, PyTorch +
# sm_120, cuDNN, COLMAP (with 4.1 flag rename), ffmpeg, gsplat rasterizer,
# nerfstudio CLI, the colmap_utils.py patch, TensorBoard/viser stack,
# the splatfacto-w external method, workspace bind mount, and env vars.
#
# Usage:
#   ./smoke_test.sh                                   # all tests
#   ./smoke_test.sh --skip-jit                        # skip the slow gsplat JIT compile
#   ./smoke_test.sh --skip-train                      # skip §21 (5 short training runs, ~2 min)
#   CONTAINER=foo IMAGE=foo:latest ./smoke_test.sh    # override defaults
#
# Exits non-zero if any test fails. Never stops on failure — runs every
# test so you see the full picture.

set -u

CONTAINER="${CONTAINER:-splat-ns-v2}"
IMAGE="${IMAGE:-splat-nerfstudio-v2:latest}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/home/meroe/docker-stacks/docker_splat_v2/data}"
SKIP_JIT=0
SKIP_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --skip-jit) SKIP_JIT=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    esac
done

if [ -t 1 ]; then
    G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; D=$'\e[2m'; N=$'\e[0m'
else
    G=; R=; Y=; B=; D=; N=
fi

PASS=0; FAIL=0; WARN=0
FAILED=()

section() { printf '\n%s=== %s ===%s\n' "$B" "$*" "$N"; }
pass()    { printf '  %sPASS%s  %s\n' "$G" "$N" "$1"; PASS=$((PASS+1)); }
fail()    { printf '  %sFAIL%s  %s\n' "$R" "$N" "$1"
            [ $# -gt 1 ] && printf '        %s%s%s\n' "$D" "$2" "$N"
            FAIL=$((FAIL+1)); FAILED+=("$1"); }
warn()    { printf '  %sWARN%s  %s\n' "$Y" "$N" "$1"
            [ $# -gt 1 ] && printf '        %s%s%s\n' "$D" "$2" "$N"
            WARN=$((WARN+1)); }
info()    { printf '        %s%s%s\n' "$D" "$1" "$N"; }

# Run a command in the container and capture stdout+stderr.
dexec()    { docker exec    "$CONTAINER" "$@" 2>&1; }
dexec_in() { docker exec -i "$CONTAINER" "$@" 2>&1; }

printf '%sSmoke test for %s (image: %s)%s\n' "$B" "$CONTAINER" "$IMAGE" "$N"
printf '%sStarted: %s%s\n' "$D" "$(date -Iseconds)" "$N"

# ============================================================================
# 1. Container runtime + GPU passthrough + published ports
# ============================================================================
section "1. Container runtime + GPU passthrough"

if ! docker ps --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    fail "$CONTAINER is running" "start it with: docker start $CONTAINER"
    printf '\n%sAborting remaining tests — container is not running.%s\n' "$R" "$N"
    exit 1
fi
pass "$CONTAINER is running"
info "$(docker ps --filter "name=^${CONTAINER}$" --format 'status={{.Status}}')"

nvsmi=$(dexec nvidia-smi)
if [ $? -eq 0 ] && echo "$nvsmi" | grep -q "NVIDIA-SMI"; then
    pass "nvidia-smi inside container"
    info "$(echo "$nvsmi" | grep -E '(Driver Version|CUDA Version)' | head -1 | sed 's/^ *//')"
    info "$(echo "$nvsmi" | grep -E 'NVIDIA .+On' | head -1 | awk -F'|' '{print $2}' | sed 's/^ *//;s/ *$//')"
else
    fail "nvidia-smi inside container" "$(echo "$nvsmi" | head -3)"
fi

ports=$(docker port "$CONTAINER" 2>&1)
echo "$ports" | grep -q '6006/tcp -> 0.0.0.0:6006' \
    && pass "port 6006 published (TensorBoard)" \
    || fail "port 6006 published" "got: $(echo "$ports" | grep 6006 || echo none)"
echo "$ports" | grep -q '7007/tcp -> 0.0.0.0:7007' \
    && pass "port 7007 published (nerfstudio viewer)" \
    || fail "port 7007 published" "got: $(echo "$ports" | grep 7007 || echo none)"

# ============================================================================
# 2. Dockerfile env vars
# ============================================================================
section "2. Dockerfile env vars (TORCH_CUDA_ARCH_LIST=12.0, FORCE_CUDA=1)"

envs=$(dexec sh -c 'echo "$TORCH_CUDA_ARCH_LIST|$FORCE_CUDA|$CUDA_HOME|$PIP_CONSTRAINT"')
IFS='|' read -r arch force_cuda cuda_home pip_con <<< "$envs"
[ "$arch"      = "12.0" ]                && pass "TORCH_CUDA_ARCH_LIST=12.0"      || fail "TORCH_CUDA_ARCH_LIST" "got '$arch'"
[ "$force_cuda" = "1" ]                  && pass "FORCE_CUDA=1"                   || fail "FORCE_CUDA" "got '$force_cuda'"
[ "$cuda_home" = "/usr/local/cuda" ]     && pass "CUDA_HOME=/usr/local/cuda"      || fail "CUDA_HOME" "got '$cuda_home'"
[ "$pip_con"   = "/etc/pip/constraint.txt" ] && pass "PIP_CONSTRAINT set"         || warn "PIP_CONSTRAINT" "got '$pip_con'"

# ============================================================================
# 3. PyTorch + CUDA + sm_120 Blackwell
# ============================================================================
section "3. PyTorch + CUDA + sm_120"

out=$(dexec python -c "import torch,json; print(json.dumps({'v':torch.__version__,'cuda':torch.cuda.is_available(),'n':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cap':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,'archs':torch.cuda.get_arch_list()}))")
if echo "$out" | grep -q '"cuda": true'; then
    pass "torch.cuda.is_available() == True"
    info "$out"
else
    fail "torch + CUDA" "$out"
fi
echo "$out" | grep -q '"cap": \[12, 0\]'        && pass "GPU capability (12,0) — Blackwell sm_120" || fail "GPU capability (12,0)" "$(echo "$out" | grep -o '"cap":[^,]*')"
echo "$out" | grep -Eq '"sm_120"'               && pass "sm_120 in torch arch list"                 || warn "sm_120 in torch arch list" "arch list: $(echo "$out" | grep -o '"archs":[^]]*]')"

# ============================================================================
# 4. Live GPU compute (catches broken kernels even if import works)
# ============================================================================
section "4. Live GPU compute (matmul + conv)"

out=$(dexec python -c "import torch; x=torch.randn(4096,4096,device='cuda'); y=(x@x.T).sum().item(); torch.cuda.synchronize(); print('OK', y)")
echo "$out" | grep -q '^OK ' \
    && pass "4096x4096 matmul on GPU" \
    || fail "GPU matmul" "$(echo "$out" | tail -3)"

out=$(dexec python -c "import torch,torch.nn.functional as F; x=torch.randn(1,3,64,64,device='cuda'); w=torch.randn(8,3,3,3,device='cuda'); y=F.conv2d(x,w); torch.cuda.synchronize(); print('OK', tuple(y.shape))")
echo "$out" | grep -q '^OK ' \
    && pass "conv2d on GPU (cuDNN path)" \
    || fail "cuDNN conv" "$(echo "$out" | tail -3)"

# ============================================================================
# 5. cuDNN
# ============================================================================
section "5. cuDNN"

out=$(dexec python -c "import torch; print('ver',torch.backends.cudnn.version(),'avail',torch.backends.cudnn.is_available())")
echo "$out" | grep -q 'avail True' \
    && { pass "cuDNN available"; info "$out"; } \
    || fail "cuDNN available" "$out"

# ============================================================================
# 6. COLMAP (built from source with sm_120) + the 4.1 flag
# ============================================================================
section "6. COLMAP (sm_120) + 4.1 FeatureExtraction flag"

ver=$(dexec colmap -h 2>&1 | head -1)
echo "$ver" | grep -qi 'colmap' \
    && { pass "colmap CLI present"; info "$ver"; } \
    || fail "colmap CLI present" "$ver"

help=$(dexec colmap feature_extractor --help)
echo "$help" | grep -q '\-\-FeatureExtraction.use_gpu' \
    && pass "--FeatureExtraction.use_gpu flag (COLMAP 4.1 new name)" \
    || fail "--FeatureExtraction.use_gpu flag" "not found in feature_extractor help"

echo "$help" | grep -q '\-\-SiftExtraction.use_gpu' \
    && warn "old --SiftExtraction.use_gpu still present" "unexpected — should have been replaced" \
    || pass "old --SiftExtraction.use_gpu flag is gone"

# Verify the pinned COLMAP commit was baked into the image and matches the
# Dockerfile's ARG default. COLMAP's `-h` does print a 7-char abbreviated
# commit (`Commit ffb5404e on YYYY-MM-DD`), but matching that against a
# 40-char SHA prefix is fiddly. /etc/colmap-commit.sha stores the full SHA
# and makes the comparison trivial. Drift here means someone bumped
# COLMAP_COMMIT in the Dockerfile but forgot to rebuild — or built with
# --build-arg COLMAP_COMMIT=other, in which case the smoke test correctly
# flags that the canonical pin no longer describes what's running.
sha_file='/etc/colmap-commit.sha'
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dockerfile="${script_dir}/Dockerfile"

if ! dexec test -f "$sha_file"; then
    fail "$sha_file present in image" \
        "file missing — image pre-dates the COLMAP_COMMIT pin? rebuild with current Dockerfile"
else
    actual_sha=$(dexec cat "$sha_file" | tr -d '[:space:]')
    if [[ "$actual_sha" =~ ^[0-9a-f]{40}$ ]]; then
        pass "$sha_file holds a 40-char hex SHA"
        info "image:    $actual_sha"
    else
        fail "$sha_file holds a 40-char hex SHA" "got: '$actual_sha'"
    fi

    if [ -f "$dockerfile" ]; then
        expected_sha=$(grep -E '^ARG[[:space:]]+COLMAP_COMMIT=' "$dockerfile" \
            | head -1 | cut -d= -f2- | tr -d '[:space:]')
    else
        expected_sha=""
    fi

    if [ -n "$expected_sha" ]; then
        info "Dockerfile: $expected_sha"
        [ "$actual_sha" = "$expected_sha" ] \
            && pass "image COLMAP commit matches Dockerfile pin" \
            || fail "image COLMAP commit matches Dockerfile pin" \
                    "drift detected — rebuild with: docker compose build --no-cache nerfstudio"
    else
        warn "could not read expected COLMAP_COMMIT from Dockerfile" \
            "Dockerfile not found at $dockerfile, or no 'ARG COLMAP_COMMIT=' line"
    fi
fi

# ============================================================================
# 7. ffmpeg
# ============================================================================
section "7. ffmpeg"

fv=$(dexec ffmpeg -version 2>&1 | head -1)
echo "$fv" | grep -q '^ffmpeg version' \
    && { pass "ffmpeg installed"; info "$fv"; } \
    || fail "ffmpeg installed" "$fv"

hw=$(dexec ffmpeg -hide_banner -hwaccels 2>&1)
echo "$hw" | grep -q 'cuda' \
    && pass "ffmpeg has cuda hwaccel" \
    || warn "ffmpeg cuda hwaccel" "missing; ns-process-data will still work on CPU"

# ============================================================================
# 8. gsplat
# ============================================================================
section "8. gsplat rasterizer"

out=$(dexec python -c "import gsplat; print('gsplat', gsplat.__version__)")
echo "$out" | grep -q '^gsplat ' \
    && { pass "gsplat import"; info "$out"; } \
    || fail "gsplat import" "$out"

out=$(dexec python -c "from gsplat.rendering import rasterization; print('rasterization OK')")
echo "$out" | grep -q 'rasterization OK' \
    && pass "gsplat.rendering.rasterization import" \
    || fail "gsplat.rendering.rasterization import" "$out"

# ============================================================================
# 9. gsplat JIT kernel (real rasterize on sm_120) — SLOW FIRST RUN
# ============================================================================
if [ "$SKIP_JIT" -eq 1 ]; then
    section "9. gsplat JIT rasterize (skipped --skip-jit)"
else
    section "9. gsplat JIT rasterize on GPU (first run may take ~30s to compile)"
    out=$(dexec_in python - <<'PY'
import torch
from gsplat.rendering import rasterization
N = 100; device = 'cuda'
means  = torch.randn(N, 3, device=device)
quats  = torch.randn(N, 4, device=device); quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=device) * 0.1
opacs  = torch.rand(N, device=device)
colors = torch.rand(N, 3, device=device)
K    = torch.tensor([[[400, 0, 128], [0, 400, 128], [0, 0, 1]]], device=device, dtype=torch.float32)
view = torch.eye(4, device=device)[None]
img, alpha, _ = rasterization(means, quats, scales, opacs, colors, view, K, 256, 256)
torch.cuda.synchronize()
print('OK', tuple(img.shape), tuple(alpha.shape))
PY
)
    echo "$out" | grep -q '^OK ' \
        && { pass "gsplat rasterize on GPU"; info "$(echo "$out" | tail -1)"; } \
        || fail "gsplat rasterize on GPU" "$(echo "$out" | tail -5)"
fi

# ============================================================================
# 10. nerfstudio CLI
# ============================================================================
section "10. nerfstudio CLI"

# Import nerfstudio and probe its installed version via importlib.metadata.
# Git-pinned nerfstudio main doesn't expose a `__version__` attribute, so we
# can't use `nerfstudio.__version__` like we could with the PyPI 1.1.5 build.
out=$(dexec python -c "import nerfstudio; from importlib.metadata import version; print('nerfstudio', version('nerfstudio'))")
echo "$out" | grep -q '^nerfstudio ' \
    && { pass "nerfstudio import"; info "$out"; } \
    || fail "nerfstudio import" "$out"

for cmd in ns-train ns-process-data ns-viewer ns-export ns-render; do
    if dexec which "$cmd" > /dev/null 2>&1; then
        pass "$cmd on PATH"
    else
        fail "$cmd on PATH"
    fi
done

# ns-train --help triggers full config loading — catches import errors that
# the plain `import nerfstudio` won't surface.
out=$(dexec ns-train --help 2>&1)
echo "$out" | grep -qi 'usage' \
    && pass "ns-train --help (loads all method configs)" \
    || fail "ns-train --help" "$(echo "$out" | tail -5)"

# Check registered methods. `all_methods` is the merged dict of built-ins +
# external entry-point methods; `method_configs` is built-ins only, so use
# `all_methods` here to cover splatfacto-w as well.
out=$(dexec python -c "from nerfstudio.configs.method_configs import all_methods; print(sorted([k for k in all_methods if 'splat' in k]))")
echo "$out" | grep -q "splatfacto'" \
    && { pass "splatfacto method registered"; info "$out"; } \
    || fail "splatfacto method registered" "$out"

# splatfacto-mcmc needs the git-pinned nerfstudio main (Stage 5); PyPI 1.1.5
# does not ship it. If it's missing, the Dockerfile's NERFSTUDIO_COMMIT drifted.
echo "$out" | grep -q "splatfacto-mcmc" \
    && pass "splatfacto-mcmc method registered (git-pinned nerfstudio)" \
    || fail "splatfacto-mcmc method registered" "not in all_methods — check NERFSTUDIO_COMMIT"

# splatfacto-w (external, KevinXu02) registers splatfacto-w and
# splatfacto-w-light via the nerfstudio.method_configs entry point.
echo "$out" | grep -q "splatfacto-w'" \
    && pass "splatfacto-w method registered (KevinXu02 external)" \
    || fail "splatfacto-w method registered" "not in all_methods — check SPLATFACTOW_COMMIT"
echo "$out" | grep -q "splatfacto-w-light" \
    && pass "splatfacto-w-light method registered" \
    || fail "splatfacto-w-light method registered" "not in all_methods"

# Smoke-import the splatfacto-w package itself (catches runtime import failures
# that the method_configs registry won't surface until training starts).
out=$(dexec python -c "from splatfactow.splatfactow_config import splatfactow_config, splatfactow_light_config; print(splatfactow_config.config.method_name, splatfactow_light_config.config.method_name)")
echo "$out" | grep -q 'splatfacto-w splatfacto-w-light' \
    && { pass "splatfactow config objects importable"; info "$out"; } \
    || fail "splatfactow config import" "$out"

# ============================================================================
# 11. Verify Stage 5 sed patch to colmap_utils.py
# ============================================================================
section "11. colmap_utils.py patch (Stage 5 sed)"

pyf='/usr/local/lib/python3.12/dist-packages/nerfstudio/process_data/colmap_utils.py'

if ! dexec test -f "$pyf"; then
    fail "colmap_utils.py exists at $pyf" "file missing — nerfstudio version may have moved it"
else
    pass "colmap_utils.py exists"

    new_count=$(dexec grep -c -E 'FeatureExtraction\.use_gpu|FeatureMatching\.use_gpu' "$pyf" | tr -d '[:space:]')
    old_count=$(dexec grep -c -E 'SiftExtraction\.use_gpu|SiftMatching\.use_gpu'       "$pyf" | tr -d '[:space:]')

    [ "${new_count:-0}" -ge 1 ] \
        && pass "new FeatureExtraction/FeatureMatching.use_gpu present ($new_count refs)" \
        || fail "new flag names present" "patch did not apply"

    [ "${old_count:-0}" = "0" ] \
        && pass "old SiftExtraction/SiftMatching.use_gpu gone" \
        || fail "old flag names removed" "$old_count old refs still present — patch is partial"
fi

# ============================================================================
# 12. TensorBoard + viser (viewer stack)
# ============================================================================
section "12. TensorBoard + viser"

out=$(dexec python -c "import tensorboard; print('tensorboard', tensorboard.__version__)")
echo "$out" | grep -q '^tensorboard ' \
    && { pass "tensorboard import"; info "$out"; } \
    || fail "tensorboard import" "$out"

out=$(dexec python -c "import viser; print('viser', viser.__version__)")
echo "$out" | grep -q '^viser ' \
    && { pass "viser import"; info "$out"; } \
    || fail "viser import" "$out"

dexec which tensorboard > /dev/null 2>&1 \
    && pass "tensorboard binary on PATH" \
    || fail "tensorboard binary on PATH"

# ============================================================================
# 13. Workspace bind mount (RW)
# ============================================================================
section "13. Workspace bind mount (/workspace)"

if dexec test -d /workspace; then
    pass "/workspace exists inside container"
else
    fail "/workspace exists" "directory missing"
fi

token="smoke_$(date +%s)_$$"
wr_out=$(dexec sh -c "touch /workspace/.${token} && rm /workspace/.${token}" 2>&1)
if [ $? -eq 0 ]; then
    pass "/workspace is writable"
else
    fail "/workspace writable" "$wr_out"
fi

# Verify the mount goes back to the host path we expect.
mount_src=$(docker inspect "$CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}')
if [ -n "$mount_src" ]; then
    pass "/workspace is a bind mount"
    info "host path: $mount_src"
    [ "$mount_src" = "$HOST_DATA_DIR" ] \
        && pass "mount source == $HOST_DATA_DIR" \
        || warn "mount source != $HOST_DATA_DIR" "got $mount_src"
else
    fail "/workspace bind mount" "no bind mount for /workspace in docker inspect"
fi

# ============================================================================
# 14. Verify Stage 5 sed patch for torch.load weights_only=False (v3.2)
# ============================================================================
section "14. torch.load weights_only=False patch (Stage 5 sed, v3.2)"

wo_files=(
    '/usr/local/lib/python3.12/dist-packages/nerfstudio/utils/eval_utils.py'
    '/usr/local/lib/python3.12/dist-packages/nerfstudio/engine/trainer.py'
    '/usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/downloads/download_data.py'
)

for f in "${wo_files[@]}"; do
    base=$(basename "$f")
    if ! dexec test -f "$f"; then
        fail "$base exists" "file missing — nerfstudio version may have moved it"
        continue
    fi
    patched=$(dexec grep -c 'torch\.load([^)]*weights_only=False'   "$f" | tr -d '[:space:]')
    stale=$(  dexec grep -c 'torch\.load([^)]*map_location="cpu")$' "$f" | tr -d '[:space:]')
    [ "${patched:-0}" -ge 1 ] \
        && pass "$base has weights_only=False torch.load ($patched ref(s))" \
        || fail "$base weights_only patch applied" "no patched torch.load found"
    [ "${stale:-0}" = "0" ] \
        && pass "$base has no unpatched torch.load(..., map_location=\"cpu\")" \
        || fail "$base unpatched torch.load remains" "$stale ref(s) without weights_only"
done

# ============================================================================
# 15. ns-export-splatfactow (custom PLY exporter for splatfacto-w/-light)
# ============================================================================
section "15. ns-export-splatfactow CLI"

if dexec which ns-export-splatfactow > /dev/null 2>&1; then
    pass "ns-export-splatfactow on PATH"
else
    fail "ns-export-splatfactow on PATH" "custom exporter missing — was image_bin/ copied in the Dockerfile?"
fi

# --help loads the script + its imports (nerfstudio.scripts.exporter,
# splatfactow.splatfactow_model). Catches import-time regressions before
# anyone tries a real export.
help_out=$(dexec ns-export-splatfactow --help 2>&1)
echo "$help_out" | grep -q 'ns-export-splatfactow \[-h\]' \
    && pass "ns-export-splatfactow --help loads (imports clean)" \
    || fail "ns-export-splatfactow --help" "$(echo "$help_out" | tail -5)"

echo "$help_out" | grep -q '\-\-appearance-mode' \
    && pass "ns-export-splatfactow exposes --appearance-mode" \
    || fail "ns-export-splatfactow --appearance-mode flag" "flag missing from --help"

# ============================================================================
# 16. Viewer split-mode patch (Stage 5 sed, v4.4)
#
# The patch filters gl_z_buf_depth out of the output dropdowns in the viser
# viewer so split mode doesn't crash when that internal key is defaulted as
# the split output at a mismatched resolution. We verify:
#   - file still present at its expected path
#   - both patched anchors present (1 ref each)
#   - both unpatched anchors absent (0 refs each)
#   - gl_z_buf_depth is still referenced for 3D compositing (line 283 reader
#     plus the two writers at lines 188/199 — 3 refs total) so depth
#     compositing itself still works
#   - the patched file still imports cleanly (py_compile)
# ============================================================================
section "16. Viewer split-mode patch (v4.4)"

rsm='/usr/local/lib/python3.12/dist-packages/nerfstudio/viewer/render_state_machine.py'

if ! dexec test -f "$rsm"; then
    fail "render_state_machine.py exists at $rsm" "file missing — nerfstudio version may have moved it"
else
    pass "render_state_machine.py exists"

    new_set=$(dexec grep -c -F 'set(outputs.keys()) - {"gl_z_buf_depth"}'                      "$rsm" | tr -d '[:space:]')
    new_list=$(dexec grep -c -F '[k for k in outputs.keys() if k != "gl_z_buf_depth"]'         "$rsm" | tr -d '[:space:]')
    old_set=$(dexec  grep -c -E 'output_keys = set\(outputs\.keys\(\)\)$'                      "$rsm" | tr -d '[:space:]')
    old_list=$(dexec grep -c -F 'update_output_options(list(outputs.keys()))'                  "$rsm" | tr -d '[:space:]')
    ref_total=$(dexec grep -c -F 'gl_z_buf_depth'                                              "$rsm" | tr -d '[:space:]')

    [ "${new_set:-0}"  = "1" ] && pass "set(...) difference patch present"                     || fail "set difference patch applied" "got $new_set"
    [ "${new_list:-0}" = "1" ] && pass "list-comprehension patch present"                      || fail "list-comprehension patch applied" "got $new_list"
    [ "${old_set:-0}"  = "0" ] && pass "unpatched 'set(outputs.keys())' absent"                || fail "unpatched set form removed" "$old_set ref(s) remain"
    [ "${old_list:-0}" = "0" ] && pass "unpatched 'update_output_options(list(...))' absent"   || fail "unpatched list form removed" "$old_list ref(s) remain"
    [ "${ref_total:-0}" -ge 3 ] && { pass "gl_z_buf_depth still referenced ($ref_total refs; depth compositing intact)"; info "expected >= 3: 2 writers + 1 reader + our 2 filter mentions"; } \
                                || fail "gl_z_buf_depth referenced" "only $ref_total ref(s); depth compositing may be broken"

    py_ok=$(dexec python -c "import py_compile; py_compile.compile('$rsm', doraise=True); print('OK')" 2>&1)
    [ "$py_ok" = "OK" ] && pass "render_state_machine.py py_compile clean" || fail "render_state_machine.py py_compile" "$py_ok"
fi

# ============================================================================
# 17. splatfacto-w v4.7 patches (dataparser v2 overlay + eval colour-cache guard)
#
# Both fixes live in splatfactow/ inside the image:
#   - nerfw_dataparser.py is a COPY overlay. We check the v2 marker (it routes
#     intrinsics through nerfstudio's parse_colmap_camera_params and honours
#     images_<N>), and that its public symbol still imports.
#   - splatfactow_model.py gets `patch`-applied anchors. We check all three
#     anchors are present exactly once, the unpatched line is gone, the radii
#     fix from v4.4b is still there, and the file compiles.
#   - Functional probe (CPU only, no dataset needed): build a 2-camera
#     Cameras object the way the v2 parser does and confirm nerfstudio keeps
#     the real width/height instead of deriving int(2*cx) x int(2*cy).
# ============================================================================
section "17. splatfacto-w v4.7 patches"

sfw='/usr/local/lib/python3.12/dist-packages/splatfactow'

v2_marker=$(dexec grep -c -F 'parse_colmap_camera_params(cam)' "$sfw/nerfw_dataparser.py" | tr -d '[:space:]')
[ "${v2_marker:-0}" -ge 1 ] \
    && pass "nerfw_dataparser.py v2 overlay present (uses parse_colmap_camera_params)" \
    || fail "nerfw_dataparser.py v2 overlay" "marker missing — image_bin/nerfw_dataparser.py not COPY'd?"

v2_old=$(dexec grep -c -F '_SUPPORTED_CAMERA_MODELS' "$sfw/nerfw_dataparser.py" | tr -d '[:space:]')
[ "${v2_old:-0}" = "0" ] \
    && pass "v4.4 PINHOLE/OPENCV-only allow-list gone" \
    || fail "v4.4 allow-list removed" "$v2_old ref(s) remain — stale overlay"

out=$(dexec python -c "from splatfactow.nerfw_dataparser import splatfactow_dataparser, NerfWDataParserConfig; print(type(splatfactow_dataparser.config).__name__)" 2>&1 | tail -1)
[ "$out" = "NerfWDataParserConfig" ] \
    && pass "splatfactow_dataparser spec imports" \
    || fail "splatfactow_dataparser import" "$out"

c_guard=$(dexec grep -c -F 'self.cached_colors.shape[0] == appearance_features.shape[0]' "$sfw/splatfactow_model.py" | tr -d '[:space:]')
c_train=$(dexec grep -c -F 'self.cached_colors = None if self.training else colors'           "$sfw/splatfactow_model.py" | tr -d '[:space:]')
c_bg=$(dexec    grep -c -F 'self.cached_bg_sh = None if self.training else bg_sh_coeffs'      "$sfw/splatfactow_model.py" | tr -d '[:space:]')
c_old=$(dexec   grep -c -E '^\s*self\.cached_colors = colors$'                                 "$sfw/splatfactow_model.py" | tr -d '[:space:]')
c_radii=$(dexec grep -c -F 'info["radii"][0].amax(dim=-1)'                                     "$sfw/splatfactow_model.py" | tr -d '[:space:]')

[ "${c_guard:-0}" = "1" ] && pass "cache guard: gaussian-count check present"         || fail "cache guard count check" "got $c_guard"
[ "${c_train:-0}" = "1" ] && pass "cache guard: training renders don't cache colours"  || fail "cache guard colours" "got $c_train"
[ "${c_bg:-0}"    = "1" ] && pass "cache guard: training renders don't cache bg SH"    || fail "cache guard bg SH" "got $c_bg"
[ "${c_old:-0}"   = "0" ] && pass "unpatched 'self.cached_colors = colors' absent"     || fail "unpatched cache line removed" "$c_old ref(s) remain"
[ "${c_radii:-0}" = "1" ] && pass "v4.4b radii fix still present"                     || fail "radii fix present" "got $c_radii"

py_ok=$(dexec python -c "import py_compile; py_compile.compile('$sfw/splatfactow_model.py', doraise=True); py_compile.compile('$sfw/nerfw_dataparser.py', doraise=True); print('OK')" 2>&1 | tail -1)
[ "$py_ok" = "OK" ] && pass "splatfactow patched files py_compile clean" || fail "splatfactow py_compile" "$py_ok"

# Functional probe: real size must survive (3840x2160 with cx=1919.5 would have
# become 3839x2160 under the v4.4 overlay).
out=$(dexec_in python - <<'PY' 2>&1 | tail -1
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils
cams = Cameras(camera_to_worlds=torch.eye(4)[None, :3, :4].repeat(2, 1, 1),
               fx=torch.tensor([3000., 3000.]), fy=torch.tensor([3000., 3000.]),
               cx=torch.tensor([1919.5, 1921.9]), cy=torch.tensor([1085.4, 1079.1]),
               height=torch.tensor([2160, 2160]), width=torch.tensor([3840, 3840]),
               distortion_params=torch.stack([camera_utils.get_distortion_params(k1=-0.1, k2=0.01)] * 2),
               camera_type=CameraType.PERSPECTIVE)
cams.rescale_output_resolution(1.0 / 4)
print(int(cams.width[0]), int(cams.height[0]), int(cams.width[1]), int(cams.height[1]), bool(cams.distortion_params[0, 0] != 0))
PY
)
[ "$out" = "960 540 960 540 True" ] \
    && pass "Cameras keep explicit width/height + distortion through rescale ($out)" \
    || fail "Cameras size/distortion probe" "$out"

# ============================================================================
# 18. hloc toolbox (v4.7) — learned features for `--sfm-tool hloc`
#
# hloc must be importable from the image itself (not a container-layer
# install), together with the pieces nerfstudio's hloc_utils imports and the
# extractor/matcher modules behind the feature/matcher types that are known
# to work in this build. r2d2 / d2net-ss / adalam are documented as
# unavailable (submodules not checked out; kornia API), so they are not
# asserted here.
# ============================================================================
section "18. hloc toolbox (v4.7)"

out=$(dexec python -c "import hloc, pycolmap, lightglue, kornia; print(hloc.__version__, pycolmap.__version__, kornia.__version__)" 2>&1 | tail -1)
echo "$out" | grep -q -E '^1\.5 3\.13\.0 0\.8\.3$' \
    && pass "hloc 1.5 + pycolmap 3.13.0 + kornia 0.8.3 importable" \
    || fail "hloc / pycolmap / kornia import" "$out"

hloc_dir=$(dexec python -c "import hloc, os; print(os.path.dirname(hloc.__file__))" 2>&1 | tail -1)
[ "$hloc_dir" = "/opt/hloc/hloc" ] \
    && pass "hloc resolves to the baked /opt/hloc checkout" \
    || fail "hloc location" "$hloc_dir"

out=$(dexec_in python - <<'PY' 2>&1 | tail -1
import importlib
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction
import nerfstudio.process_data.hloc_utils
mods = [("extractors", "superpoint"), ("extractors", "disk"), ("extractors", "dog"), ("extractors", "netvlad"),
        ("matchers", "lightglue"), ("matchers", "superglue"), ("matchers", "nearest_neighbor")]
for kind, name in mods:
    importlib.import_module(f"hloc.{kind}.{name}")
print("OK", len(mods))
PY
)
[ "$out" = "OK 7" ] \
    && pass "hloc extractors/matchers behind superpoint*/disk/sift/NN*/superglue*/lightglue import" \
    || fail "hloc module imports" "$out"

out=$(dexec test -f /opt/hloc/third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth && echo yes || echo no)
[ "$out" = "yes" ] \
    && pass "SuperPoint weights present (SuperGluePretrainedNetwork submodule checked out)" \
    || fail "SuperPoint weights" "third_party/SuperGluePretrainedNetwork not initialised"

# ============================================================================
# 19. CUDA allocator: expandable segments actually enabled (Tier 2.6)
#
# docker-compose.yml sets both PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF
# because, in the NGC 25.11 torch build, the new name alone does not switch
# expandable segments on. The only reliable check is the allocator's own
# snapshot after a real allocation: every segment must report is_expandable.
# ============================================================================
section "19. CUDA allocator (expandable segments)"

env_out=$(dexec env 2>/dev/null | grep -E '^PYTORCH_(CUDA_)?ALLOC_CONF=' | sort | tr '\n' ' ')
echo "$env_out" | grep -q 'PYTORCH_ALLOC_CONF=expandable_segments:True' \
    && pass "PYTORCH_ALLOC_CONF set in the container" \
    || fail "PYTORCH_ALLOC_CONF" "got: $env_out"
echo "$env_out" | grep -q 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
    && pass "PYTORCH_CUDA_ALLOC_CONF set in the container (the one this torch build honours)" \
    || fail "PYTORCH_CUDA_ALLOC_CONF" "got: $env_out — recreate the container (docker compose up -d) after editing docker-compose.yml"

out=$(dexec_in python - <<'PY' 2>&1 | grep -E '^(EXPANDABLE|ERR)' | tail -1
import torch
try:
    x = torch.zeros(16 * 1024 * 1024, device="cuda")  # 64 MB -> at least one segment
    segs = torch.cuda.memory._snapshot()["segments"]
    print("EXPANDABLE", all(s.get("is_expandable") for s in segs), len(segs))
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:120])
PY
)
case "$out" in
    "EXPANDABLE True"*) pass "allocator segments are expandable ($out)" ;;
    *) fail "allocator expandable segments" "$out" ;;
esac

# ============================================================================
# 20. cuDNN autotuning off in ns-train (Stage 5 sed, v4.8)
#
# nerfstudio's train.py sets torch.backends.cudnn.benchmark = True; the first
# LPIPS eval then autotunes AlexNet convolutions at full resolution and that
# single call peaked at +12.4 GiB (1920 px). The patch makes it opt-in via
# NS_CUDNN_BENCHMARK=1. Verify the anchor, the compile, and the effective
# value after importing the module both ways.
# ============================================================================
section "20. cuDNN autotuning patch (v4.8)"

trf='/usr/local/lib/python3.12/dist-packages/nerfstudio/scripts/train.py'
c_new=$(dexec grep -c 'NS_CUDNN_BENCHMARK' "$trf" | tr -d '[:space:]')
c_old=$(dexec grep -c '^torch.backends.cudnn.benchmark = True' "$trf" | tr -d '[:space:]')
[ "${c_new:-0}" = "1" ] && pass "train.py carries the NS_CUDNN_BENCHMARK switch" || fail "cudnn patch anchor" "got $c_new"
[ "${c_old:-0}" = "0" ] && pass "unpatched 'cudnn.benchmark = True' absent"        || fail "unpatched cudnn line removed" "$c_old ref(s) remain"

out=$(dexec bash -c 'CUDA_VISIBLE_DEVICES= python -c "import nerfstudio.scripts.train, torch; print(torch.backends.cudnn.benchmark)"' 2>&1 | tail -1)
[ "$out" = "False" ] && pass "cudnn.benchmark is False after importing ns-train (default)" || fail "cudnn.benchmark default" "$out"
out=$(dexec bash -c 'NS_CUDNN_BENCHMARK=1 CUDA_VISIBLE_DEVICES= python -c "import nerfstudio.scripts.train, torch; print(torch.backends.cudnn.benchmark)"' 2>&1 | tail -1)
[ "$out" = "True" ] && pass "NS_CUDNN_BENCHMARK=1 restores upstream behaviour" || fail "cudnn override" "$out"

# ============================================================================
# 21. Training smoke: all five methods on a synthetic dataset (v4.8)
#
# Registration and import checks (§10) passed 62/62 on an image where two
# methods could not start, and the eval-VRAM misattribution of 2026-10-05 was
# only caught by running the generated command. So: actually train. A
# synthetic 8-camera COLMAP model + rendered images is generated inside the
# container with nerfstudio's own writers (no user data, ~1 MB), every method
# trains 300 iterations headless with an eval image every 100 steps, a PLY is
# exported through both exporters, and the splatfacto-w colour-cache check
# from the v4.7 patch runs deterministically on the resulting checkpoint.
# Skip with --skip-train. Each run takes ~8 s on a 5070 Ti.
# ============================================================================
section "21. Training smoke (5 methods × 300 iterations, synthetic data)"

if [ "$SKIP_TRAIN" = "1" ]; then
    warn "training smoke skipped (--skip-train)"
else
    SYN="/tmp/smoke_train_$$"
    dexec mkdir -p "$SYN" > /dev/null 2>&1

    out=$(dexec_in env CUDA_VISIBLE_DEVICES= python - "$SYN/data" <<'PY' 2>&1 | grep -E '^(SYNTH_OK|Traceback|.*Error)' | tail -1
import sys, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from nerfstudio.data.utils.colmap_parsing_utils import (
    Camera, Image as CImage, Point3D, rotmat2qvec,
    write_cameras_binary, write_images_binary, write_points3D_binary,
)
from nerfstudio.process_data.colmap_utils import colmap_to_json
root = Path(sys.argv[1]); W, H, F = 160, 120, 140.0
sparse = root / "colmap" / "sparse" / "0"; sparse.mkdir(parents=True, exist_ok=True)
(root / "images").mkdir(exist_ok=True)
rng = np.random.default_rng(0)
pts = rng.uniform(-0.8, 0.8, size=(400, 3)); cols = rng.integers(40, 255, size=(400, 3))
cams = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=np.array([F, F, W / 2, H / 2]))}
imgs = {}
for i in range(8):
    a = 2 * math.pi * i / 8
    C = np.array([4.0 * math.cos(a), 4.0 * math.sin(a), 1.5])
    f = -C / np.linalg.norm(C)
    r = np.cross(f, np.array([0, 0, 1.0])); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f]); t = -R @ C
    img = np.full((H, W, 3), 90, dtype=np.uint8)
    pc = (R @ pts.T).T + t
    for (x, y, z), c in zip(pc, cols):
        if z <= 0.1: continue
        u, v = int(F * x / z + W / 2), int(F * y / z + H / 2)
        if 1 <= u < W - 1 and 1 <= v < H - 1:
            img[v - 1:v + 2, u - 1:u + 2] = c
    name = f"frame_{i + 1:05d}.png"
    Image.fromarray(img).save(root / "images" / name)
    imgs[i + 1] = CImage(id=i + 1, qvec=rotmat2qvec(R), tvec=t, camera_id=1, name=name,
                         xys=np.zeros((0, 2)), point3D_ids=np.array([], dtype=np.int64))
p3d = {j + 1: Point3D(id=j + 1, xyz=pts[j], rgb=cols[j], error=0.5,
                      image_ids=np.array([], dtype=np.int64), point2D_idxs=np.array([], dtype=np.int64)) for j in range(400)}
write_cameras_binary(cams, sparse / "cameras.bin")
write_images_binary(imgs, sparse / "images.bin")
write_points3D_binary(p3d, sparse / "points3D.bin")
n = colmap_to_json(recon_dir=sparse, output_dir=root)
print(f"SYNTH_OK frames={n}")
PY
)
    if echo "$out" | grep -q '^SYNTH_OK frames=8'; then
        pass "synthetic dataset generated ($out)"
    else
        fail "synthetic dataset generation" "$out"
    fi

    COMMON="--data $SYN/data --max-num-iterations 300 --steps-per-save 100000 --steps-per-eval-image 100 --steps-per-eval-all-images 100000 --vis tensorboard --experiment-name smoke"
    for m in splatfacto splatfacto-big splatfacto-mcmc splatfacto-w-light splatfacto-w; do
        extra=""
        [ "$m" = "splatfacto-w" ] && extra="splatfactow_dataparser --colmap-path colmap/sparse/0"
        t0=$(date +%s)
        # shellcheck disable=SC2086
        log=$(dexec bash -c "cd /tmp && timeout 600 ns-train $m $COMMON --output-dir $SYN/out_$m $extra 2>&1; echo EXIT=\$?")
        rc=$(echo "$log" | grep -o 'EXIT=[0-9]*' | tail -1 | cut -d= -f2)
        banner=$(echo "$log" | grep -c 'Training Finished')
        ckpt=$(dexec bash -c "find $SYN/out_$m -name '*.ckpt' | wc -l" | tr -d '[:space:]')
        if [ "$rc" = "0" ] && [ "$banner" -ge 1 ] && [ "${ckpt:-0}" -ge 1 ]; then
            pass "ns-train $m: 300 iterations, banner, checkpoint ($(( $(date +%s) - t0 )) s)"
        else
            fail "ns-train $m" "exit=$rc banner=$banner ckpt=$ckpt — $(echo "$log" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E 'Error|assert' | tail -2 | cut -c1-200 | tr '\n' ' ')"
        fi
    done

    cfg_std=$(dexec bash -c "ls $SYN/out_splatfacto/smoke/splatfacto/*/config.yml 2>/dev/null | head -1" | tr -d '[:space:]')
    cfg_w=$(dexec bash -c "ls $SYN/out_splatfacto-w/smoke/splatfacto-w/*/config.yml 2>/dev/null | head -1" | tr -d '[:space:]')

    if [ -n "$cfg_std" ]; then
        out=$(dexec bash -c "cd /tmp && timeout 300 ns-export gaussian-splat --load-config $cfg_std --output-dir $SYN/exp_std > /dev/null 2>&1; stat -c %s $SYN/exp_std/splat.ply 2>/dev/null" | tr -d '[:space:]')
        [ "${out:-0}" -gt 1000 ] && pass "ns-export gaussian-splat wrote splat.ply ($out bytes)" || fail "ns-export gaussian-splat" "no PLY ($out)"
    else
        fail "ns-export gaussian-splat" "no splatfacto config.yml to export from"
    fi

    if [ -n "$cfg_w" ]; then
        out=$(dexec bash -c "cd /tmp && timeout 300 ns-export-splatfactow --load-config $cfg_w --output-dir $SYN/exp_w > /dev/null 2>&1; stat -c %s $SYN/exp_w/splat.ply 2>/dev/null" | tr -d '[:space:]')
        [ "${out:-0}" -gt 1000 ] && pass "ns-export-splatfactow wrote splat.ply ($out bytes)" || fail "ns-export-splatfactow" "no PLY ($out)"

        # Deterministic colour-cache check (v4.7 patch): train render on camera k,
        # cull, eval render on the SAME camera must not assert.
        out=$(dexec_in python - "$cfg_w" <<'PY' 2>&1 | grep -E '^RESULT' | tail -1
import sys, torch
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
_, pipeline, _, _ = eval_setup(Path(sys.argv[1]), test_mode="val")
model, dm = pipeline.model, pipeline.datamanager
def cam(i):
    c = dm.train_dataset.cameras[i:i+1].to(model.device); c.metadata = {"cam_idx": i}; return c
ok = True
for same in (False, True, True):
    model.train(); c, _ = dm.next_train(0); k = c.metadata["cam_idx"]; model.get_outputs(c)
    with torch.no_grad():
        m = torch.zeros(model.num_points, dtype=torch.bool, device=model.device); m[:5] = True; model.cull_gaussians(m)
    model.eval(); e = k if same else (k + 1) % len(dm.train_dataset)
    try:
        with torch.no_grad(): model.get_outputs_for_camera(cam(e))
    except AssertionError as ex:
        ok = False
print("RESULT", "PASS" if ok else "FAIL", model.num_points)
PY
)
        echo "$out" | grep -q '^RESULT PASS' && pass "splatfacto-w colour cache survives cull between train and eval render ($out)" || fail "splatfacto-w colour-cache check" "$out"
    else
        fail "ns-export-splatfactow / cache check" "no splatfacto-w config.yml"
    fi

    dexec rm -rf "$SYN" > /dev/null 2>&1
fi

# ============================================================================
# Summary
# ============================================================================
section "Summary"
printf '  %sPASS%s: %d   %sFAIL%s: %d   %sWARN%s: %d\n' "$G" "$N" "$PASS" "$R" "$N" "$FAIL" "$Y" "$N" "$WARN"

if [ "$FAIL" -gt 0 ]; then
    printf '\n%sFailed tests:%s\n' "$R" "$N"
    for t in "${FAILED[@]}"; do printf '  - %s\n' "$t"; done
fi

printf '\n%sTo inspect all image layers used by this container, run in another terminal:%s\n' "$B" "$N"
printf '  docker history %s\n' "$IMAGE"
printf '  docker history --no-trunc --format '\''table {{.CreatedBy}}\t{{.Size}}'\'' %s\n' "$IMAGE"
printf '  docker inspect %s --format '\''{{.Config.Image}} parent={{.Parent}}'\''\n' "$IMAGE"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1

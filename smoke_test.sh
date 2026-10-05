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
#   CONTAINER=foo IMAGE=foo:latest ./smoke_test.sh    # override defaults
#
# Exits non-zero if any test fails. Never stops on failure — runs every
# test so you see the full picture.

set -u

CONTAINER="${CONTAINER:-splat-ns-v2}"
IMAGE="${IMAGE:-splat-nerfstudio-v2:latest}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/home/meroe/docker-stacks/docker_splat_v2/data}"
SKIP_JIT=0
for arg in "$@"; do
    case "$arg" in
        --skip-jit) SKIP_JIT=1 ;;
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

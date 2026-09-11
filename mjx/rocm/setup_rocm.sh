#!/usr/bin/env bash
# Provision MJX (MuJoCo on JAX) on a DigitalOcean AMD GPU Droplet.
# Host image expected: gpu-amd-base (Ubuntu 24.04, ROCm 7.14, amdgpu-dkms 6.19.14).
# Target: AMD Instinct MI355X / MI350X (gfx950) or MI300X/MI325X (gfx942).
set -euo pipefail

START=$(date +%s)
elapsed() { echo "[+$(( $(date +%s) - START ))s] $*"; }

# gfx950 = MI350X/MI355X, gfx942 = MI300X/MI325X. Override: ARCH=gfx942 ./setup_mi355x.sh
ARCH="${ARCH:-gfx950}"
ROCM_VER="${ROCM_VER:-7.14.1}"
JAX_VER="${JAX_VER:-0.10.0}"
AMD_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"

elapsed "host check"
rocminfo 2>/dev/null | grep -m4 -E 'Name:|gfx' || echo "WARNING: rocminfo unavailable"
amd-smi static 2>/dev/null | head -20 || rocm-smi || true

elapsed "installing uv"
command -v uv >/dev/null 2>&1 || {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}
export PATH="$HOME/.local/bin:$PATH"

elapsed "creating venv"
cd "$HOME"
uv venv --python 3.12 mjx-env
# shellcheck disable=SC1091
source mjx-env/bin/activate

elapsed "installing ROCm runtime libraries for ${ARCH} (largest download, be patient)"
uv pip install --index-url "$AMD_INDEX" "rocm[libraries,device-${ARCH}]==${ROCM_VER}"

elapsed "installing JAX ROCm plugin"
uv pip install --index-url "$AMD_INDEX" \
  "jax_rocm7_plugin==${JAX_VER}+rocm${ROCM_VER}" \
  "jax_rocm7_pjrt==${JAX_VER}+rocm${ROCM_VER}"
uv pip install "jax==${JAX_VER}" "jaxlib==${JAX_VER}"

elapsed "installing MuJoCo"
uv pip install mujoco mujoco-mjx numpy

# ROCm workarounds from the AMD MuJoCo blog post.
export LLVM_PATH=/opt/rocm/llvm
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export MUJOCO_GL=disable
cat >> "$HOME/mjx-env/bin/activate" <<'ENVEOF'
export LLVM_PATH=/opt/rocm/llvm
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export MUJOCO_GL=disable
ENVEOF

elapsed "verifying JAX sees the GPU"
python -c "import jax; d=jax.devices(); print('devices:', d); assert d[0].platform=='gpu', 'NO GPU VISIBLE TO JAX'; print('OK')"

elapsed "fetching humanoid model"
mkdir -p "$HOME/models"
curl -fsSL -o "$HOME/models/humanoid.xml" \
  https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml

elapsed "SETUP COMPLETE - run: source ~/mjx-env/bin/activate && python bench_mjx_humanoid.py"

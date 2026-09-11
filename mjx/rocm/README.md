# MJX on AMD ROCm (Instinct MI355X)

Benchmarking MuJoCo MJX under JAX on AMD Instinct GPUs via ROCm, on DigitalOcean
GPU Droplets. Measured on an **AMD Instinct MI355X (gfx950)**.

This is the AMD counterpart to [`mjx/macosx`](../macosx). Same simulator, same
humanoid model, different backend.

## Why MJX and not mjlab or MuJoCo Warp

This matters, because it determines what can run on AMD at all.

| Stack | Backend | Runs on AMD? |
|---|---|---|
| **MJX (JAX)** | JAX → XLA → ROCm | **Yes** |
| MuJoCo Warp | NVIDIA Warp → CUDA SIMT | No |
| mjlab | built on MuJoCo Warp | No |
| mjbatch | C++ thread pool, CPU only | N/A, no GPU path at all |

`mjlab` states it requires an NVIDIA GPU. It sits on MuJoCo Warp, which sits on
NVIDIA Warp, which emits CUDA SIMT kernels and has no ROCm backend.
`mjbatch` is a CPU library: its dependencies are `mujoco`, `numpy` and
`nanobind`, with no JAX and no Warp. Renting a GPU for it buys nothing.

MJX on JAX is therefore the only MuJoCo path to AMD silicon. MuJoCo Playground
exposes the same choice as `--impl jax` versus `--impl warp`.

## Hardware and pricing (DigitalOcean, verified 2026-09-11)

| Slug | GPU | Arch | $/hr | Type | Region |
|---|---|---|---|---|---|
| `gpu-mi300x1-192gb` | MI300X | gfx942 | 2.59 | on-demand | none listed |
| `gpu-mi325x1-256gb` | MI325X | gfx942 | 3.80 | on-demand | tor1 |
| `gpu-mi350x1-288gb-spot` | MI350X | gfx950 | 4.00 | **spot** | none listed |
| `gpu-mi355x1-288gb-spot` | MI355X | gfx950 | 4.50 | **spot** | mem1 |

Notes that cost real money to discover:

- **MI350X and MI355X are spot-only.** There is no on-demand variant.
- **MI300X appears in zero regions** and cannot actually be placed, despite
  reporting `available: true` in the sizes API.
- **GPU Droplets are gated by account tier.** Tier 1 allows **zero**. Tier 2
  allows one and costs a $50 prepayment or accrued invoice history. A new
  account cannot create a GPU Droplet at any price without clearing this.

Image slug `gpu-amd-base` ("AMD AI/ML Ready Image") ships Ubuntu 24.04 with
ROCm 7.14 and amdgpu-dkms 6.19.14 preinstalled, which makes provisioning fast.

## Quick start

```bash
# One-shot: create, provision, benchmark, retrieve, destroy.
./run_bench.sh

# Cheaper gfx942 hardware
ARCH=gfx942 SIZE=gpu-mi325x1-256gb REGION=tor1 RATE_PER_HOUR=3.80 ./run_bench.sh

# Tighter budget ceiling
MAX_SECONDS=900 ./run_bench.sh
```

Requires `doctl` authenticated, an SSH key on the DigitalOcean account, and a
GPU Droplet quota of at least 1.

To run the benchmark on a host you already have:

```bash
ARCH=gfx950 ./setup_rocm.sh
source ~/mjx-env/bin/activate
python bench_mjx_humanoid.py --batch-sizes 1024,2048,4096,8192,16384 --steps 1000
```

## Files

| File | Purpose |
|---|---|
| `run_bench.sh` | Provider driver: create, provision, benchmark, retrieve, destroy |
| `setup_rocm.sh` | Installs ROCm wheels, JAX ROCm plugin, MuJoCo. Provider-agnostic |
| `bench_mjx_humanoid.py` | The benchmark. Backend-agnostic, runs on CUDA and CPU too |
| `results/` | Raw JSON output |
| [`RESULTS.md`](RESULTS.md) | Measurements and how to read them |
| [`LESSONS.md`](LESSONS.md) | Failures, including the ones that cost money |

## Environment

```
device       AMD Instinct MI355X VF (gfx950)
ROCm         7.14.1
JAX          0.10.0  (jax_rocm7_plugin, jax_rocm7_pjrt)
MuJoCo       3.13.0
mujoco-mjx   separate package, not bundled with mujoco
host         Ubuntu 24.04, kernel 6.8.0
model        github.com/google-deepmind/mujoco model/humanoid/humanoid.xml
```

Three environment variables are required, two of them from AMD's ROCm JAX MuJoCo
guide:

```bash
export LLVM_PATH=/opt/rocm/llvm
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export MUJOCO_GL=disable    # NOT osmesa; see LESSONS.md
```

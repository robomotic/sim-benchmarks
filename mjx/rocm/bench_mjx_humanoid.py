"""Throughput benchmark for MJX humanoid simulation.

Measures batched physics steps per second across a sweep of batch sizes.
Backend-agnostic: runs on AMD via the JAX ROCm plugin, or NVIDIA via CUDA.
Designed to produce a defensible number in a few minutes of paid GPU time.
"""

import argparse
import functools
import json
import os
import platform
import time

import jax
import mujoco
from mujoco import mjx


def build(model_path, batch_size):
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Replicate the single state into a batch, jittering qpos so the batch is
    # not degenerate and contact patterns diverge across environments.
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def make_one(key):
        noise = jax.random.uniform(key, (mj_model.nq,), minval=-0.01, maxval=0.01)
        return mjx_data.replace(qpos=mjx_data.qpos + noise)

    batch = jax.vmap(make_one)(keys)
    return mj_model, mjx_model, batch


def benchmark(model_path, batch_size, n_steps, warmup_steps):
    mj_model, mjx_model, batch = build(model_path, batch_size)

    # n must be static: lax.scan needs a concrete length at trace time.
    @functools.partial(jax.jit, static_argnums=(1,))
    def rollout(data, n):
        def body(d, _):
            return jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, d), None
        d, _ = jax.lax.scan(body, data, None, length=n)
        return d

    # Compilation happens on this call; exclude it from the timing.
    t0 = time.perf_counter()
    state = rollout(batch, warmup_steps)
    jax.block_until_ready(state)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = rollout(state, n_steps)
    jax.block_until_ready(state)
    wall_s = time.perf_counter() - t0

    total_steps = batch_size * n_steps
    return {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "compile_and_warmup_s": round(compile_s, 3),
        "wall_s": round(wall_s, 4),
        "steps_per_sec": round(total_steps / wall_s, 1),
        "sim_seconds_per_wall_sec": round(
            total_steps * mj_model.opt.timestep / wall_s, 2
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/humanoid.xml"))
    p.add_argument("--batch-sizes", default="1024,2048,4096,8192")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--out", default="mjx_bench_results.json")
    p.add_argument(
        "--target-steps",
        type=int,
        default=0,
        help="Hold total work roughly constant per batch size: n_steps becomes "
        "target_steps // batch_size, floored at --min-steps. Keeps a knee hunt "
        "affordable, since a fixed step count at batch 262144 runs for minutes.",
    )
    p.add_argument("--min-steps", type=int, default=50)
    args = p.parse_args()

    device = jax.devices()[0]
    meta = {
        "device_kind": device.device_kind,
        "platform": device.platform,
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "host": platform.platform(),
    }
    print(json.dumps(meta, indent=2))
    print(f"\n{'batch':>8} {'steps/s':>14} {'sim s / wall s':>16} {'compile s':>11}")
    print("-" * 53)

    results = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        n_steps = args.steps
        if args.target_steps:
            n_steps = max(args.min_steps, args.target_steps // bs)
        try:
            r = benchmark(args.model, bs, n_steps, args.warmup)
        except Exception as exc:  # out of memory at large batch is expected
            print(f"{bs:>8}  FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        results.append(r)
        print(
            f"{r['batch_size']:>8} {r['steps_per_sec']:>14,.0f} "
            f"{r['sim_seconds_per_wall_sec']:>16,.1f} "
            f"{r['compile_and_warmup_s']:>11.1f}"
        )
        # Flush after every batch size. MI355X is spot-only on DigitalOcean and
        # can be reclaimed mid-sweep; partial results must survive that.
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

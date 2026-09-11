# Lessons from benchmarking on rented GPUs

Every item here came from something going wrong on a metered machine. Recorded
so the next person does not repay for them.

## 1. A `sleep`-based kill-switch does not survive laptop suspend

**Cost: roughly $13.50, against a $5 budget.**

The driver armed a detached watchdog:

```bash
setsid bash -c "sleep 1800; doctl compute droplet delete mjx-bench --force" &
```

The laptop then suspended twice, for 37 minutes and 1 hour 41 minutes. Linux
`sleep` does not advance while the machine is suspended, so a 30-minute timer
sitting through 2h18m of suspend still had time remaining on resume. It never
fired. The Droplet billed for about three hours.

Worse, the SSH session died at the first suspend, so the benchmark itself was
killed 16 minutes in. The remaining 2h45m was an idle GPU billing at full rate.

**Fixes, all three applied in `run_bench.sh`:**

```bash
# 1. Absolute wall-clock deadline, polled. A resume past the deadline fires at once.
DEADLINE=$(( $(date +%s) + MAX_SECONDS ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 20; done
doctl compute droplet delete "$NAME" --force

# 2. Block suspend for the duration of the run.
exec systemd-inhibit --what=sleep:idle -- "$0" "$@"

# 3. Run the benchmark DETACHED on the remote host, then poll for a marker,
#    so a dropped connection no longer kills the work.
```

Test a watchdog against the failure mode you actually have. Verifying it
survived `SIGKILL` proved nothing about suspend, which was the real risk on a
laptop.

## 2. Retrieve results before destroying anything

When the runaway Droplet was found, the reflex was to delete immediately to stop
the meter. Correct instinct, but the benchmark writes its JSON to the Droplet,
so deleting destroyed the only copy of any partial data. A ten-second `scp`
first would have cost nothing.

`run_bench.sh` now salvages `mjx_bench_results.json` and `bench.log` inside the
teardown trap, before the delete.

## 3. Write results incrementally

Spot capacity can vanish mid-sweep. The benchmark flushes its JSON after every
batch size rather than at the end, so a reclaim keeps whatever finished.

## 4. `MUJOCO_GL=osmesa` breaks a headless benchmark

AMD's ROCm JAX MuJoCo guide sets `MUJOCO_GL=osmesa` because it renders video.
Copying that into a headless benchmark makes MuJoCo load a GL renderer that is
not installed:

```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

Use `MUJOCO_GL=disable` when not rendering. Do not inherit environment settings
from a guide whose goals differ from yours.

## 5. Dry-run on CPU before spending

MJX runs on CPU. Every bug below was caught locally for free, and each would
have burned paid GPU minutes:

- `mjx` is the separate `mujoco-mjx` package, not part of `mujoco`.
- `jax.lax.scan` needs a static length; the step count must be passed via
  `functools.partial(jax.jit, static_argnums=(1,))` or it raises
  `ConcretizationTypeError`.
- The `MUJOCO_GL` failure above reproduces exactly on CPU.

## 6. Verify the GPU is actually being used, in the script

`setup_rocm.sh` asserts the device and exits non-zero otherwise:

```python
import jax
d = jax.devices()
assert d[0].platform == 'gpu', 'NO GPU VISIBLE TO JAX'
```

Failing in the first minute is much cheaper than discovering it later. Note the
DigitalOcean dashboard shows only CPU, disk and memory for a GPU Droplet unless
"Improved Metrics and monitoring" was enabled at creation, so an empty GPU graph
is not evidence of a problem.

## 7. Check the quota before building anything

A new DigitalOcean account is Tier 1, which permits **zero** GPU Droplets.
Creation fails with:

```
422 creating this/these droplet(s) will exceed your GPU limit
```

Tier 2 permits one, and costs a $50 prepayment or accrued invoice history. This
is unrelated to how much credit you hold. First-line support answered a quota
request twice with instructions for creating a Droplet, so expect to escalate
and ask explicitly for a human.

## 8. Confirm a delete actually took

Deletion is not instantaneous. An early check sees a stale listing and reports a
false failure. Poll instead:

```bash
for _ in $(seq 1 20); do
  doctl compute droplet list --no-header --format Name | grep -qx "$NAME" || break
  sleep 3
done
```

And refuse to start at all when a Droplet already exists, so an orphan from a
previous session is noticed instead of joined.

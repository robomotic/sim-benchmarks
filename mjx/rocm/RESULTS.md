# Results: MJX humanoid on AMD Instinct MI355X

Date: 2026-09-11. Device: AMD Instinct MI355X VF (gfx950), ROCm 7.14.1,
JAX 0.10.0, MuJoCo 3.13.0. Model: standard MuJoCo `humanoid.xml`.

## Headline

**106,837 physics steps per second at batch 16,384**, which is about **534x
realtime** for the humanoid. Throughput was still rising at the largest batch
measured, so this is a lower bound, not a peak.

## Sweep 1 — the usable result

Step count fixed at 1,000 for every batch size, so batch size is the only
variable. This is the measurement to cite.

| batch | steps/s | sim s per wall s | compile s |
|---|---|---|---|
| 1,024 | 36,643 | 183.2 | 19.5 |
| 2,048 | 51,339 | 256.7 | 18.1 |
| 4,096 | 62,996 | 315.0 | 19.8 |
| 8,192 | 81,496 | 407.5 | 26.8 |
| 16,384 | 106,837 | 534.2 | 23.0 |

Monotonic throughout, with no sign of flattening. The knee is above 16,384 and
remains unmeasured.

Raw: [`results/mi355x_sweep1_2026-09-11.json`](results/mi355x_sweep1_2026-09-11.json)

## Sweep 2 — confounded, do not cite

An attempt to reach batch 262,144 while capping cost, by scaling step count
inversely with batch size to hold total work constant. This silently broke the
comparison.

| batch | n_steps | steps/s |
|---|---|---|
| 8,192 | 488 | 65,037 |
| 16,384 | 244 | 78,738 |
| 32,768 | 122 | 81,147 |
| 65,536 | 61 | 80,468 |
| 131,072 | 50 | 133,188 |
| 262,144 | 50 | 129,803 |

Against sweep 1 at identical batch sizes:

| batch | 1,000 steps | reduced steps | delta |
|---|---|---|---|
| 8,192 | 81,496 | 65,037 (488) | **-20%** |
| 16,384 | 106,837 | 78,738 (244) | **-26%** |

Fewer steps per call means fixed dispatch overhead is amortized over less work,
so measured throughput falls. Because step count and batch size moved together,
the two effects cannot be separated. The apparent peak at 131,072 is an artifact
of where the `--min-steps` floor of 50 engaged, not a hardware property.

**Lesson: hold step count fixed across a batch sweep.** Cost control belongs in
the choice of that single fixed value, never in varying it per point.

Raw: [`results/mi355x_sweep2_knee_2026-09-11.json`](results/mi355x_sweep2_knee_2026-09-11.json)

## Reproducibility

The batch-1,024 smoke test at 200 steps ran on three separate Droplets:

| run | steps/s |
|---|---|
| 1 | 12,095 |
| 2 | 12,070 |
| 3 | 12,161 |

Spread under 1%. Run-to-run variance is not a concern at this scale.

Note this is far below sweep 1's 36,643 at the same batch size, because the
smoke test uses 200 steps against 1,000. Same effect as sweep 2. It is a
liveness check, not a measurement.

## Compile cost

XLA compilation ran 18 to 30 seconds per batch size at these sizes, and over
100 seconds at 262,144. It is excluded from timings via a warmup call, but it
dominates short runs and is why the smoke test understates throughput.

The ROCm XLA autotuner also emits `No reference output found even though buffer
checking` warnings. Harmless, but noisy.

## On comparing with `mjx/macosx`

Not directly comparable, and the difference is easy to miss. The Apple Silicon
suite reports 114,841 steps/s for the humanoid on an M4 Max **at batch 128**.
This suite's headline is **at batch 16,384**. Those are different operating
points: a small batch measures latency and single-stream efficiency, a large
batch measures saturated throughput.

A GPU is underutilized at batch 128, and a laptop CPU will not hold 16,384
environments in the first place. Reading one number against the other would be
meaningless in either direction.

A fair comparison needs both platforms swept over the same batch-size range with
the same fixed step count. `bench_mjx_humanoid.py` runs on CPU unchanged, so
this is straightforward to do and has not been done yet.

## Open questions

- **Where is the knee?** Requires a fixed-step sweep from 8,192 to 262,144.
  Estimated 13 minutes and about $1.00 on MI355X.
- **How does this compare to NVIDIA?** `bench_mjx_humanoid.py` is
  backend-agnostic, so an H100 or L40S run is directly comparable.
- **MJX versus MuJoCo Warp on equivalent NVIDIA hardware?** The interesting
  question, since Warp is the reason AMD is locked out of mjlab.

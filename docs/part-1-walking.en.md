# 1. Learning to walk

!!! abstract "Summary"
    Open Duck Mini v2 trained with the official environment and rewards, **unchanged**. One run, seed 0, 24 October 2026, on an Apple M5 MacBook (10 CPU cores, 6 used; the machine was rendering other videos at the same time). 72.4 M steps in 3.57 h. The duck learned to stand in 11 minutes and to step at 68.4 M steps.

## The run

| | |
|---|---|
| Wall-clock time | **3.57 h** (04:15:55 → 07:50:08) |
| Environment steps | **72,437,760** (≈ 48 % of the official 150 M recipe) |
| Parallel ducks | 1,536 (6 CPU devices × 256 envs) |
| Episodes | 429,369 (falls and 20 s time-outs together; the log does not separate them) |
| Speed | 5,636 env steps/s on average (1,924 – 7,358 per chunk) |
| Simulated time | 72.4 M × 0.02 s ≈ 16.8 days, about 113× real time |
| Checkpoints | 19, one every 4,024,320 steps |

## What the policy did

Measured with [`eval_checkpoint.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_checkpoint.py): flat floor, no pushes, forward command 0.15 m/s, 5 s, deterministic policy. The "9 noisy ducks" column samples actions with the exploration noise, as in training ([`eval_grid.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_grid.py)).

| Checkpoint | Wall-clock | Forward distance in 5 s | Feet | 9 noisy ducks |
|---|---|---|---|---|
| 0 steps | 0 min | 0.4 cm (stands frozen) | no steps | **8 of 9 fall** |
| 4.0 M | **10 min 45 s** | 1.2 cm | no steps | 0 of 9 fall |
| 8 M – 64.4 M | 29 – 196 min | 2.1 – 4.3 cm | no steps (feet lift ≤ 6 mm) | 0 of 9 fall |
| **68.4 M** | 205 min | **29.8 cm** | first lifted feet, ~1.2 cm | 0 of 9, mean 18 cm |
| 72.4 M (last) | 214 min | **28.8 cm** | small, shuffling steps | 0 of 9, mean 22 cm |
| `BEST_WALK_ONNX` (pretrained, not ours) | – | **58.5 cm** | 8 + 8 clean steps | – |

In short: within about **11 minutes** it learned to stay on its feet, then sat on that plateau for **about 3 hours**, and only between 64.4 M and 68.4 M steps did it start lifting its feet. At 72.4 M it covers half the distance of the pretrained policy in the same test. A 0.15 m/s command would be 75 cm in 5 s, so even the pretrained policy is below the target here.

![Episode return, episode length and imitation reward vs environment steps](assets/reward_curve.png)

Mean episode return went from 17.5 (first 1 M steps) to 78.3 (mean of the last 10 M), episode length from 1.2 s to 3.5 s (limit 20 s). The clearest learning signal is the imitation term: −127 → +168.

## Why CPU?

We tried the Mac's GPU first. With `jax-metal 0.1.1` + `jax 0.4.34` + `mujoco-mjx 3.2.3`, compiling the MJX step fails with `failed to legalize operation 'mhlo.cholesky'`: the Metal plugin does not support an operation the physics needs. That is missing software support, not a hardware limit.

## The CPU trick

One JAX CPU device with multi-threaded XLA reached 2,368 env steps/s (1,024 envs). Splitting the CPU into 6 virtual devices with one thread each (`--xla_force_host_platform_device_count=6 --xla_cpu_multi_thread_eigen=false`) lets Brax `pmap` across cores: 4,621 steps/s with 6 × 256 envs, about 2× faster. `train_mac.py` sets this for you.

## Reproduce it

Step 6 of [Setup](setup.md) is the exact command of our run. Brax rounds the step count up to whole PPO iterations, so 72,000,000 ends at 72,437,760.

```bash
# Measure your checkpoint or ours
.venv/bin/python -I scripts/eval_checkpoint.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/c72.mp4 --no_video       # "dx_m": 0.2882
.venv/bin/python -I scripts/eval_grid.py external/Open_Duck_Playground \
  runs/my-run/ckpt/ckpt_000000000.onnx out/c0_grid.mp4                     # "fell": 8
```

For the pretrained comparison, download [`BEST_WALK_ONNX.onnx`](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/BEST_WALK_ONNX.onnx) from Open_Duck_Mini and run the same command (`"dx_m": 0.5845`).

### How long on your Mac?

Our M5 (10 cores, 6 devices × 256 envs) averaged 5,636 steps/s while other jobs were running: 72 M steps took 3.6 h, 150 M would take about 7.5 h. An idle machine should be faster. The first compile takes a few minutes. Fewer cores means proportionally longer; run `smoke_pmap.py` to estimate (training also spends time on the PPO update, so expect less than the smoke-test number).

## What this is not

- **Not the official recipe.** Upstream trains 150 M steps with 8,192 envs on an NVIDIA GPU. We used 1,536 envs, a smaller batch, no evaluation runs, and stopped at 72.4 M. We do not know how `BEST_WALK_ONNX` was trained.
- **Not a walking robot yet.** At 72.4 M steps it shuffles 29 cm in 5 s with small steps. A first gait, not a good one.
- **Not statistics.** One run, one seed. Another seed would lift its feet at a different step; "68 M" belongs to this run only.
- **Not Microduck**, and **no real robot**. The ONNX has the upstream interface, but we have not run it on hardware.
- **Not identical physics in evaluation.** Training runs in MJX, evaluation in C MuJoCo with the same model and control loop; small numerical differences are possible. Once the trunk starts to tip, `eval_checkpoint.py` makes the robot's meshes collide with the floor so a falling duck does not sink through it on video (in training only the feet touch the floor and a fall ends the episode).
- **Not a fork.** Open_Duck_Playground is used as is, from a pinned commit; nothing here modifies it.

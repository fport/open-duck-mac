---
title: 'Scripts and files'
---

Every script prints its usage with `--help`.

## Scripts

| Script | What it does |
|---|---|
| [`train_mac.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/train_mac.py) | Brax PPO on the unchanged `joystick.py` env (flat terrain), `BerkeleyHumanoidJoystickFlatTerrain` hyperparameters, pmap over virtual CPU devices; logs, ONNX + params per checkpoint |
| [`onnx_export.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/onnx_export.py) | Brax policy params → ONNX with plain `onnx` (no TensorFlow/tf2onnx); also outputs `loc_out`/`scale` for noisy sampling |
| [`smoke.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/smoke.py), [`smoke_pmap.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/smoke_pmap.py) | Does the env compile and step, and how fast, on one device or D devices |
| [`eval_checkpoint.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_checkpoint.py) | Rollout of an ONNX policy in C MuJoCo with upstream `MjInfer`; distance, falls, foot steps; optional video |
| [`eval_grid.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/eval_grid.py) | The same with exploration noise for 9 seeds; how many fall, mean distance, optional 3×3 video |
| [`make_log.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/make_log.py) | Raw logs + rollout JSONs → `train_log.csv`, `ckpt_summary.json`, `reward_curve.png` |
| [`reward_hacking/hile_env.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/hile_env.py) | Part 2's reward file (`VARIANTS`) |
| [`reward_hacking/train_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/train_hile.py) | Training with reward variants, warm start with `--restore` |
| [`reward_hacking/make_hile_repo.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/make_hile_repo.py) | Adds massless floor-colliding boxes to a copy of Open_Duck_Playground |
| [`reward_hacking/render_ckpt_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_ckpt_hile.py), [`render_grid_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_grid_hile.py) | Part 2's evaluators (training physics, extra measurements) |
| [`sim2real/eval_world.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/sim2real/eval_world.py) | `eval_checkpoint.py` + environment variables that change the world (`FORCE`, `FORCE_RAND`, `FLOOR_FRIC`, `MASS_SCALE`) |

## Result folders

| Folder | Contents |
|---|---|
| [`results/2026-10-24/`](https://github.com/fport/open-duck-mac/tree/main/results/2026-10-24) | Part 1: `train_log.csv` (one table), `train_log_raw.csv`, `checkpoints.csv`, `ckpt_summary.json`, `train_metrics.jsonl` (PPO losses, steps/s per chunk), `run_config.json` (PPO parameters), `sim/*.json` (every measurement), the 68.4 M and 72.4 M ONNX files, `params/ckpt_068413440.pkl` (for warm starts) |
| [`results/2026-10-24-reward-hacking/`](https://github.com/fport/open-duck-mac/tree/main/results/2026-10-24-reward-hacking) | Part 2: `summary.json`; per variant `run.json`, `train_log.csv`, `train_metrics_ep.jsonl`, `onnx/`, `sim/` |

`make_log.py` rebuilds part 1's `train_log.csv` and `ckpt_summary.json` from these files byte for byte.

## Environment

- Python 3.12, `requirements-lock.txt` (full environment), `requirements.txt` (direct dependencies: jax 0.6.2, mujoco / mujoco-mjx 3.3.3, brax 0.12.4, playground 0.0.5, onnx, onnxruntime, imageio).
- Open_Duck_Playground commit `b9be205ac64488c23504ca42e5ec790337adeec3`.
- Pretrained comparison: `BEST_WALK_ONNX.onnx` from Open_Duck_Mini commit `b23317a`.

## Licence and credits

- **Open Duck Mini v2** and **Open_Duck_Playground** by [Antoine Pirrone](https://github.com/apirrone) and contributors: robot, MJCF model, `joystick.py` environment, rewards, reference motion (`polynomial_coefficients.pkl`), `mujoco_infer.py` and the pretrained `BEST_WALK_ONNX`. Open_Duck_Mini is Apache-2.0. Open_Duck_Playground's source files carry Apache-2.0 headers, but its repository root had no LICENSE file at `b9be205`.
- **[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)**, **[MuJoCo / MJX](https://github.com/google-deepmind/mujoco)** and **[Brax](https://github.com/google/brax)** by Google DeepMind and contributors, Apache-2.0.
- The code in this repository is released under the [Apache License 2.0](https://github.com/fport/open-duck-mac/blob/main/LICENSE). `results/` (logs, measurements, our ONNX policies) are released under the same licence.

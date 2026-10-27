# 2. Reward hacking

![Three ducks after reward hacking: crawling on its belly ("be fast"), frozen in place ("don't waste energy"), lying down with its feet in the air ("lift your feet")](assets/reward-hacking.jpg)

!!! abstract "Summary"
    Starting from our 68.4 M checkpoint we changed **only the reward weights** (and, in one variant, the fall rule) and let PPO keep training. No motion, pose or trajectory is scripted anywhere: every behaviour below is what PPO found. The robot learns what you **score**, not what you **want**.

The reward file is [`scripts/reward_hacking/hile_env.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/hile_env.py) (`VARIANTS`). It subclasses the unchanged upstream `Joystick` environment and adds two terms: raw forward speed (`fwd_vel`) and upstream's unused `reward_feet_air_time`. Warm start: policy and observation normalisation restored from 68.4 M, value network from scratch.

One run per variant, seed 0, 24 October 2026, same Mac, 5 CPU devices × 256 envs, fixed command 0.15 m/s forward.

## Four "instructions"

| Variant (the "instruction") | Weights changed | Time to the hack | What PPO found |
|---|---|---|---|
| `hizli` — "be fast" | raw forward speed 25 (= upstream speed weight 2.5 × 10); imitation, alive, energy, smoothness 0 | **14.85 min** (first checkpoint, 3.0 M steps) | Drops onto its belly and crawls: 49 cm at 14.9 min, 54.6 cm at 9.0 M steps (2.1× the 25.8 cm start), feet off the floor 87 % of the time, 0 steps. 9 noisy ducks: 8 crawl forward, 1 tips over backwards. |
| `hizli_serbest` — same, falling allowed | as `hizli`, episode does not end on a fall | 26 min run | Same crawl (54.0 cm): the 90° fall rule was not what kept it upright. |
| `enerji` — "don't waste energy" | torque² penalty −1.0 (upstream −0.001, ×1000), action rate −5 (×10), speed tracking ×0.1, imitation 0 | **7.50 min** (first checkpoint) | Freezes: 2.5 cm in 5 s, 0 steps, torque² −85 %, joint travel −93 %. |
| `ayak` — "lift your feet" | `feet_air_time` 3000, speed tracking 0, imitation 0 | **30.3 min** | At 10.7 min it takes 13 real steps (one deterministic test; 5 of 9 noisy ducks fall). At 30.3 min it lies down and holds its feet 6–11 cm in the air, never touching the floor. 9 noisy ducks: all 9 lie down (4 forward, 5 backwards). |

"Time to the hack" is the first checkpoint at which the behaviour shows up, so it is an upper bound: in the `hizli` log the trunk already starts to drop at 2 M steps (9.8 min).

## Why does "lift your feet" lie down?

Upstream's `reward_feet_air_time` is `(air_time − 0.1) × first_contact`, clipped above at 0.4 but **not below**. So a foot that lands after less than 0.1 s in the air is *penalised* (× 3000). A foot that never lands is neither rewarded nor penalised, while `alive` (20) keeps paying. In the first 0.5 M steps this term summed to −310.8 per episode; at the end, +3,877.8.

That is our reading of the formula, not an ablation: we did not run the experiment that would isolate the cause.

## Collision boxes

In the upstream model only the two foot soles collide with the floor; a falling trunk passes through it (harmless in training, where a fall ends the episode). For these runs [`make_hile_repo.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/make_hile_repo.py) makes a **copy** of Open_Duck_Playground and adds massless, invisible boxes on the trunk, head and four leg parts that collide only with the floor (mass and inertia are asserted unchanged).

!!! note
    Crawling and lying down are only possible because of these boxes. Upstream itself is not modified.

## Run it

After [Setup](setup.md):

```bash
.venv/bin/python -I scripts/reward_hacking/make_hile_repo.py external/Open_Duck_Playground external/Playground_hile
nice -n 10 .venv/bin/python -I scripts/reward_hacking/train_hile.py external/Playground_hile runs/hizli \
  --variant hizli --restore results/2026-10-24/params/ckpt_068413440.pkl \
  --devices 5 --envs_per_dev 256 --num_timesteps 9000000 --num_evals 4 --render_python .venv/bin/python
```

`--variant` is one of `hizli` (9 M steps in our run), `hizli_serbest` (5 M), `enerji` (4 M), `ayak` (6 M). The patched model's sha256 starts with `d02225da52343b13`, the same as recorded in every `run.json`.

[`render_ckpt_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_ckpt_hile.py) and [`render_grid_hile.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/reward_hacking/render_grid_hile.py) are the evaluators used for the numbers above: training physics, no render-only collision, extra measurements (torque², feet-in-air fraction, maximum tilt).

!!! warning "Pickles"
    `ckpt_068413440.pkl` is a Python pickle of the Brax parameters. Only load pickles you trust.

## Write your own reward

Add an entry to `VARIANTS` in `hile_env.py`:

```python
"kibar": dict(scales=dict(torques=-0.01, action_rate=-2.0, imitation=2.0), terminate=True,
              note="my experiment: smoother, more imitation"),
```

Any weight you leave out keeps its official value (`BASE`). Then run `train_hile.py --variant kibar`. 20–40 minutes is usually enough to see what PPO makes of it. More ideas: [What can you try?](experiments.md)

## Data

Everything is in [`results/2026-10-24-reward-hacking/`](https://github.com/fport/open-duck-mac/tree/main/results/2026-10-24-reward-hacking): `summary.json`, and per variant `run.json` (weights, timings, source hashes), `train_log.csv`, `train_metrics_ep.jsonl` (every reward term per log step), the ONNX policy at each checkpoint and `sim/*.json` (every rollout measurement, including the 9 noisy seeds).

The `run.json` hashes of `hile_env.py` and `train_hile.py` refer to the files as they were during training; since then only comments and docstrings were edited, and `train_hile.py` gained one `sys.path` line to find `scripts/onnx_export.py`.

## Limits

One seed and one short run per variant (4–9 M steps), all continuing from a walking policy. Another seed may find a different hack.

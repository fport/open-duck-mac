<div align="center">

English · [Türkçe](README.tr.md)

<img src=".github/assets/hero.jpg" width="100%" alt="Open Duck Mini v2 in MuJoCo, three stills: an untrained policy fallen over, our policy after 68.4 M steps mid-step, and the pretrained BEST_WALK_ONNX policy">

<sub>Left: 0 steps, with exploration noise: falls. Middle: our policy at 68.4 M steps, the first checkpoint that lifts its feet. Right: the pretrained <code>BEST_WALK_ONNX</code> (not trained by us). Simulation only.</sub>

# open-duck-mac

**Teach Open Duck Mini v2 to walk from scratch on a MacBook, CPU only.<br>Then break it, and measure why it breaks.**

[![Simulation: MuJoCo MJX](https://img.shields.io/badge/simulation-MuJoCo_MJX-FFAF00)](https://github.com/google-deepmind/mujoco)
[![Apple Silicon, CPU only](https://img.shields.io/badge/Apple_Silicon-CPU_only-111111?logo=apple&logoColor=white)](https://fport.github.io/open-duck-mac/en/part-1-walking/#why-cpu)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docs: TR / EN](https://img.shields.io/badge/docs-TR_%2F_EN-FF7A3D)](https://fport.github.io/open-duck-mac/en/)
[![YouTube @getporti](https://img.shields.io/badge/YouTube-@getporti-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@getporti)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-05071A)](LICENSE)

[Docs](https://fport.github.io/open-duck-mac/en/) · [Setup](https://fport.github.io/open-duck-mac/en/setup/) · [Concepts](https://fport.github.io/open-duck-mac/en/concepts/) · [Experiments](https://fport.github.io/open-duck-mac/en/part-1-walking/) · [What can you try?](https://fport.github.io/open-duck-mac/en/experiments/) · [Videos](https://www.youtube.com/@getporti)

</div>

---

[Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) learns to walk with PPO in [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) on an Apple Silicon Mac: JAX on the CPU, MJX physics, Brax PPO. The robot, environment and rewards are Antoine Pirrone's, unchanged. What is ours: a Mac runner, a TensorFlow-free ONNX exporter, evaluators, and the logs of every run behind our videos.

The [documentation](https://fport.github.io/open-duck-mac/en/) (Turkish and English) takes you from an empty Mac to your own experiments: setup, concepts without the maths, every experiment, ideas to try.

## Videos

Short videos in Turkish on [YouTube @getporti](https://www.youtube.com/@getporti). Each one is an experiment in this repository.

| | Video | Experiment |
|---|---|---|
| Duck 1 | [I didn't buy the viral duck, I trained its sibling on my Mac](https://youtube.com/shorts/gfCUTIHldt0) | [1. Learning to walk](https://fport.github.io/open-duck-mac/en/part-1-walking/) |
| Duck 2 | "I told the duck: be fast. It learned to crawl." (28 Oct 2026) | [2. Reward hacking](https://fport.github.io/open-duck-mac/en/part-2-reward-hacking/) |
| Duck 3 | coming: the simulator lies | [3. Sim2real](https://fport.github.io/open-duck-mac/en/part-3-sim2real/) (in progress) |

## Results

| | |
|---|---|
| Learning to walk | **72.4 M steps in 3.57 h** on an M5 MacBook (6 CPU cores, 1,536 parallel ducks). Stood up in ~11 min, plateaued ~3 h, first lifted its feet at **68.4 M** steps: 29.8 cm in 5 s (pretrained policy: 58.5 cm). |
| Reward hacking | Same policy, only the reward changed. "Be fast" → crawls on its belly (14.9 min). "Don't waste energy" → freezes (7.5 min). "Lift your feet" → lies down with its feet in the air (30 min). |
| Sim2real | The simulated motor (±3.23 N·m) is 70–75 % stronger than the real STS3215 servo (~1.86 N·m). At the real limit the same policy covers **half the distance**; slippery floors and extra mass barely matter. |

One seed, one run each, simulation only, no real robot. The Mac's GPU cannot run MJX (`jax-metal` lacks `cholesky`), so everything runs on the CPU; splitting it into 6 single-threaded virtual devices doubles the speed.

## Quick start

```bash
git clone https://github.com/fport/open-duck-mac.git && cd open-duck-mac
git clone https://github.com/apirrone/Open_Duck_Playground.git external/Open_Duck_Playground
git -C external/Open_Duck_Playground checkout b9be205ac64488c23504ca42e5ec790337adeec3

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
mkdir -p "$(.venv/bin/python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mujoco_playground").origin))')/external_deps/mujoco_menagerie"

# measure our trained policy (prints "dx_m": 0.2882)
.venv/bin/python -I scripts/eval_checkpoint.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/c72.mp4

# train (our full run; after 10 M steps, ≈ 30 min, it already stands)
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/my-run \
  --devices 6 --envs_per_dev 256 --num_timesteps 72000000 --num_evals 19 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

Step-by-step setup, troubleshooting and every option: **[documentation](https://fport.github.io/open-duck-mac/en/setup/)**.

## Repository

```
scripts/                 train_mac.py, onnx_export.py, eval_checkpoint.py, eval_grid.py, make_log.py, smoke*.py
scripts/reward_hacking/  part 2: reward file (hile_env.py), trainer, collision-box patch, evaluators
scripts/sim2real/        part 3: eval_world.py (change torque, friction, mass)
results/2026-10-24/      part 1: logs, checkpoints table, every rollout measurement, ONNX policies, warm-start params
results/2026-10-24-reward-hacking/   part 2: every log and measurement of the four variants
docs/                    the documentation site (Astro Starlight, Turkish + English)
```

## Credits and licence

**Open Duck Mini v2** and **Open_Duck_Playground** by [Antoine Pirrone](https://github.com/apirrone) and contributors: robot, model, environment, rewards, reference motion and the pretrained `BEST_WALK_ONNX`. [MuJoCo / MJX](https://github.com/google-deepmind/mujoco), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and [Brax](https://github.com/google/brax) by Google DeepMind and contributors. If you build on this, credit Open Duck Mini first: it is their robot and their environment.

Code and `results/` are released under the [Apache License 2.0](LICENSE). Full credits: [documentation](https://fport.github.io/open-duck-mac/en/reference/).

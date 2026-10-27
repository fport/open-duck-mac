---
title: 'Setup: from an empty Mac to your first training run'
---

This page takes you from an Apple Silicon Mac with nothing installed to the duck's first training run. We only ran this on macOS. Linux CPUs should work too, but we have not tested them.

## What you need

| | |
|---|---|
| Computer | Apple Silicon Mac (M1 or later). Ours is an M5 with 10 cores. |
| Disk | A few GB (Python environment + checkpoints). |
| Time | ~15 min of setup. The full training run took 3.6 h on our machine; first results show up after 10–15 minutes. |
| Knowledge | Being able to paste commands into a terminal is enough. Python helps but is not required. |

:::note[Why not the GPU?]
The Mac's GPU (Metal) cannot run this physics: `jax-metal` does not support the `cholesky` operation MJX needs (`failed to legalize operation 'mhlo.cholesky'`). That is missing software support, not a hardware limit, so everything runs on the CPU. Details: [1. Learning to walk](/open-duck-mac/en/part-1-walking/#why-cpu).
:::

## 1. Basic tools

```bash
# Git and compiler tools (says "already installed" if present)
xcode-select --install

# uv: a fast Python and package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional: `ffmpeg` for the 9-duck grid video (`brew install ffmpeg`). Without it the scripts write measurements only and skip the mosaic.

## 2. Get this repository and the environment

```bash
git clone https://github.com/fport/open-duck-mac.git
cd open-duck-mac

git clone https://github.com/apirrone/Open_Duck_Playground.git external/Open_Duck_Playground
git -C external/Open_Duck_Playground checkout b9be205ac64488c23504ca42e5ec790337adeec3
```

`b9be205` (5 Aug 2025, "fix onnx export") is the commit we trained on. Stay on it to reproduce our numbers. `external/` is in `.gitignore`.

## 3. Python environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` is the exact environment of our run; `requirements.txt` lists only the direct dependencies.

MuJoCo Playground tries to clone a large robot library (`mujoco_menagerie`) on first import; this robot does not need it. An empty folder skips the clone:

```bash
mkdir -p "$(.venv/bin/python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mujoco_playground").origin))')/external_deps/mujoco_menagerie"
```

## 4. Smoke test (1–2 minutes)

```bash
.venv/bin/python -I scripts/smoke.py external/Open_Duck_Playground --envs 64
.venv/bin/python -I scripts/smoke_pmap.py external/Open_Duck_Playground --devices 6 --envs_per_dev 256
```

The first shows that the environment compiles and steps. The second splits the CPU into D virtual devices and prints steps per second. Try `--devices` 4, 6 and 8 and pick the fastest, leaving a few cores free. On our machine: 1 device × 1,024 envs = 2,368 steps/s; 6 × 256 = 4,621 steps/s.

## 5. Try before training: our policies

Training takes hours. Measure our trained policy first:

```bash
.venv/bin/python -I scripts/eval_checkpoint.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/c72.mp4
```

You should see `"dx_m": 0.2882` (28.8 cm in 5 s). `out/c72.mp4` is the video. Add `--no_video` for numbers only.

## 6. Your first training run

```bash
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/first \
  --devices 6 --envs_per_dev 256 --num_timesteps 10000000 --num_evals 4 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

10 M steps take ~30 minutes on our machine. The duck learns to stand in that time but does not walk yet; walking started around 70 M. Our full run:

```bash
nice -n 10 .venv/bin/python -I scripts/train_mac.py external/Open_Duck_Playground runs/my-run \
  --devices 6 --envs_per_dev 256 --num_timesteps 72000000 --num_evals 19 --seed 0 \
  --render_script scripts/eval_checkpoint.py
```

After every chunk (~9 min) you get a `.onnx` and `.pkl` in `runs/<name>/ckpt/`, a line in `checkpoints.csv` and, with `--render_script`, a 5 s video + measurements in the background. Drop `--render_script` if the CPU is busy.

:::tip[Keep using your computer]
`nice -n 10` gives training a low priority, so your browser and editor stay responsive. Closing the lid puts the Mac to sleep and pauses training; for long runs you can prefix the command with `caffeinate -i`.
:::

## 7. Plot and logs

```bash
.venv/bin/python scripts/make_log.py runs/my-run
```

This writes `train_log.csv`, `ckpt_summary.json` and `reward_curve.png`. [Concepts](/open-duck-mac/en/concepts/) and [part 1](/open-duck-mac/en/part-1-walking/) explain how to read them.

If something goes wrong: [Troubleshooting & FAQ](/open-duck-mac/en/faq/).

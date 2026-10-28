---
title: "Try it in 5 minutes"
description: "Install with one command, watch the walking duck, weaken its motor, train your own."
---

This page does everything with one command, `./duck`. The long way, and what each step does, is on [Setup](/open-duck-mac/en/setup/).

## What you need

- An Apple Silicon Mac (M1 or later). We tested on macOS only.
- `git` and [`uv`](https://docs.astral.sh/uv/). If you don't have them:

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal after installing `uv`.

## 1. Install (once, 3–5 min)

```bash
git clone https://github.com/fport/open-duck-mac.git
cd open-duck-mac
./duck setup
```

What it does:

1. Downloads the duck's simulation ([Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)) into `external/`, at the commit our results used (`b9be205`).
2. Installs a Python 3.12 environment in `.venv/` (`requirements-lock.txt`, a few hundred MB the first time).
3. Smoke test: does the simulation build and step?
4. Splits the CPU into 4, 6 and 8 parts, measures which trains fastest and saves it in `.duck.json`.

If something goes wrong, `./duck doctor` says what is missing and how to fix it.

## 2. Watch the walking duck (a few seconds)

```bash
./duck try
```

It walks the duck we trained for 72.4 M steps for 5 seconds, opens the video and measures it:

```text
  distance (5 s): 28.8 cm  ·  did not fall
  feet (L / R): 8×, 11 mm / 8×, 11 mm  ·  alternation -0.34  →  WALKS
```

The "WALKS" verdict comes from [`gait.py`](/open-duck-mac/en/part-3-sim2real/#walking-or-sliding-the-step-counter-was-wrong): at least 10 cm, and at least 4 oscillations and 5 mm of lift on both feet. A negative alternation means the feet move in turn.

## 3. Break the world (under a second each)

The same duck with its motor lowered to the rating on the box ([part 3](/open-duck-mac/en/part-3-sim2real/)):

```bash
./duck eval duck1 --force 1.86 --no-video      # 13.9 cm: half the distance
./duck eval catalog --force 1.86 --no-video    # 37.6 cm: a duck trained for that motor
./duck eval forgot --force 1.86 --no-video     # 2.5 cm: same training, forgot by 301 M
```

The options combine; `./duck eval --list` shows them all:

| Option | What changes |
|---|---|
| `--force 1.86` | every motor at most 1.86 N·m (catalog rating; the sim uses 3.23) |
| `--force-rand 1.4 1.9` | every motor random in that range (`--world-seed` picks the draw) |
| `--mass 1.2` | every part 20 % heavier |
| `--friction 0.5` | floor friction (normally 0.6) |
| `--vx 0.3` | a faster walk command (normally 0.15 m/s) |
| `--stochastic --seed 3` | noisy actions, as during training |

Find the limit: the duck falls within 1 s at `--mass 3` and 1.3 s at `--force 0.4`; at `--friction 0.05` it slips in place. Where is the threshold in between?

Without `--no-video` it also renders a video into `out/eval/` (a few seconds); `--open` opens it.

## 4. Train your own duck

```bash
./duck train --preset quick --detach    # 10 M steps, ~30 min on our machine
./duck watch                            # lists your runs
./duck watch quick-1028-1430            # tries each new checkpoint: does it walk?
```

| Preset | Steps | Time (our M5) | What to expect |
|---|---|---|---|
| `quick` | 10 M | ~30 min | learns to stand, does not walk yet |
| `full` | 72 M | ~3.6 h | the part 1 recipe; first steps around 68 M |
| `sim2real` | 150 M | ~7.5 h | motor strength random in 1.40–1.90 N·m; ours walked at 88.8 M |

- `--detach` runs the training in the background; you can close the terminal. Closing the lid sleeps the Mac and pauses training.
- Training runs at low priority (`nice`), so the browser and editor stay smooth, but most cores are busy.
- One training at a time: a second one slows both down, and `./duck train` won't start it.
- `--seed 7` tries another seed; `--steps 20` changes the number of steps (millions).
- The training reward does not show walking ([in part 3](/open-duck-mac/en/part-3-sim2real/#continuing-to-301-m) the gait disappeared while the reward held). That's why `./duck watch` tries every checkpoint.

## With Claude Code

This repo ships a [Claude Code](https://claude.com/claude-code) skill: [`.claude/skills/open-duck-lab/SKILL.md`](https://github.com/fport/open-duck-mac/blob/main/.claude/skills/open-duck-lab/SKILL.md). It loads by itself when you open the repo in Claude Code. Then ask things like "install the duck and try it", "what happens with a weaker motor?" or "start a short training", and it runs the `./duck` commands. Before a long training it tells you the time and asks first.

## What next?

- [What can you try?](/open-duck-mac/en/experiments/): 10 experiments, easy to hard.
- [Concepts](/open-duck-mac/en/concepts/): policy, reward, checkpoint, domain randomization.
- Stuck: `./duck doctor`, then [Troubleshooting & FAQ](/open-duck-mac/en/faq/).

---
name: open-duck-lab
description: Set up and run Open Duck Mini v2 experiments on a Mac CPU with this repo's ./duck command, and explain the results. Covers install, the smoke test, watching the pretrained duck, testing a policy in a changed world (weaker motor, heavier body, slippery floor), training your own duck and checking which checkpoints walk. Use when someone asks to install, try, train, evaluate or compare the duck, to run an experiment from the docs, or what a number here means.
---

# Open Duck lab

This repo trains the Open Duck Mini v2 (a small open-source biped) to walk in simulation on an Apple Silicon Mac,
CPU only (JAX + MuJoCo MJX + Brax PPO). Everything here is **simulation**: nobody has run these policies on a real
robot. Say so whenever a result could be read as a hardware result. Answer in the user's language.

## The path for a newcomer

| Step | Command | Time on our M5 | What they get |
|---|---|---|---|
| 1 | `./duck setup` | 3–5 min | Python env, the pinned simulation, a smoke test, the fastest CPU layout saved in `.duck.json` |
| 2 | `./duck try` | a few seconds | a 5 s video of our walking duck + distance + "walks / does not walk" |
| 3 | `./duck eval duck1 --force 1.86 --no-video` | under a second | the same duck with the motor at its catalog value: about half the distance |
| 4 | `./duck train --preset quick --detach` | ~30 min | their own duck; it learns to stand, not yet to walk |
| 5 | `./duck watch <run>` | a second per checkpoint | a table: distance and whether each checkpoint walks |

Start with `./duck doctor` when anything fails; it prints the fix. Then the FAQ: `docs/src/content/docs/en/faq.md`.

## Commands

- `./duck setup [--no-bench]` idempotent; safe to rerun. Never changes `external/Open_Duck_Playground` if it already exists.
- `./duck doctor` system, install, CPU layout, and whether a training is running.
- `./duck try [--no-open]`
- `./duck eval <policy> [--force N·m] [--force-rand LO HI --world-seed S] [--mass X] [--friction MU] [--vx M/S] [--stochastic --seed S] [--no-video] [--open]`
  - policies: `duck1` (part 1, 72.4 M, walks), `duck1-first` (68.4 M, first steps), `catalog` (part 3, 137.8 M, walks at 1.86 N·m), `forgot` (part 3, 301.2 M, stands), or any `.onnx` path such as `runs/<run>/ckpt/ckpt_*.onnx`.
  - `./duck eval --list` prints both lists.
- `./duck train --preset quick|full|sim2real [--name N] [--steps MILLIONS] [--chunks K] [--seed S] [--detach]`
  - quick 10 M (~30 min), full 72 M (~3.6 h, first steps around 68 M), sim2real 150 M with motor strength U(1.40, 1.90) N·m (~7.5 h; ours first walked at 88.8 M).
  - runs go to `runs/<name>/` (gitignored): `train.log`, `ckpt/ckpt_<steps>.{onnx,pkl}`, `checkpoints.csv`, `train_log_raw.csv`.
- `./duck watch [<run>] [--once] [--force sim|N·m]` without a run it lists runs. Default motor for the test: 1.86 for sim2real runs, the sim's 3.23 otherwise.

## Rules

- **Ask before any training longer than ~30 minutes**: say the preset, the estimated time (the command prints it) and that it uses most CPU cores. Prefer `--detach` for anything long, then `./duck watch`.
- One training at a time. `./duck train` refuses if one is running; do not add `--force` unless the user asks.
- A closed lid sleeps the Mac and pauses training (`caffeinate -i` only prevents idle sleep).
- Do not edit `external/Open_Duck_Playground`: every number in `results/` used commit `b9be205`. Reward experiments use their own copy (part 2, `scripts/reward_hacking/`).
- Report numbers as measured, with the test: deterministic 5 s, flat floor, no pushes, forward command 0.15 m/s.

## Reading a result

- **distance**: forward travel in 5 s. Reference values from `results/`: `duck1` 28.8 cm; `duck1 --force 1.86` 13.9 cm; `catalog --force 1.86` 37.6 cm; `forgot --force 1.86` 2.5 cm.
- **walks / does not walk** comes from `scripts/sim2real/gait.py`: at least 10 cm, and both feet oscillate (≥ 4 cycles, ≥ 5 mm lift). The left/right correlation is negative when the feet alternate. A duck can cover 25 cm by jittering one foot; that is not walking.
- **The training reward is not a walking test.** In part 3 the mean reward did not drop while the gait disappeared after 170 M. Always test checkpoints with `./duck watch` or `./duck eval`.
- `fell` is a trunk tipped > 60° or below 9 cm. No policy here falls on a flat floor; a weak motor slows the duck down instead.
- A fresh `quick` run standing still at 10 M is expected, not a bug.

## Experiment menu (docs: `docs/src/content/docs/en/experiments.md`)

| Experiment | How | Time |
|---|---|---|
| Break the world, measure the policy | `./duck eval duck1 ...` with `--force`, `--mass`, `--friction`; compare with `catalog` | seconds |
| Walk at another speed | `./duck eval duck1 --vx 0.3` | seconds |
| Noise as in training | `./duck eval duck1 --stochastic --seed 0..5` | seconds |
| Your own seed | `./duck train --preset quick --seed 7`, then `--preset full` | 30 min – 3.6 h |
| Weak motor from scratch | `./duck train --preset sim2real --detach`; watch from 80 M on | ~7.5 h |
| Your own reward | part 2 page, `scripts/reward_hacking/hile_env.py` (`VARIANTS`) | 20–40 min |

## Where things are

- `scripts/duck.py` the CLI (stdlib only) · `scripts/train_mac.py`, `scripts/sim2real/` the experiment scripts
- `results/` our runs, with the ONNX files the `eval` names point to
- `docs/src/content/docs/` the documentation (Turkish at the top level, English in `en/`); part 1 walking, part 2 reward hacking, part 3 sim2real

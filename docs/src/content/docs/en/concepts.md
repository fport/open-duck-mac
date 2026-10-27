---
title: 'Concepts, without the maths'
---

This page explains the terms used in the videos and maps each one to a file in the repository. Read it in order and every table in the later parts will make sense.

## Reinforcement learning (RL)

We do **not** write a walking motion for the robot. We only give it a **score** (reward) and let it find, by trial and error, the movements that raise that score.

| Term | What it is for the duck |
|---|---|
| **Environment** | The virtual duck + floor + gravity in MuJoCo. File: `joystick.py` in Open_Duck_Playground. |
| **Observation** | The 101 numbers the duck "sees" each step: angular velocity and acceleration from the IMU, the command (forward/turn speed and head targets), angle and velocity of the 14 joints, its last three actions and motor targets, whether each foot touches the floor, and its phase in the reference gait. |
| **Action** | Target angles for 14 motors: 5 per leg, 4 for the neck and head. |
| **Step** | 0.02 s. The duck decides 50 times per second. |
| **Episode** | One attempt. Ends when the duck falls or after 20 s, then starts again. |
| **Policy** | The small neural network that turns an observation into an action. This is what gets trained. |
| **Reward** | The score given every step: a weighted sum of a few terms. |

## The official reward: 7 terms

In Open Duck's walking environment the reward has these 7 parts (`joystick.py`, `reward_config.scales`):

| Term | Weight | Rewards / penalises |
|---|---|---|
| `tracking_lin_vel` | 2.5 | Moving at the requested speed |
| `tracking_ang_vel` | 6.0 | Turning at the requested rate |
| `alive` | 20.0 | Not falling |
| `imitation` | 1.0 | Imitating a reference gait |
| `torques` | −0.001 | Straining the motors (penalty) |
| `action_rate` | −0.5 | Sudden, jittery motion (penalty) |
| `stand_still` | −0.2 | Moving when told to stand (penalty) |

In [part 2](/open-duck-mac/en/part-2-reward-hacking/) we break these weights and watch what happens.

## Imitation reward

With only a "go forward" reward, the duck can find very strange solutions. Open Duck uses the idea from Disney's BDX paper: a **reference gait** is generated (a parametric walk engine, `polynomial_coefficients.pkl`) and the duck earns points the closer its joints follow it. The duck still learns by itself; the reference only says what walking should look like.

## PPO and parallel ducks

![1,536-tile mosaic: one duck per tile; an orange tile is a duck that fell and restarts](../../../assets/parallel-ducks.jpg)

<sub>Representative: 9 noisy recordings of the step-0 policy repeated across 1,536 tiles. Orange = the duck fell, the episode ended, it restarts.</sub>

**PPO** (Proximal Policy Optimization) is the algorithm that updates the policy. Roughly: let the ducks act for a while, see which actions earned more than expected, nudge the policy that way with a **small** step, repeat. The "small" part matters: big steps destroy what was learned.

With one duck this would be very slow, so we simulate **1,536 ducks** at once (6 virtual devices × 256). They share one policy; because each is in a different random state, PPO sees a wide range of experience at once. "72.4 M steps" is the total over all ducks.

## MuJoCo, MJX, JAX

- **MuJoCo**: the physics engine. Mass, friction, contacts, motors.
- **MJX**: MuJoCo written in **JAX**. It runs thousands of ducks as one large computation.
- **JAX**: Google's numerical library. It compiles code (XLA) to run fast on a CPU or GPU.
- **Brax**: PPO written in JAX. This is our training loop.

## Domain randomization

The simulator is not a perfect copy of reality. So that the policy does not memorise one perfect world, the world is changed a little in every episode. In the official training (`playground/common/randomize.py`):

- floor friction: 0.5–1.0
- mass of every part: ×0.9–1.1; ±0.1 kg on the trunk
- trunk centre of mass: ±5 cm
- motor stiffness (kp): ×0.9–1.1; joint friction ×0.9–1.1
- joint zero offsets: ±0.03 rad
- also: action and IMU delay (0–3 steps), observation noise, random pushes

One important thing is **not** randomized: the maximum torque a motor can produce. [Part 3](/open-duck-mac/en/part-3-sim2real/) is about exactly that.

## Checkpoints, ONNX, warm start

- **Checkpoint**: a snapshot of the policy at one point in training. `ckpt_068413440` = the duck at step 68,413,440.
- **`.pkl`**: Brax's weights (to continue training).
- **`.onnx`**: the same policy in a standard format that runs anywhere (evaluation, a real robot).
- **Warm start**: starting a new training run from a checkpoint instead of from scratch. Parts 2 and 3 all start from 68.4 M.

## Two kinds of test

| | Deterministic test | 9 noisy ducks |
|---|---|---|
| Script | `eval_checkpoint.py` | `eval_grid.py` |
| How | The policy's most likely action; same result every time | With the exploration noise used in training, 9 seeds |
| Tells you | "What does this duck do in clean conditions?" | "How robust is it? How many fall?" |

Both: 5 s, flat floor, no pushes, forward command 0.15 m/s. "Fell" = trunk tipped more than 60° (stricter than training's 90° rule).

## Sim2real

Moving a policy learned in simulation onto a real robot. Every difference in between (motor strength, delay, mass, friction) makes the robot behave differently in reality. Domain randomization is the most common way to close that gap; measuring the motors accurately (system identification) is the other. The Open Duck team measures its motors with [BAM](https://github.com/Rhoban/bam).

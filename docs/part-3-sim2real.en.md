# 3. The simulator lies (sim2real)

!!! info "In progress"
    The measurement part of this chapter is done; the retraining runs (27 October 2026) are still going. This page will be updated with their results and scripts.

## The question

Is the simulated duck the same as the real one? In the Open Duck model every joint can produce at most **±3.23 N·m** of torque (`open_duck_mini_v2.xml`, class `sts3215`, `forcerange`). For the real STS3215 servo:

- a community project ([soyeon24/open-duck-mini-v2-build](https://github.com/soyeon24/open-duck-mini-v2-build)) reports that the servo stalls at **~1.86 N·m**;
- retailers list a peak (stall) torque of **19.5 kg·cm ≈ 1.91 N·m** ([Core Electronics](https://core-electronics.com.au/feetech-sts3215-smart-servo.html), [RobotShop](https://www.robotshop.com/products/magnetic-encoding-servo-sts3215-74v-19kgcm)); sources disagree on the voltage it applies to.

So the simulated motor is **70–75 % stronger** than the real one. The official training randomizes floor friction, mass, motor stiffness and delays ([Concepts](concepts.md#domain-randomization)) but **not the torque limit**.

## Experiment 1: no retraining, different worlds { #experiment-1 }

We took the trained policies unchanged and only changed the world. Deterministic test, 5 s, flat floor, no pushes, 0.15 m/s.

| World | Our 72.4 M | 9 noisy (ours) | Pretrained `BEST_WALK_ONNX` |
|---|---|---|---|
| Normal (3.23 N·m) | 28.8 cm | 0/9 fell, mean 22.1 cm | 58.5 cm · 8+8 steps |
| Slippery floor (friction 0.5) | 24.9 cm | 0/9, mean 20.6 cm | 57.3 cm |
| Mass ×1.1 | 27.9 cm | 0/9, mean 18.1 cm | 56.1 cm |
| Mass ×1.2 | 24.7 cm | – | 54.3 cm |
| Torque 2.5 N·m | 25.3 cm | – | 52.3 cm |
| **Torque 1.86 N·m** | **13.9 cm (−52 %)** | 0/9, mean 9.2 cm | **40.2 cm (−31 %)** · 8+5 steps |
| Torque 1.6 N·m | 9.8 cm | – | 27.3 cm · 8+0 steps |
| Torque 1.4 N·m | 5.8 cm | – | 19.5 cm · 7+0 steps |

What we see:

- What training already randomizes (friction, mass) barely affects the policy, even mass ×1.2, which is outside the training range (×0.9–1.1).
- The torque limit, never randomized in training, halves the distance.
- No duck falls over; a weak motor slows it down instead of tipping it. soyeon24 reports the same.
- At 1.6 N·m the pretrained policy no longer steps with its right foot (8+0).

To reproduce, [`scripts/sim2real/eval_world.py`](https://github.com/fport/open-duck-mac/blob/main/scripts/sim2real/eval_world.py) is a copy of `eval_checkpoint.py` that changes the world through environment variables:

```bash
FORCE=1.86 .venv/bin/python -I scripts/sim2real/eval_world.py external/Open_Duck_Playground \
  results/2026-10-24/onnx/ckpt_072437760.onnx out/f186.mp4 --no_video        # "dx_m": 0.1393
FLOOR_FRIC=0.5 .venv/bin/python -I scripts/sim2real/eval_world.py ...            # slippery floor
MASS_SCALE=1.2 .venv/bin/python -I scripts/sim2real/eval_world.py ...            # every part 20 % heavier
FORCE_RAND=1.40,1.90 WSEED=3 .venv/bin/python -I scripts/sim2real/eval_world.py ...  # random per joint
```

The chosen world is written to the `"world"` field of the output JSON.

## Experiment 2: retraining with a weak motor (running)

Three runs, all warm-started from 68.4 M, 10 M steps each; rewards, physics and the official randomization unchanged, only the torque limit differs:

| Run | Torque limit | Why |
|---|---|---|
| `tork186` | fixed 1.86 N·m on every joint | match the simulator to the real servo |
| `tork_rand` | per episode and per joint, U(1.40, 1.90) N·m | domain randomization: don't memorise one value |
| `kontrol323` | official 3.23 N·m | fair comparison: is any gain just from more training? |

All three will then be measured in the worlds of Experiment 1.

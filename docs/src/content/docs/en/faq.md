---
title: 'Troubleshooting & FAQ'
---

## Setup

<details>
<summary><code>failed to legalize operation 'mhlo.cholesky'</code></summary>

`jax-metal` (Mac GPU) is installed and JAX is using it. This physics does not run on Metal. Use a clean `.venv` installed from `requirements-lock.txt`; it must not contain `jax-metal`.

</details>

<details>
<summary>The first import takes very long or downloads something</summary>

MuJoCo Playground is trying to clone `mujoco_menagerie`. Run the empty-folder command from [Setup](/open-duck-mac/en/setup/#3-python-environment).

</details>

<details>
<summary><code>ModuleNotFoundError: playground</code></summary>

Make sure the first argument to the scripts is the Open_Duck_Playground folder (`external/Open_Duck_Playground`). The scripts load the `playground` package from there.

</details>

<details>
<summary>Why <code>python -I</code> everywhere?</summary>

`-I` (isolated mode) stops Python from loading stray modules from the current folder or environment variables. It's safer when you run inside a downloaded folder.

</details>

## Training

<details>
<summary>Steps per second are very low</summary>

Try `--devices` 4, 6 and 8 with `smoke_pmap.py`. Heavy background work (video rendering, browser tabs) can halve the speed; in our run chunks ranged from 1,924 to 7,358 steps/s.

</details>

<details>
<summary>Nothing seems to happen after the first checkpoint</summary>

The first JAX compile takes a few minutes. After that each chunk takes ~9 minutes on our machine. `checkpoints.csv` and the `steps=` lines in the terminal show progress.

</details>

<details>
<summary>The duck hasn't walked for hours. Is something broken?</summary>

Probably not. Ours learned to stand in 11 minutes, then sat on a plateau for ~3 hours and only lifted its feet at 68.4 M steps. If episode return and the imitation term keep rising, it is learning.

</details>

<details>
<summary>Can I stop and resume training?</summary>

`train_mac.py` has no resume option. `--restore <ckpt.pkl>` in `reward_hacking/train_hile.py` continues from a checkpoint (policy and observation normalisation; value network from scratch).

</details>

<details>
<summary>The Mac went to sleep and training stopped</summary>

Closing the lid puts the Mac to sleep. Prefix long runs with `caffeinate -i`.

</details>

## Results

<details>
<summary>My numbers differ slightly from yours</summary>

Evaluation runs in C MuJoCo and training in MJX; different JAX/MuJoCo versions or CPUs can cause small numerical differences. `requirements-lock.txt` and the pinned commit (`b9be205`) get you closest. Training itself can take a different path depending on seed and hardware.

</details>

<details>
<summary>Can I run this policy on a real Open Duck Mini?</summary>

The ONNX files have the upstream interface, but we have not tried hardware. Remember that the simulated motor is stronger than the real one: [part 3](/open-duck-mac/en/part-3-sim2real/).

</details>

<details>
<summary>Is this the same as Microduck?</summary>

No. Microduck is a separate product by Pollen Robotics (part of Hugging Face); this repository uses Open Duck Mini v2. Differences and what you can do on a Mac: [Microduck](/open-duck-mac/en/microduck/).

</details>

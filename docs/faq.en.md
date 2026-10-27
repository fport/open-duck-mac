# Troubleshooting & FAQ

## Setup

??? question "`failed to legalize operation 'mhlo.cholesky'`"
    `jax-metal` (Mac GPU) is installed and JAX is using it. This physics does not run on Metal. Use a clean `.venv` installed from `requirements-lock.txt`; it must not contain `jax-metal`.

??? question "The first import takes very long or downloads something"
    MuJoCo Playground is trying to clone `mujoco_menagerie`. Run the empty-folder command from [Setup](setup.md#3-python-environment).

??? question "`ModuleNotFoundError: playground`"
    Make sure the first argument to the scripts is the Open_Duck_Playground folder (`external/Open_Duck_Playground`). The scripts load the `playground` package from there.

??? question "Why `python -I` everywhere?"
    `-I` (isolated mode) stops Python from loading stray modules from the current folder or environment variables. It's safer when you run inside a downloaded folder.

## Training

??? question "Steps per second are very low"
    Try `--devices` 4, 6 and 8 with `smoke_pmap.py`. Heavy background work (video rendering, browser tabs) can halve the speed; in our run chunks ranged from 1,924 to 7,358 steps/s.

??? question "Nothing seems to happen after the first checkpoint"
    The first JAX compile takes a few minutes. After that each chunk takes ~9 minutes on our machine. `checkpoints.csv` and the `steps=` lines in the terminal show progress.

??? question "The duck hasn't walked for hours. Is something broken?"
    Probably not. Ours learned to stand in 11 minutes, then sat on a plateau for ~3 hours and only lifted its feet at 68.4 M steps. If episode return and the imitation term keep rising, it is learning.

??? question "Can I stop and resume training?"
    `train_mac.py` has no resume option. `--restore <ckpt.pkl>` in `reward_hacking/train_hile.py` continues from a checkpoint (policy and observation normalisation; value network from scratch).

??? question "The Mac went to sleep and training stopped"
    Closing the lid puts the Mac to sleep. Prefix long runs with `caffeinate -i`.

## Results

??? question "My numbers differ slightly from yours"
    Evaluation runs in C MuJoCo and training in MJX; different JAX/MuJoCo versions or CPUs can cause small numerical differences. `requirements-lock.txt` and the pinned commit (`b9be205`) get you closest. Training itself can take a different path depending on seed and hardware.

??? question "Can I run this policy on a real Open Duck Mini?"
    The ONNX files have the upstream interface, but we have not tried hardware. Remember that the simulated motor is stronger than the real one: [part 3](part-3-sim2real.md).

??? question "Is this the same as Microduck?"
    No. Microduck is a separate product by Hugging Face / Pollen Robotics. This repository uses Open Duck Mini v2.

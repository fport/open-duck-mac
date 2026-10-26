"""Train Open Duck Mini v2 to walk FROM SCRATCH on an Apple Silicon Mac: JAX CPU + MJX + Brax PPO.

A Mac adaptation of Open_Duck_Playground's runner.py. The environment is used UNCHANGED:
  * env: playground/open_duck_mini_v2/joystick.py (task "flat_terrain"); rewards, pushes and domain
    randomization untouched (imitation reward on; polynomial_coefficients.pkl ships with the repo)
  * PPO: the "BerkeleyHumanoidJoystickFlatTerrain" hyperparameters from mujoco_playground (the ones
    runner.py uses); only the scale changes for CPU: num_envs, batch_size, num_timesteps, num_evals
  * JAX on CPU with D virtual devices (XLA host-platform device count) and one thread per device, so
    Brax pmaps over D cores. On a 10-core M-series Mac, 6 devices was the sweet spot for us.
  * evaluation is off (expensive on CPU); episode metrics are logged from the training rollouts
  * after every chunk: params (.pkl, numpy pytree) + ONNX (onnx_export.py, no TensorFlow needed) and,
    optionally, a background rollout render/eval (--render_script, e.g. eval_checkpoint.py)

Outputs in <out_dir>: run_config.json, train_log_raw.csv, train_metrics.jsonl, checkpoints.csv,
ckpt/ckpt_<steps>.{onnx,pkl} (or --ckpt_dir), sim/ckpt_<steps>.{mp4,json,log} when rendering.
Turn the raw log into train_log.csv + reward_curve.png with make_log.py.

Our run (2026-10-24, 72.4 M steps, 3.57 h):
  python -I scripts/train_mac.py ../Open_Duck_Playground runs/run1 --devices 6 --envs_per_dev 256 \\
      --num_timesteps 72000000 --num_evals 19 --seed 0 --render_script scripts/eval_checkpoint.py
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", help="path to an Open_Duck_Playground checkout")
ap.add_argument("out_dir", help="output directory for logs (and checkpoints unless --ckpt_dir)")
ap.add_argument("--devices", type=int, default=6, help="virtual CPU devices = cores used (default 6)")
ap.add_argument("--envs_per_dev", type=int, default=256, help="parallel envs per device (default 256)")
ap.add_argument("--num_timesteps", type=int, default=60_000_000, help="total env steps (official recipe: 150M on GPU)")
ap.add_argument("--num_evals", type=int, default=17, help="number of chunks; a checkpoint is written after each")
ap.add_argument("--num_minibatches", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt_dir", default="", help="where to write ckpt_*.onnx/.pkl (default <out_dir>/ckpt)")
ap.add_argument("--render_script", default="",
                help="optional script run in the background per checkpoint as: <python> -I <script> <repo> <onnx> <out.mp4>")
ap.add_argument("--render_python", default="", help="python for --render_script (default: this interpreter)")
args = ap.parse_args()

# Must be set before jax is imported.
os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count={args.devices} "
                           "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import csv, functools, json, pathlib, pickle, subprocess, time

REPO = str(pathlib.Path(args.repo).resolve())
OUT = pathlib.Path(args.out_dir).resolve()
OUT.mkdir(parents=True, exist_ok=True)
CKDIR = pathlib.Path(args.ckpt_dir).resolve() if args.ckpt_dir else OUT / "ckpt"
CKDIR.mkdir(parents=True, exist_ok=True)
HERE = pathlib.Path(__file__).resolve().parent
RENDER_SCRIPT = str(pathlib.Path(args.render_script).resolve()) if args.render_script else ""
RENDER_PY = args.render_python or sys.executable
sys.path.insert(0, str(HERE))
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np
import jax
from brax.training import logger as brax_logger
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from playground.common import randomize
from playground.open_duck_mini_v2 import joystick
import onnx_export

T0 = time.time()
NUM_ENVS = args.devices * args.envs_per_dev
STATE = {"episodes": 0, "steps_seen": 0}

# Episode counter: brax's EpisodeMetricsLogger receives the done mask on every unroll.
_orig_update = brax_logger.EpisodeMetricsLogger.update_episode_metrics
def _counting_update(self, metrics, dones):
    STATE["episodes"] += int(np.sum(np.asarray(dones)))
    STATE["steps_seen"] += int(np.prod(np.asarray(dones).shape))
    return _orig_update(self, metrics, dones)
brax_logger.EpisodeMetricsLogger.update_episode_metrics = _counting_update

LOG = OUT / "train_log_raw.csv"
CK = OUT / "checkpoints.csv"
TRAIN_KEYS = ["sum_reward", "length", "reward/tracking_lin_vel", "reward/tracking_ang_vel",
              "reward/alive", "reward/imitation", "reward/action_rate", "reward/torques", "reward/stand_still"]

def _w(path, row, header):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def progress(num_steps, metrics):
    el = time.time() - T0
    if "episode/sum_reward" in metrics:
        # Note: with pmap every device may call this; make_log.py de-duplicates rows per env_steps.
        vals = [float(metrics.get(f"episode/{k}", float("nan"))) for k in TRAIN_KEYS]
        _w(LOG, [time.strftime("%H:%M:%S"), round(el, 1), int(num_steps), NUM_ENVS, STATE["episodes"]] + vals,
           ["clock", "wall_s", "env_steps", "num_envs", "episodes_total"] + ["ep_" + k.replace("/", "_") for k in TRAIN_KEYS])
        print(f"[{el/60:6.1f} min] steps={num_steps:,} episodes={STATE['episodes']:,} "
              f"R={vals[0]:.2f} len={vals[1]:.0f}", flush=True)
    else:
        with open(OUT / "train_metrics.jsonl", "a") as f:
            f.write(json.dumps({"wall_s": round(el, 1), "env_steps": int(num_steps),
                                **{k: float(v) for k, v in metrics.items()}}) + "\n")

env = joystick.Joystick(task="flat_terrain")
ACT = env.action_size
ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")

def policy_params_fn(current_step, make_policy, params):
    el = time.time() - T0
    tag = f"ckpt_{int(current_step):09d}"
    onnx_path = CKDIR / f"{tag}.onnx"
    onnx_export.from_brax_params(params, ACT, str(onnx_path))
    np_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(CKDIR / f"{tag}.pkl", "wb") as f:
        pickle.dump(np_params, f)
    _w(CK, [time.strftime("%H:%M:%S"), round(el, 1), int(current_step), NUM_ENVS, STATE["episodes"], str(onnx_path)],
       ["clock", "wall_s", "env_steps", "num_envs", "episodes_total", "onnx"])
    print(f"[{el/60:6.1f} min] CHECKPOINT {tag} episodes={STATE['episodes']:,}", flush=True)
    if RENDER_SCRIPT:
        out_mp4 = OUT / "sim" / f"{tag}.mp4"; out_mp4.parent.mkdir(exist_ok=True)
        logf = open(OUT / "sim" / f"{tag}.log", "w")
        subprocess.Popen(["nice", "-n", "15", RENDER_PY, "-I", RENDER_SCRIPT, REPO, str(onnx_path), str(out_mp4)],
                         stdout=logf, stderr=subprocess.STDOUT, cwd=str(OUT))

train_params = dict(ppo_params)
nf = train_params.pop("network_factory")
network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)
train_params.update(
    num_timesteps=args.num_timesteps, num_evals=args.num_evals, num_envs=NUM_ENVS,
    batch_size=NUM_ENVS // args.num_minibatches, num_minibatches=args.num_minibatches,
    run_evals=False, log_training_metrics=True, training_metrics_steps=1_000_000, seed=args.seed,
)
print("PPO params:", train_params, "devices:", jax.local_device_count(), flush=True)
with open(OUT / "run_config.json", "w") as f:
    json.dump({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in train_params.items()} |
              {"network_factory": str(nf), "devices": args.devices, "start": time.strftime("%Y-%m-%d %H:%M:%S")},
              f, indent=1)

# A separate eval_env instance is required: if brax re-wraps the training env for eval, domain
# randomization hits a leaked tracer (UnexpectedTracerError). The official runner does the same.
ppo.train(environment=env, eval_env=joystick.Joystick(task="flat_terrain"),
          wrap_env_fn=wrapper.wrap_for_brax_training, network_factory=network_factory,
          randomization_fn=randomize.domain_randomize, progress_fn=progress,
          policy_params_fn=policy_params_fn, **train_params)
print(f"DONE in {(time.time()-T0)/3600:.2f} h, episodes={STATE['episodes']:,}", flush=True)

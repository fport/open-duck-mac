"""Sim2real experiment: the OFFICIAL Open Duck Mini v2 joystick env, rewards and domain randomization
UNCHANGED, plus ONE change: the actuator torque limit (MJCF forcerange, ±3.23 N·m for every STS3215 joint).

  --force F         every joint limited to ±F N·m in every env (e.g. 1.86 = the 19 kg·cm catalog rating of the STS3215)
  --force_rand A B  per env and per joint, limit ~ U(A, B) N·m, drawn per env together with the official randomization
  (neither)         official ±3.23 (control run: same warm start, same number of steps)
  --restore PKL     warm start: policy + observation normalisation from a checkpoint, value net from scratch
  --restore_value   with --restore: also restore the value net (plain continuation of the same run)
  --step_offset N   with --restore: add N to logged steps and checkpoint names (steps already trained)

Copy of ../train_mac.py; differences are only the options above.
  python -I train_sim2real.py <Open_Duck_Playground> <out_dir> --force 1.86 --restore ckpt_068413440.pkl \
      --devices 5 --envs_per_dev 256 --num_timesteps 10000000 --num_evals 5
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
ap.add_argument("--force", type=float, default=0.0)
ap.add_argument("--force_rand", type=float, nargs=2, default=None)
ap.add_argument("--restore", default="")
ap.add_argument("--restore_value", action="store_true")
ap.add_argument("--step_offset", type=int, default=0)
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))  # onnx_export.py
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
import jax.numpy as jp

if args.force_rand: LO, HI = args.force_rand
elif args.force: LO = HI = args.force
else: LO = HI = None

def randomization_fn(model, rng):
    model, in_axes = randomize.domain_randomize(model, rng)
    if LO is None:
        return model, in_axes
    @jax.vmap
    def rand_force(key):
        lim = jax.random.uniform(jax.random.fold_in(key, 1862), (model.nu,), minval=LO, maxval=HI)
        return jp.stack([-lim, lim], axis=-1)
    model = model.tree_replace({"actuator_forcerange": rand_force(rng)})
    in_axes = in_axes.tree_replace({"actuator_forcerange": 0})
    return model, in_axes

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
    num_steps = int(num_steps) + args.step_offset
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
    tag = f"ckpt_{int(current_step) + args.step_offset:09d}"
    onnx_path = CKDIR / f"{tag}.onnx"
    onnx_export.from_brax_params(params, ACT, str(onnx_path))
    np_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(CKDIR / f"{tag}.pkl", "wb") as f:
        pickle.dump(np_params, f)
    _w(CK, [time.strftime("%H:%M:%S"), round(el, 1), int(current_step) + args.step_offset, NUM_ENVS, STATE["episodes"], str(onnx_path)],
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
              {"network_factory": str(nf), "devices": args.devices,
               "force_limit": ("official 3.23" if LO is None else [LO, HI]), "restore": args.restore or None, "restore_value": args.restore_value, "step_offset": args.step_offset, "start": time.strftime("%Y-%m-%d %H:%M:%S")},
              f, indent=1)

# A separate eval_env instance is required: if brax re-wraps the training env for eval, domain
# randomization hits a leaked tracer (UnexpectedTracerError). The official runner does the same.
ppo.train(environment=env, eval_env=joystick.Joystick(task="flat_terrain"),
          wrap_env_fn=wrapper.wrap_for_brax_training, network_factory=network_factory,
          randomization_fn=randomization_fn, progress_fn=progress,
          policy_params_fn=policy_params_fn,
          restore_params=(pickle.load(open(args.restore, "rb")) if args.restore else None),
          restore_value_fn=args.restore_value, **train_params)
print(f"DONE in {(time.time()-T0)/3600:.2f} h, episodes={STATE['episodes']:,}", flush=True)

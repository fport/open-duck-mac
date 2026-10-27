"""Reward hacking experiment training: a copy of ../train_mac.py; differences:
  * env hile_env.HileJoystick(<variant>) (reward weights / flip rule differ, physics+observation the same)
  * --restore <pkl>: CONTINUE from yesterday's checkpoint (warm start; policy + observation normalisation, value net from scratch)
  * all episode metrics (every reward term + diag/*) to train_metrics_ep.jsonl; run.json (weights, clock, steps, episodes,
    source file sha256s) updated at every checkpoint
  * at every checkpoint render_script (single rollout) + grid_script (3x3) in the background

  python -I train_hile.py <Playground_hile> <out_dir> --variant hizli --restore ckpt_068413440.pkl \
      --devices 5 --envs_per_dev 256 --num_timesteps 10000000 --num_evals 4 --render_python <venv python>
"""
import argparse, os, sys, time, csv, json, subprocess, pathlib, pickle, hashlib
ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("out_dir")
ap.add_argument("--variant", required=True)
ap.add_argument("--restore", default="")
ap.add_argument("--devices", type=int, default=5)
ap.add_argument("--envs_per_dev", type=int, default=256)
ap.add_argument("--num_timesteps", type=int, default=10_000_000)
ap.add_argument("--num_evals", type=int, default=4)
ap.add_argument("--num_minibatches", type=int, default=32)
ap.add_argument("--render_python", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_render", action="store_true")
args = ap.parse_args()

os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count={args.devices} "
                           "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
OUT = pathlib.Path(args.out_dir).resolve(); OUT.mkdir(parents=True, exist_ok=True)
for sub in ("params", "onnx", "sim"): (OUT / sub).mkdir(exist_ok=True)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))  # onnx_export.py lives in scripts/
os.chdir(args.repo); sys.path.insert(0, args.repo)

import functools
import numpy as np
import jax
from brax.training import logger as brax_logger
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from playground.common import randomize
import onnx_export, hile_env

T0 = time.time()
NUM_ENVS = args.devices * args.envs_per_dev
STATE = {"episodes": 0, "ckpts": []}
_orig_update = brax_logger.EpisodeMetricsLogger.update_episode_metrics
def _counting_update(self, metrics, dones):
    STATE["episodes"] += int(np.sum(np.asarray(dones)))
    return _orig_update(self, metrics, dones)
brax_logger.EpisodeMetricsLogger.update_episode_metrics = _counting_update

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
RUN = dict(variant=args.variant, note=hile_env.VARIANTS[args.variant]["note"], scales=hile_env.scales_for(args.variant),
           scales_yesterday=hile_env.BASE, terminate_on_flip=hile_env.VARIANTS[args.variant]["terminate"],
           command="sabit: ileri 0,15 m/s, yan 0, dönüş 0", restore=args.restore or None,
           restore_note="politika + gözlem normalizasyonu dünkü checkpoint'ten, değer ağı sıfırdan" if args.restore else "sıfırdan",
           num_envs=NUM_ENVS, devices=args.devices, num_timesteps_target=args.num_timesteps, seed=args.seed,
           start=time.strftime("%Y-%m-%d %H:%M:%S"), host="Mac (Apple Silicon) JAX CPU, nice 10",
           sources={p.name: sha(p) for p in [HERE / "hile_env.py", HERE / "train_hile.py",
                                              pathlib.Path(args.repo) / "playground/open_duck_mini_v2/joystick.py",
                                              pathlib.Path(args.repo) / "playground/common/rewards.py",
                                              pathlib.Path(args.repo) / "playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml"]},
           choreography="yok: hiçbir hareket/poz/yörünge yazılmadı; yalnız ödül ağırlıkları + devrilme kuralı değişti",
           progress=[], checkpoints=[])
def save_run():
    RUN["elapsed_min"] = round((time.time() - T0) / 60, 2); RUN["episodes_total"] = STATE["episodes"]
    json.dump(RUN, open(OUT / "run.json", "w"), indent=1, ensure_ascii=False)

def progress(num_steps, metrics):
    el = time.time() - T0
    ep = {k.replace("episode/", ""): round(float(v), 5) for k, v in metrics.items() if k.startswith("episode/")}
    row = {"clock": time.strftime("%H:%M:%S"), "wall_s": round(el, 1), "env_steps": int(num_steps),
           "episodes_total": STATE["episodes"], **ep}
    if ep:
        with open(OUT / "train_metrics_ep.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
        L = max(ep.get("length", 1), 1)
        brief = {"clock": row["clock"], "wall_min": round(el / 60, 2), "env_steps": int(num_steps), "episodes": STATE["episodes"],
                 "R": ep.get("sum_reward"), "len_s": round(L * 0.02, 2),
                 "dist_m_per_ep": round(ep.get("diag/fwd_vel", 0) * 0.02, 3),
                 "fallen_frac": round(ep.get("diag/fallen", 0) / L, 3), "feet_air_mean": round(ep.get("diag/feet_air", 0) / L, 3),
                 "trunk_z_mean": round(ep.get("diag/trunk_z", 0) / L, 4), "torque_sq_mean": round(ep.get("diag/torque_sq", 0) / L, 4)}
        RUN["progress"].append(brief); save_run()
        print(json.dumps(brief), flush=True)
    else:
        with open(OUT / "train_metrics.jsonl", "a") as f:
            f.write(json.dumps({"wall_s": round(el, 1), "env_steps": int(num_steps),
                                **{k: float(v) for k, v in metrics.items()}}) + "\n")

env = hile_env.HileJoystick(args.variant)
ACT = env.action_size
ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
RENDER = HERE / "render_ckpt_hile.py"; GRID = HERE / "render_grid_hile.py"

def policy_params_fn(current_step, make_policy, params):
    el = time.time() - T0
    tag = f"ckpt_{int(current_step):09d}"
    onnx_path = OUT / "onnx" / f"{tag}.onnx"
    onnx_export.from_brax_params(params, ACT, str(onnx_path))
    np_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(OUT / "params" / f"{tag}.pkl", "wb") as f: pickle.dump(np_params, f)
    RUN["checkpoints"].append({"tag": tag, "clock": time.strftime("%H:%M:%S"), "wall_min": round(el / 60, 2),
                               "env_steps_this_run": int(current_step), "episodes_total": STATE["episodes"]})
    save_run()
    print(f"[{el/60:6.1f} min] CHECKPOINT {tag} episodes={STATE['episodes']:,}", flush=True)
    if args.render_python and not args.no_render:
        logf = open(OUT / "sim" / f"{tag}.log", "w")
        cmd = (f"nice -n 15 {args.render_python} -I {RENDER} {args.repo} {onnx_path} {OUT/'sim'/(tag+'.mp4')} 5 && "
               f"nice -n 15 {args.render_python} -I {GRID} {args.repo} {onnx_path} {OUT/'sim'/(tag+'_grid.mp4')} 5")
        subprocess.Popen(["bash", "-c", cmd], stdout=logf, stderr=subprocess.STDOUT, cwd=str(OUT / "sim"))

train_params = dict(ppo_params)
nf = train_params.pop("network_factory")
network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)
train_params.update(
    num_timesteps=args.num_timesteps, num_evals=args.num_evals, num_envs=NUM_ENVS,
    batch_size=NUM_ENVS // args.num_minibatches, num_minibatches=args.num_minibatches,
    run_evals=False, log_training_metrics=True, training_metrics_steps=500_000, seed=args.seed,
)
restore = None
if args.restore:
    restore = pickle.load(open(args.restore, "rb"))
RUN["ppo"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in train_params.items()}
save_run()
print("variant", args.variant, RUN["scales"], "terminate", RUN["terminate_on_flip"], "devices", jax.local_device_count(), flush=True)
ppo.train(environment=env, eval_env=hile_env.HileJoystick(args.variant), wrap_env_fn=wrapper.wrap_for_brax_training,
          network_factory=network_factory, randomization_fn=randomize.domain_randomize, progress_fn=progress,
          policy_params_fn=policy_params_fn, restore_params=restore, restore_value_fn=False, **train_params)
RUN["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_run()
print(f"DONE in {(time.time()-T0)/3600:.2f} h, episodes={STATE['episodes']:,}", flush=True)

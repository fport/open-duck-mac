"""Smoke test 2: pmap over D virtual CPU devices (one core each), N envs per device.

This is the layout train_mac.py uses. It sets XLA_FLAGS itself (before importing jax) so JAX sees D
host devices, then times raw env steps with zero actions. Compare against smoke.py to pick D and N.
On our 10-core M-series Mac: 1 device x 1024 envs = 2,368 steps/s; 6 devices x 256 = 4,621 steps/s.

  python -I scripts/smoke_pmap.py ../Open_Duck_Playground [--envs_per_dev 256] [--devices 6]
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", help="path to an Open_Duck_Playground checkout")
ap.add_argument("--envs_per_dev", type=int, default=256)
ap.add_argument("--devices", type=int, default=6)
ap.add_argument("--steps", type=int, default=50, help="timed steps after compilation (default 50)")
a = ap.parse_args()

os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count={a.devices} "
                           "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
repo = os.path.abspath(a.repo)
os.chdir(repo); sys.path.insert(0, repo)
import jax, jax.numpy as jp
D, n = a.devices, a.envs_per_dev
print("devices", len(jax.devices()))
from playground.open_duck_mini_v2 import joystick

env = joystick.Joystick(task="flat_terrain")
reset = jax.pmap(jax.vmap(env.reset)); step = jax.pmap(jax.vmap(env.step))
keys = jax.random.split(jax.random.PRNGKey(0), D * n).reshape(D, n, 2)
t = time.time(); s = reset(keys); jax.block_until_ready(s.obs); print(f"reset compile {time.time() - t:.1f} s")
act = jp.zeros((D, n, env.action_size))
t = time.time(); s = step(s, act); jax.block_until_ready(s.obs); print(f"step compile {time.time() - t:.1f} s")
t = time.time()
for _ in range(a.steps): s = step(s, act)
jax.block_until_ready(s.obs); dt = time.time() - t
print(f"D={D} n={n} total={D * n} steps/s={D * n * a.steps / dt:.0f}")

"""Smoke test 1: does the Open Duck Mini v2 env build and step on JAX CPU, and how fast?

Builds the official joystick env (flat_terrain), jits vmap(reset) and vmap(step) over N envs with
zero actions, and prints compile times and raw env steps/s (no learning, no PPO update).

  python -I scripts/smoke.py ../Open_Duck_Playground [--envs 64]
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", help="path to an Open_Duck_Playground checkout")
ap.add_argument("--envs", type=int, default=64, help="number of parallel envs (default 64)")
ap.add_argument("--steps", type=int, default=50, help="timed steps after compilation (default 50)")
a = ap.parse_args()

repo = os.path.abspath(a.repo)
os.chdir(repo); sys.path.insert(0, repo)
import jax, jax.numpy as jp
print("devices", jax.devices())
from playground.open_duck_mini_v2 import joystick

env = joystick.Joystick(task="flat_terrain")
print("obs", env.observation_size, "act", env.action_size)
n = a.envs
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
keys = jax.random.split(jax.random.PRNGKey(0), n)
t = time.time(); s = reset(keys); jax.block_until_ready(s.obs); print(f"reset compile {time.time() - t:.1f} s")
act = jp.zeros((n, env.action_size))
t = time.time(); s = step(s, act); jax.block_until_ready(s.obs); print(f"step compile {time.time() - t:.1f} s")
t = time.time()
for _ in range(a.steps): s = step(s, act)
jax.block_until_ready(s.obs); dt = time.time() - t
print(f"n={n} steps/s={n * a.steps / dt:.0f}")

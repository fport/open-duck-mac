"""Evaluate trained policies, without retraining, in several modified worlds (friction, mass, torque limit).
  python3 scripts/sim2real/sweep.py <out_dir> <name>=<onnx> [...]   (deterministic 5 s + 9 stochastic seeds in 4 worlds)
  env: PY (default .venv/bin/python), PLAYGROUND (default external/Open_Duck_Playground), NOISY=0 to skip seeds, PAR=3"""
import sys, os, json, subprocess, pathlib, itertools, concurrent.futures as cf
ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", str(ROOT / ".venv/bin/python"))
REPO = os.environ.get("PLAYGROUND", str(ROOT / "external/Open_Duck_Playground"))
HERE = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
pols = dict(a.split("=", 1) for a in sys.argv[2:])
WORLDS = {"base": {}, "fric050": {"FLOOR_FRIC": "0.5"}, "mass110": {"MASS_SCALE": "1.1"}, "mass120": {"MASS_SCALE": "1.2"},
          "f250": {"FORCE": "2.5"}, "f186": {"FORCE": "1.86"}, "f160": {"FORCE": "1.6"}, "f140": {"FORCE": "1.4"}}
NOISY = {"base", "fric050", "mass110", "f186"}
jobs = []
for (pn, onnx), wn in itertools.product(pols.items(), WORLDS):
    jobs.append((pn, onnx, wn, None))
    if os.environ.get("NOISY", "1") == "1" and wn in NOISY and "best" not in pn:
        jobs += [(pn, onnx, wn, s) for s in range(9)]
def run(j):
    pn, onnx, wn, seed = j
    tag = f"{pn}__{wn}" + ("" if seed is None else f"__s{seed}")
    cmd = ["nice", "-n", "15", PY, "-I", str(HERE / "eval_world.py"), REPO, onnx, str(out / f"{tag}.mp4"), "--no_video"]
    if seed is not None: cmd += ["--stochastic", "--seed", str(seed)]
    r = subprocess.run(cmd, env=dict(os.environ, **WORLDS[wn]), capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return tag, (json.loads(line[-1]) if line else {"error": r.stderr[-300:]})
res = {}
with cf.ThreadPoolExecutor(int(os.environ.get("PAR", "3"))) as ex:
    for tag, d in ex.map(run, jobs): res[tag] = d
json.dump(res, open(out / "sweep.json", "w"), indent=1)
for (pn, _), wn in itertools.product(pols.items(), WORLDS):
    d = res[f"{pn}__{wn}"]; line = f"{pn:10s} {wn:8s} det dx={d.get('dx_m',float('nan'))*100:6.1f} cm fell={d.get('fell_t')} steps={d.get('foot_steps')}"
    ns = [res[k] for k in res if k.startswith(f"{pn}__{wn}__s")]
    if ns: line += f" | 9 noisy: fell {sum(n.get('fell_t') is not None for n in ns)}/9, mean dx {sum(n['dx_m'] for n in ns)/9*100:.1f} cm"
    print(line, flush=True)

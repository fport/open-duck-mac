"""Evaluate every new checkpoint of a running training at the STS3215 catalog torque (1.86 N·m, deterministic 5 s)
and log whether it walks, until the training log says DONE. A checkpoint walks when it covers >= 10 cm and both feet
oscillate (gait.py: >= 4 cycles, >= 5 mm lift). The first version counted foot_steps instead and missed every
walking checkpoint of part 3 (it counts the walking 72.4 M policy as 0+2 steps).
  python3 scripts/sim2real/watch_ckpts.py <ckpt_dir> <train_log> <out_csv> [--exit_on_walk] [--once] [--force 1.86|sim]
--force sim tests with the model's own limit (3.23 N·m); --once makes one pass over the checkpoints and exits."""
import sys, os, json, time, subprocess, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", str(ROOT / ".venv/bin/python"))
REPO = os.environ.get("PLAYGROUND", str(ROOT / "external/Open_Duck_Playground"))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); from gait import gait, walks
ck, log, out = map(pathlib.Path, sys.argv[1:4])
EXIT_ON_WALK, ONCE = "--exit_on_walk" in sys.argv, "--once" in sys.argv
FORCE = sys.argv[sys.argv.index("--force") + 1] if "--force" in sys.argv else "1.86"
seen = set()
(out.parent / "watch").mkdir(parents=True, exist_ok=True)
if out.exists(): seen = {r["tag"] for r in csv.DictReader(open(out))}
def ev(onnx):
    mp4 = out.parent / "watch" / f"{onnx.stem}.mp4"
    r = subprocess.run(["nice", "-n", "15", PY, "-I", str(HERE / "eval_world.py"), REPO, str(onnx), str(mp4), "--no_video"],
                       env=dict(os.environ, **({} if FORCE == "sim" else {"FORCE": FORCE})), capture_output=True, text=True)
    js = mp4.with_suffix(".json")
    return json.load(open(js)) if r.returncode == 0 and js.exists() else None
while True:
    done = log.exists() and "DONE in" in log.read_text()
    for onnx in sorted(ck.glob("ckpt_*.onnx")):
        if onnx.stem in seen: continue
        time.sleep(5)  # let the exporter finish writing
        d = ev(onnx); seen.add(onnx.stem)
        g = gait(d) if d else None
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["tag", "clock", "dx_cm", "cycles_l", "cycles_r", "lift_l_mm", "lift_r_mm", "lr_corr", "walks", "fell_t"])
            w.writerow([onnx.stem, time.strftime("%H:%M:%S")] + ([g["dx_cm"], g["feet"][0]["cycles"], g["feet"][1]["cycles"],
                       g["feet"][0]["lift_mm"], g["feet"][1]["lift_mm"], g["lr_corr"], int(walks(g)), d.get("fell_t")] if g else ["error"] + [""] * 7))
        print(onnx.stem, json.dumps(g), flush=True)
        if g and walks(g):
            print(f"WALKS {onnx.stem}", flush=True)
            if EXIT_ON_WALK: sys.exit(0)
    if done: print("TRAINING_DONE", flush=True); sys.exit(0)
    if ONCE: sys.exit(0)
    time.sleep(120)

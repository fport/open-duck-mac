"""Evaluate every new checkpoint of a running training at the STS3215 catalog torque (1.86 N·m, deterministic 5 s)
and exit as soon as one walks, or when the training log says DONE.
  python3 scripts/sim2real/watch_ckpts.py <ckpt_dir> <train_log> <out_csv>"""
import sys, os, json, time, subprocess, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", str(ROOT / ".venv/bin/python"))
REPO = os.environ.get("PLAYGROUND", str(ROOT / "external/Open_Duck_Playground"))
HERE = pathlib.Path(__file__).resolve().parent
ck, log, out = map(pathlib.Path, sys.argv[1:4])
WALK_DX, WALK_STEPS = 0.10, 4
seen = set()
(out.parent / "watch").mkdir(parents=True, exist_ok=True)
if out.exists(): seen = {r["tag"] for r in csv.DictReader(open(out))}
def ev(onnx):
    r = subprocess.run(["nice", "-n", "15", PY, "-I", str(HERE / "eval_world.py"), REPO, str(onnx), str(out.parent / "watch" / f"{onnx.stem}.mp4"), "--no_video"],
                       env=dict(os.environ, FORCE="1.86"), capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else {"error": r.stderr[-200:]}
while True:
    done = log.exists() and "DONE in" in log.read_text()
    for onnx in sorted(ck.glob("ckpt_*.onnx")):
        if onnx.stem in seen: continue
        time.sleep(5)  # let the exporter finish writing
        d = ev(onnx); seen.add(onnx.stem)
        steps = sum(d.get("foot_steps") or [0])
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["tag", "clock", "dx_cm", "foot_steps", "fell_t"])
            w.writerow([onnx.stem, time.strftime("%H:%M:%S"), round(d.get("dx_m", float("nan")) * 100, 1), steps, d.get("fell_t")])
        print(onnx.stem, d.get("dx_m"), steps, flush=True)
        if d.get("dx_m", 0) >= WALK_DX and steps >= WALK_STEPS:
            print(f"WALKS {onnx.stem}", flush=True); sys.exit(0)
    if done: print("TRAINING_DONE_NO_WALK", flush=True); sys.exit(0)
    time.sleep(120)

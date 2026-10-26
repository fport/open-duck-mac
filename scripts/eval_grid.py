"""'As in training' test: one checkpoint, 9 exploration-noise seeds (eval_checkpoint.py --stochastic).

Runs 9 rollouts (3 at a time) and writes <out>.json: how many of the 9 ducks fell and their mean forward
distance. With ffmpeg on PATH (and without --no_video) it also stacks the 9 videos into a 3x3 grid mp4.
The deterministic policy of an untrained network barely moves and does not fall; sampling with the
exploration noise shows what the duck actually experiences during training.

  python -I scripts/eval_grid.py ../Open_Duck_Playground runs/run1/ckpt/ckpt_000000000.onnx \\
      runs/run1/sim/ckpt_000000000_grid.mp4 [--secs 5] [--tile 360 640] [--no_video]
"""
import argparse, json, os, pathlib, shutil, subprocess, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", help="path to an Open_Duck_Playground checkout")
ap.add_argument("onnx", help="policy .onnx written by onnx_export.py (needs the loc_out/scale outputs)")
ap.add_argument("out", help="output path ending in .mp4; <out>.json and <out minus .mp4>_tiles/ are written")
ap.add_argument("--secs", type=float, default=5.0)
ap.add_argument("--vx", type=float, default=0.15)
ap.add_argument("--tile", type=int, nargs=2, default=[360, 640], metavar=("W", "H"), help="tile size (default 360 640)")
ap.add_argument("--parallel", type=int, default=3, help="rollouts at a time (default 3)")
ap.add_argument("--python", default=sys.executable, help="interpreter for eval_checkpoint.py")
ap.add_argument("--no_video", action="store_true", help="metrics only")
a = ap.parse_args()

if not a.out.endswith(".mp4"):
    sys.exit("out must end in .mp4")
here = pathlib.Path(__file__).resolve().parent
tiles = pathlib.Path(a.out[:-4] + "_tiles"); tiles.mkdir(parents=True, exist_ok=True)
TW, TH = a.tile
procs, outs = [], []
for sd in range(9):
    o = tiles / f"s{sd}.mp4"; outs.append(o)
    cmd = ["nice", "-n", "15", a.python, "-I", str(here / "eval_checkpoint.py"), a.repo, a.onnx, str(o),
           "--secs", str(a.secs), "--vx", str(a.vx), "--width", str(TW), "--height", str(TH),
           "--stochastic", "--seed", str(sd), "--azimuth", "18", "30"] + (["--no_video"] if a.no_video else [])
    procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    while len([p for p in procs if p.poll() is None]) >= a.parallel:
        next(p for p in procs if p.poll() is None).wait()
for p in procs: p.wait()
bad = [i for i, p in enumerate(procs) if p.returncode != 0]
if bad:
    sys.exit(f"rollouts failed for seeds {bad}; run eval_checkpoint.py by hand with --stochastic --seed N to see why")

if not a.no_video:
    if shutil.which("ffmpeg"):
        inputs = sum([["-i", str(o)] for o in outs], [])
        layout = "|".join(f"{(i % 3) * TW}_{(i // 3) * TH}" for i in range(9))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", f"xstack=inputs=9:layout={layout}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", a.out], check=True)
    else:
        print("ffmpeg not found: skipping the 3x3 mosaic (tiles are in", tiles, ")")
rows = []
for o in outs:
    with open(str(o).replace(".mp4", ".json")) as f:
        d = json.load(f)
    rows.append({k: d[k] for k in ("seed", "dist_m", "dx_m", "fell_t", "final_trunk_z")})
summ = {"onnx": os.path.basename(a.onnx), "secs": a.secs, "vx": a.vx, "n": 9,
        "fell": sum(r["fell_t"] is not None for r in rows),
        "mean_dx_m": round(sum(r["dx_m"] for r in rows) / 9, 4), "seeds": rows}
with open(a.out.replace(".mp4", ".json"), "w") as f:
    json.dump(summ, f, indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "seeds"}))

"""[reward hacking version: copy of ../eval_grid.py restyled, uses render_ckpt_hile.py, default tile 360x640 -> 1080x1920, extra measurements]
[v6 bright 4K version: copy of render_grid.py; uses render_ckpt_hile.py, tile size as argument, tiles are kept]
v2 (24.10): camera GRID_AZ0/GRID_AZ1, floor reflectance FL_R via env (render_ckpt_hile.py).
3x3 'as in training' grid: same checkpoint, 9 different exploration-noise seeds (STOCH=1), 360x640 tiles
-> 1080x1920 mp4 + summary json (how many ducks fell, mean distance).

  python -I render_grid.py <repo> <onnx> <out_grid.mp4> [secs=5] [python=sys.executable]
"""
import sys, os, json, subprocess, pathlib, tempfile
repo, onnx, out = sys.argv[1], sys.argv[2], sys.argv[3]
secs = sys.argv[4] if len(sys.argv) > 4 else "5"
py = sys.argv[5] if len(sys.argv) > 5 else sys.executable
TW = int(sys.argv[6]) if len(sys.argv) > 6 else 360
TH = int(sys.argv[7]) if len(sys.argv) > 7 else 640
here = pathlib.Path(__file__).resolve().parent
tmp = pathlib.Path(out.replace(".mp4", "_tiles")); tmp.mkdir(parents=True, exist_ok=True)
procs, outs = [], []
for sd in range(9):
    o = tmp / f"s{sd}.mp4"; outs.append(o)
    env = dict(os.environ, STOCH="1", SEED=str(sd), AZ0=os.environ.get("GRID_AZ0", "18"), AZ1=os.environ.get("GRID_AZ1", "30"))
    procs.append(subprocess.Popen(["nice", "-n", "15", py, "-I", str(here / "render_ckpt_hile.py"), repo, onnx, str(o), secs,
                                   "0.15", str(TW), str(TH)], env=env, cwd=str(tmp), stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL))
    if len([p for p in procs if p.poll() is None]) >= 3:
        procs[-3].wait()
for p in procs: p.wait()
inputs = sum([["-i", str(o)] for o in outs], [])
layout = "|".join(f"{(i%3)*TW}_{(i//3)*TH}" for i in range(9))
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", f"xstack=inputs=9:layout={layout}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "slow", out], check=True)
rows = []
for o in outs:
    d = json.load(open(str(o).replace(".mp4", ".json"))); d.pop("frames")
    rows.append({k: d[k] for k in ("seed", "dist_m", "dx_m", "fell_t", "flip_t", "flipped_s", "final_trunk_z", "foot_steps",
                                    "foot_lift_max_m", "both_feet_air_frac", "torque_sq_mean", "joint_travel_rad", "max_trunk_z")})
summ = {"onnx": onnx, "secs": float(secs), "n": 9, "fell": sum(r["fell_t"] is not None for r in rows),
        "mean_dx_m": round(sum(r["dx_m"] for r in rows) / 9, 4),
        "flipped_90": sum(r["flip_t"] is not None for r in rows), "max_dx_m": max(r["dx_m"] for r in rows),
        "mean_foot_steps": round(sum(sum(r["foot_steps"]) for r in rows) / 9, 2),
        "mean_both_feet_air_frac": round(sum(r["both_feet_air_frac"] for r in rows) / 9, 4),
        "mean_torque_sq": round(sum(r["torque_sq_mean"] for r in rows) / 9, 3), "seeds": rows}
json.dump(summ, open(out.replace(".mp4", ".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "seeds"}))

"""Turn a run directory into one table + a summary + a reward curve.

Inputs in <run_dir> (written by train_mac.py and eval_checkpoint.py / eval_grid.py):
  train_log_raw.csv, checkpoints.csv, sim/ckpt_<steps>.json, sim/ckpt_<steps>_grid.json (the sim/ files are optional)
Outputs:
  train_log.csv     clock, wall_min, env_steps, num_envs, episodes_total, reward_mean (mean episode return),
                    ep_len_steps, ep_len_s (x 0.02 s), kind (train|ckpt), ckpt_dx_m / ckpt_dist_m / ckpt_fell_t
                    (deterministic 5 s rollout), grid_fell_of_9, grid_mean_dx_m (9 noisy rollouts)
  ckpt_summary.json one entry per checkpoint
  reward_curve.png  episode return, episode length and the imitation reward term vs env steps (needs matplotlib)

With pmap, every CPU "device" may call the progress callback, so train_log_raw.csv can hold duplicate rows
for the same step and, from concurrent writes, a few corrupt ones. Corrupt rows are dropped and the last
valid row per env_steps is kept.

  python scripts/make_log.py runs/run1
  python scripts/make_log.py results/2026-10-24 --plot_only    # just redraw reward_curve.png
"""
import argparse, csv, json, pathlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir")
ap.add_argument("--plot_only", action="store_true", help="only redraw the PNG from existing CSVs")
ap.add_argument("--png", default="", help="output PNG path (default <run_dir>/reward_curve.png)")
a = ap.parse_args()
OUT = pathlib.Path(a.run_dir)


def _ok(r, extra=()):
    try:
        float(r["ep_sum_reward"]); float(r["ep_length"]); int(r["env_steps"]); int(r["episodes_total"]); float(r["wall_s"])
        for k in extra: float(r[k])
        return len(r["clock"]) == 8 and None not in r
    except Exception:
        return False


def load_raw():
    p = OUT / "train_log_raw.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        raw = list(csv.DictReader(f))
    return {int(r["env_steps"]): r for r in raw if _ok(r)}


def build():
    raw = list(load_raw().values())
    with open(OUT / "checkpoints.csv") as f:
        ck = list(csv.DictReader(f))
    rows = []
    for r in raw:
        rows.append(dict(clock=r["clock"], wall_min=round(float(r["wall_s"]) / 60, 2), env_steps=int(r["env_steps"]),
                         num_envs=r["num_envs"], episodes_total=int(r["episodes_total"]),
                         reward_mean=round(float(r["ep_sum_reward"]), 3), ep_len_steps=round(float(r["ep_length"]), 1),
                         ep_len_s=round(float(r["ep_length"]) * 0.02, 2), kind="train"))
    summ = []
    for c in ck:
        tag = pathlib.Path(c["onnx"]).stem
        d = OUT / "sim" / f"{tag}.json"; g = OUT / "sim" / f"{tag}_grid.json"
        dj = json.load(open(d)) if d.exists() else {}
        gj = json.load(open(g)) if g.exists() else {}
        row = dict(clock=c["clock"], wall_min=round(float(c["wall_s"]) / 60, 2), env_steps=int(c["env_steps"]),
                   num_envs=c["num_envs"], episodes_total=int(c["episodes_total"]), kind="ckpt",
                   ckpt_dx_m=dj.get("dx_m"), ckpt_dist_m=dj.get("dist_m"), ckpt_fell_t=dj.get("fell_t"),
                   grid_fell_of_9=gj.get("fell"), grid_mean_dx_m=gj.get("mean_dx_m"))
        prev = [r for r in rows if r["kind"] == "train" and r["env_steps"] <= row["env_steps"]]
        if prev:
            row.update(reward_mean=prev[-1]["reward_mean"], ep_len_steps=prev[-1]["ep_len_steps"], ep_len_s=prev[-1]["ep_len_s"])
        rows.append(row); summ.append(dict(tag=tag, **{k: v for k, v in row.items() if k != "kind"}))
    rows.sort(key=lambda r: (r["env_steps"], r["kind"] == "ckpt"))
    cols = ["clock", "wall_min", "env_steps", "num_envs", "episodes_total", "reward_mean", "ep_len_steps", "ep_len_s", "kind",
            "ckpt_dx_m", "ckpt_dist_m", "ckpt_fell_t", "grid_fell_of_9", "grid_mean_dx_m"]
    with open(OUT / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    with open(OUT / "ckpt_summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(len(rows), "rows;", len(summ), "ckpts")


def plot():
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no PNG"); return
    with open(OUT / "train_log.csv") as f:
        tl = list(csv.DictReader(f))
    tr = [r for r in tl if r["kind"] == "train"]
    ck = [r for r in tl if r["kind"] == "ckpt"]
    x = [int(r["env_steps"]) / 1e6 for r in tr]
    raw = load_raw()
    im = [(int(s) / 1e6, float(r["ep_reward_imitation"])) for s, r in sorted(raw.items()) if _ok(r, ("ep_reward_imitation",))]

    def ma(v, k=5):  # centred moving average, shorter at the ends
        h = k // 2
        return [sum(v[max(0, i - h):i + h + 1]) / len(v[max(0, i - h):i + h + 1]) for i in range(len(v))]

    n = 3 if im else 2
    fig, ax = plt.subplots(n, 1, figsize=(7, 2.6 * n + 0.6), sharex=True)
    series = [([float(r["reward_mean"]) for r in tr], "episode return (mean)"),
              ([float(r["ep_len_s"]) for r in tr], "episode length (s, max 20)")]
    if im:
        series.append(([v for _, v in im], "imitation reward term"))
    for i, (v, label) in enumerate(series):
        xs = x if i < 2 else [s for s, _ in im]
        ax[i].plot(xs, v, color="#9aa4b2", lw=1, label="raw (~1M steps)")
        ax[i].plot(xs, ma(v), color="#e8742c", lw=2, label="5-point moving average")
        ax[i].set_ylabel(label); ax[i].grid(alpha=.3)
    # first checkpoint where the deterministic rollout went > 10 cm forward
    walk = [r for r in ck if r.get("ckpt_dx_m") and float(r["ckpt_dx_m"]) > 0.10]
    if walk:
        s = int(walk[0]["env_steps"]) / 1e6
        for axis in ax: axis.axvline(s, color="#3b6fd8", ls="--", lw=1)
        ax[0].text(s, ax[0].get_ylim()[1], f"first lifted feet: {s:.1f}M ", va="top", ha="right", color="#3b6fd8", fontsize=9)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[-1].set_xlabel("env steps (millions)")
    fig.suptitle("Open Duck Mini v2, PPO from scratch on a Mac CPU (single run, seed 0)", fontsize=10)
    fig.tight_layout()
    png = pathlib.Path(a.png) if a.png else OUT / "reward_curve.png"
    fig.savefig(png, dpi=130)
    print("wrote", png)


if not a.plot_only:
    build()
plot()

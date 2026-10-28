"""Gait signature from an eval_world.py JSON: is the duck stepping or just standing?
The foot_steps counter in eval_world.py needs a lift of >4 mm above the lowest point *and* back; the Open Duck
gait lifts its feet only ~1 cm and the counter misses it (it counts the walking 72.4 M policy as 0+2 steps).
Here: per foot, oscillation cycles of the foot height around its mean after 0.5 s, the peak-to-peak lift,
and the left/right correlation (a walking gait alternates, so it is negative).
  python3 gait.py <eval.json> [<eval.json> ...]"""
import json, sys

def gait(d, t_from=0.5):
    fr = [f for f in d["frames"] if f["t"] >= t_from]
    feet = []
    for i in range(2):
        z = [f["fz"][i] for f in fr]; m = sum(z) / len(z)
        crossings = sum(1 for a, b in zip(z, z[1:]) if (a - m) * (b - m) < 0)
        feet.append({"cycles": crossings // 2, "lift_mm": round((max(z) - min(z)) * 1000, 1)})
    zl, zr = ([f["fz"][i] for f in fr] for i in range(2))
    ml, mr = sum(zl) / len(zl), sum(zr) / len(zr)
    num = sum((a - ml) * (b - mr) for a, b in zip(zl, zr))
    den = (sum((a - ml) ** 2 for a in zl) * sum((b - mr) ** 2 for b in zr)) ** 0.5
    return {"dx_cm": round(d["dx_m"] * 100, 1), "feet": feet, "lr_corr": round(num / den, 2) if den else 0.0}

def walks(g, min_dx_cm=10, min_cycles=4, min_lift_mm=5):
    return g["dx_cm"] >= min_dx_cm and all(f["cycles"] >= min_cycles and f["lift_mm"] >= min_lift_mm for f in g["feet"])

if __name__ == "__main__":
    for p in sys.argv[1:]:
        g = gait(json.load(open(p))); print(p, json.dumps(g), "WALKS" if walks(g) else "-")

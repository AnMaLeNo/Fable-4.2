import glob, sys, json, numpy as np, torch
from goldsim import *
orig = torch.load("outputs/orig.pt"); subs = [torch.load(f) for f in sorted(glob.glob("outputs/*.pt")) if not f.endswith("orig.pt")]
subs = [s for s in subs if "score" in s]; names = [s["file"] for s in subs]; target = np.array([s["score"] for s in subs])
print(len(subs), "scored submissions")
def simulate(p): 
    golds = [gold(p, d) for d in range(D)]; return np.array([score_sub(p, s, orig, golds) for s in subs])
def loss(p): return float(((simulate(p) - target) ** 2).mean())
def clip(p): return {k: float(min(max(v, LO[k]), HI[k])) for k, v in p.items()}
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng0 = np.random.default_rng(seed)
p = dict(P0) if seed == 0 else {k: float(LO[k] + rng0.random() * (HI[k] - LO[k])) for k in PNAMES}
best = loss(p); print("initial rmse %.4f" % best ** 0.5, flush=True)
rng = np.random.default_rng(seed); step = {k: 0.25 * (HI[k] - LO[k]) for k in PNAMES}
for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    for k in rng.permutation(PNAMES):
        improved = False
        for sgn in (+1, -1):
            q = clip({**p, k: p[k] + sgn * step[k]}); l = loss(q)
            if l < best - 1e-7: p, best, improved = q, l, True; break
        step[k] *= 1.3 if improved else 0.5
    print(f"iter {it}: rmse {best ** 0.5:.4f}  " + " ".join(f"{k}={p[k]:.3g}" for k in PNAMES), flush=True)
    json.dump({**p, "rmse": best ** 0.5}, open(f"gold_fit_{seed}.json", "w"))
sim = simulate(p)
for n, t, s in sorted(zip(names, target, sim), key=lambda r: r[1]): print(f"  {n:34s} real {t:.4f}  model {s:.4f}")

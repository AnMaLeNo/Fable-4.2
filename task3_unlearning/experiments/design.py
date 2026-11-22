"""Choose target parameters that maximise the expected score under the fitted gold model (gold_fit.json).
Assumes the trained model reproduces its targets (clean and altered); retain factor taken as constant."""
import json, sys, numpy as np, torch
from goldsim import *
import glob
FITS = [json.load(open(f)) for f in sorted(glob.glob("gold_wfit_*.json"))]
FITS = [f for f in FITS if f["rmse"] <= min(x["rmse"] for x in FITS) * 1.25]
print("using", len(FITS), "fitted gold models, rmse", [round(f["rmse"], 4) for f in FITS])
GOLDS = [[gold(f, d) for d in range(D)] for f in FITS]
orig = torch.load("outputs/orig.pt")
Lshape = torch.load("teacher_xheavy.pt")["L"].flatten().sort().values
def build(d):
    h = d["wP"] * SIG["P"] + d["wI"] * SIG["I"] + d["wS"] * SIG["S"]; qs = rkn(h)
    w0 = 0.45; cut = 1 - d["omega"]
    qs = torch.where(qs < cut, qs * (1 - w0) / cut, (1 - w0) + (qs - cut) * w0 / d["omega"]).clamp(0, 1)
    loss = (Lshape[(qs * (len(Lshape) - 1)).long()] * d["scale"]).clamp(1e-4, d["maxloss"])
    R = d["rP"] * WR["P"] + d["rI"] * WR["I"] + d["rS"] * WR["S"] + d["rR"] * WR["R"] + d["rrows"] * WR["rows"] + 1e-7
    R = R ** d["beta"]; R.scatter_(1, y[:, None], 0.0); R = R / R.sum(1, keepdim=True)
    py = torch.exp(-loss); P1 = R * (1 - py)[:, None]; P1.scatter_(1, y[:, None], py[:, None]); return P1, loss
def expected(d):
    P1, loss = build(d); lg = P1.clamp_min(1e-12).log(); sub = {"clean": lg, "mild": lg[orig["mild_idx"]], "ext": orig["ext"]}
    return float(np.mean([score_sub({**f, "kappa": 0.0}, sub, orig, g) for f, g in zip(FITS, GOLDS)]))
D0 = dict(wP=1.0, wI=1.0, wS=2.0, omega=0.45, scale=2.0, maxloss=20.0, rP=1.0, rI=1.0, rS=2.0, rR=0.0, rrows=0.0, beta=2.0)
LOd = dict(wP=0, wI=0, wS=0, omega=0.3, scale=1.5, maxloss=14, rP=0, rI=0, rS=0, rR=0, rrows=0, beta=1.0)
HId = dict(wP=6, wI=6, wS=6, omega=0.6, scale=2.8, maxloss=24, rP=4, rI=4, rS=4, rR=4, rrows=4, beta=3)
if __name__ == "__main__":
    print("current design (pis2_112): expected forget score %.4f" % expected(D0))
    for name, dd in [("Rmix", {**D0, "rP": 1, "rI": 1, "rS": 1, "rR": 1, "rrows": 3}), ("S-only R", {**D0, "rP": 0, "rI": 0}), ("level 1.5", {**D0, "scale": 1.5}), ("level 2.5", {**D0, "scale": 2.5}),
                     ("omega .35", {**D0, "omega": 0.35}), ("omega .55", {**D0, "omega": 0.55}), ("maxloss 14", {**D0, "maxloss": 14}), ("beta 3", {**D0, "beta": 3})]:
        print(f"  {name:12s} {expected(dd):.4f}")
    d = dict(D0); best = expected(d); rng = np.random.default_rng(0); step = {k: 0.2 * (HId[k] - LOd[k]) for k in d}
    for it in range(8):
        for k in rng.permutation(list(d)):
            ok = False
            for sgn in (1, -1):
                q = {**d, k: float(min(max(d[k] + sgn * step[k], LOd[k]), HId[k]))}; e = expected(q)
                if e > best + 1e-6: d, best, ok = q, e, True; break
            step[k] *= 1.3 if ok else 0.5
        print(f"iter {it}: expected {best:.4f}  " + " ".join(f"{k}={v:.3g}" for k, v in d.items()), flush=True)
    json.dump(d, open("design_best.json", "w"))

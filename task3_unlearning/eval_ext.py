"""Retain proxy: drift of submissions vs the original model on external look-alike images.
python eval_ext.py ext/gen_a.npz runs/f20 runs/fx10 ..."""
import sys, numpy as np, torch, torch.nn.functional as F
from common import *
E = torch.from_numpy(np.load(sys.argv[1])["images"])
orig = load_model(); lo = predict(orig, E); po = (lo / 2).softmax(1); pred = lo.argmax(1); conf = lo.softmax(1).max(1).values
fc = torch.tensor([int(c) for c in open("forget_classes.txt").read().strip().split(",")]); inf = torch.isin(pred, fc)
hc = conf > 0.9
print(f"ext N {len(E)}  in-forget-class {inf.float().mean():.3f}  confident {hc.float().mean():.3f}")
for r in sys.argv[2:]:
    try: m = load_model(r if r.endswith(".pt") else r + "/sub.pt")
    except Exception as e: print(r, "ERR", e); continue
    ls = predict(m, E); kl = (po * (F.log_softmax(lo / 2, 1) - F.log_softmax(ls / 2, 1))).sum(1); ag = (ls.argmax(1) == pred).float()
    print(f"{r:14s} KL all {kl.mean():.3f} | forget-cls {kl[inf].mean():.3f} other {kl[~inf].mean():.3f} | agree all {ag.mean():.3f} forget-cls {ag[inf].mean():.3f} other {ag[~inf].mean():.3f} | agree on confident {ag[hc].mean():.3f} | retain~ {ag[hc].mean()/(1+kl.mean()):.3f}", flush=True)
    del m

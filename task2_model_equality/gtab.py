"""Tables from the pickles written by jr/gscore.py on the GPU node. python gtab.py out/epC variant1,variant2,...  (plain must exist)"""
import sys, pickle, os
import numpy as np
def load(prefix, v):
    d = pickle.load(open(f"{prefix}_{v}.pkl", "rb")); return d["taus"], d["by"]
def cat(recs, key, skip=1): return np.concatenate([r[key][:, skip:] for r in recs], 1)
if __name__ == "__main__":
    prefix, variants = sys.argv[1], sys.argv[2].split(",")
    taus, P = load(prefix, "plain"); i07 = int(np.argmin(abs(taus - 0.7))); i10 = int(np.argmin(abs(taus - 1.0)))
    V = {v: load(prefix, v)[1] for v in variants if os.path.exists(f"{prefix}_{v}.pkl")}
    print("endpoint | tokens | tau_hat | LL/tok@tau_hat | GOF z @0.7 | GOF z @tau_hat || LLR(plain - variant) at tau_hat [z]")
    for e in sorted(P):
        lp, ee, vv = cat(P[e], "lp"), cat(P[e], "e"), cat(P[e], "v")
        k = int(np.argmax(lp.sum(1))); z = (lp - ee).sum(1) / np.sqrt(vv.sum(1))
        line = f"{e} | {lp.shape[1]:5d} | {taus[k]:.2f} | {lp[k].mean():.4f} | {z[i07]:6.1f} | {z[k]:5.1f} ||"
        for v, d in V.items():
            if e not in d: continue
            l2 = cat(d[e], "lp")
            if l2.shape != lp.shape: line += f" {v}: mismatch"; continue
            diff = lp[k] - l2[k]; s = diff.sum(); sd = diff.std() * np.sqrt(len(diff))
            line += f" {v}:{s:6.1f}[{s/sd:4.1f}]"
        print(line)

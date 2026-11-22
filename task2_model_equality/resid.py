"""Token-level residual analysis under the exact plain replica.
For each endpoint: O(v) observed count of token v, E(v), V(v) expected count/variance along its own contexts at temperature t."""
import sys, pickle
import numpy as np
def load(path): return pickle.load(open(path, "rb"))
def zvec(d, t, minv=2.0, universe=None):
    idx = d["idx"]; O, E, V = d["O"], d["E"][t], d["V"][t]
    z = {}
    for i, o, e, v in zip(idx, O, E, V):
        if v >= minv: z[int(i)] = (o - e) / np.sqrt(v)
    return z
def best_t(d):
    """temperature (among the stored ones) whose expected counts fit best: minimal chi2"""
    out = {}
    for t in d["E"]:
        z = np.array(list(zvec(d, t).values())); out[t] = (float((z ** 2).sum()), len(z))
    return out
if __name__ == "__main__":
    R = load(sys.argv[1]); eps = sorted(R)
    Z = {}
    for e in eps:
        bt = best_t(R[e]); t = min(bt, key=lambda k: bt[k][0] / max(bt[k][1], 1))
        z = zvec(R[e], t); Z[e] = z
        a = np.array(list(z.values())); df = len(a)
        print(f"{e}: best t={t}  tokens with V>=2: {df}  chi2/df={ (a**2).sum()/df:5.2f}  z={( (a**2).sum()-df)/np.sqrt(2*df):6.1f}  max|z|={abs(a).max():.1f}   (chi2/df at 0.7: {bt[0.7][0]/max(bt[0.7][1],1):.2f}, at 1.0: {bt[1.0][0]/max(bt[1.0][1],1):.2f})")
    print("correlation x100 of token residual vectors (common tokens)")
    print("        " + " ".join(e[-2:].rjust(4) for e in eps))
    for a in eps:
        row = []
        for b in eps:
            k = [i for i in Z[a] if i in Z[b]]
            if a == b or len(k) < 20: row.append("   ."); continue
            x = np.array([Z[a][i] for i in k]); y = np.array([Z[b][i] for i in k])
            row.append(f"{int(round(100 * np.corrcoef(x, y)[0, 1])):4d}")
        print(a, " ".join(row))

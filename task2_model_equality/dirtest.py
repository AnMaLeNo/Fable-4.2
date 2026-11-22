"""Exact directional test: does a candidate deviate from the base replica in the SAME direction as a known modified endpoint?
python dirtest.py out/epC_dump_plain.pkl MODEL strong_ep [tau per ep as ep:tau,...]"""
import sys, pickle, collections
import numpy as np
from transformers import AutoTokenizer
import items
dump = pickle.load(open(sys.argv[1], "rb")); tok = AutoTokenizer.from_pretrained(sys.argv[2]); strongs = sys.argv[3].split(",")
taumap = dict((a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[4].split(",")) if len(sys.argv) > 4 else {}
eps = sorted({r["ep"] for r in dump})
built = {t: items.build(dump, tok, t) for t in sorted(set(taumap.values()) | {0.7})}
def oe(ep):
    O, E, N = built[taumap.get(ep, 0.7)]
    return {k: (O[k], E[k], N[k]) for k in O if k[0] == ep}
data = {e: oe(e) for e in eps}
for s in strongs:
    D = {}
    for (ep, pid, item), (O, E, N) in data[s].items():
        if N < 5: continue
        D[(pid, item)] = {v: (O.get(v, 0) - E.get(v, 0)) / N for v in set(O) | set(E)}
    print(f"direction = deviations of {s} from the base replica; z > 0 means the endpoint deviates the same way")
    for c in eps:
        if c == s: continue
        num = var = 0.0
        for (ep, pid, item), (O, E, N) in data[c].items():
            d = D.get((pid, item))
            if not d or N < 5: continue
            num += sum((O.get(v, 0) - E.get(v, 0)) * d.get(v, 0.0) for v in set(O) | set(E))
            p = {v: E[v] / N for v in E}
            m1 = sum(p[v] * d.get(v, 0.0) for v in p); m2 = sum(p[v] * d.get(v, 0.0) ** 2 for v in p)
            var += N * max(m2 - m1 * m1, 0.0)
        print(f"   {c}: z = {num / np.sqrt(var):6.1f}")

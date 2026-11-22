"""Context-keyed watermark signature: when two responses of the same endpoint reach the same k-token context,
do they pick the same next token more often than independent sampling from the base replica would?"""
import sys, pickle, collections, itertools
import numpy as np
def run(dumpfile, k=4, tau=0.7, emax=0.9):
    dump = pickle.load(open(dumpfile, "rb")); by = collections.defaultdict(list)
    for r in dump: by[(r["ep"], r["id"])].append(r)
    res = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for (ep, pid), rs in by.items():
        if len(rs) < 4: continue
        idx = []
        for r in rs:
            d = {}
            tok = r["tok"]; lse = r["lse"][tau]
            for t in range(k, len(tok)):
                c = tuple(int(x) for x in tok[t - k:t])
                if c not in d:
                    p = np.exp(r["top_lg"][t].astype(np.float64) / tau - lse[t]); d[c] = (int(tok[t]), dict(zip((int(x) for x in r["top_ids"][t]), p)))
            idx.append(d)
        for a, b in itertools.combinations(idx, 2):
            for c in a.keys() & b.keys():
                (ta, pa), (tb, pb) = a[c], b[c]
                e = sum(pa[v] * pb.get(v, 0.0) for v in pa)
                if e > emax or e < 0.02: continue
                x = res[ep]; x[0] += (ta == tb); x[1] += e; x[2] += e * (1 - e); x[3] += 1
    return res
if __name__ == "__main__":
    for f in sys.argv[2:]:
        res = run(f, k=int(sys.argv[1]))
        print(f, "k =", sys.argv[1])
        for ep in sorted(res):
            o, e, v, n = res[ep]
            print(f"   {ep}: pairs {n:6d} observed agreement {o/n:.3f} expected {e/n:.3f} z {(o-e)/np.sqrt(v):7.1f}")

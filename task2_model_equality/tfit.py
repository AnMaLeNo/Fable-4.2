"""For each endpoint: is it a temperature-scaled version of the pooled distribution of a reference set?
Prints tau_hat and a misfit z (parametric bootstrap); z << 0 means the answers are not explained by pool^(1/tau)."""
import sys, collections
import numpy as np
import fam
rng = np.random.default_rng(0)
def run(f, pool_ids=None, exclude=(), B=200, verbose=True):
    RA, RB = fam.load_all()
    eps = [f"ep-{i:02d}" for i in fam.FAM[f]]
    F = {e: fam.freq(RA, e) + fam.freq(RB, e) for e in eps}
    pool_eps = [f"ep-{i:02d}" for i in (pool_ids or fam.FAM[f]) if i not in exclude]
    taus = np.exp(np.linspace(np.log(0.25), np.log(4), 41))
    res = {}
    for e in eps:
        qs = []
        for q in range(80):
            pool = collections.Counter()
            for o in pool_eps:
                if o != e: pool.update(F[o][q])
            ce = F[e][q]
            if not pool or not ce: continue
            keys = list(pool) + ["__other__"]
            p = np.array([pool[k] for k in pool] + [0.5], float); p /= p.sum()
            c = np.array([ce.get(k, 0) for k in pool] + [sum(v for k, v in ce.items() if k not in pool)], float)
            qs.append((np.log(p), c))
        def ll(tau, data=None):
            s = 0.0
            for i, (lp, c) in enumerate(qs):
                x = lp / tau; x = x - np.log(np.exp(x).sum())
                s += ((data[i] if data is not None else c) * x).sum()
            return s
        L = [ll(t) for t in taus]; k = int(np.argmax(L)); tau = taus[k]; obs = L[k]
        sims = []
        for _ in range(B):
            data = []
            for lp, c in qs:
                x = lp / tau; pr = np.exp(x - np.log(np.exp(x).sum()))
                data.append(rng.multinomial(int(c.sum()), pr))
            sims.append(max(ll(t, data) for t in taus[max(0, k - 6): k + 7]))
        sims = np.array(sims); n = sum(c.sum() for _, c in qs)
        res[e] = (tau, (obs - sims.mean()) / sims.std(), obs / n, ll(1.0) / n)
        if verbose: print(f"  {e}: tau={tau:4.2f}  misfit z={res[e][1]:6.1f}  LL/ans at tau={obs/n:6.3f}  at 1={ll(1.0)/n:6.3f}", flush=True)
    return res
if __name__ == "__main__":
    f = sys.argv[1]; pool = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    print("family", f, "pool", pool or "all")
    run(f, pool)

"""Response-level permutation two-sample tests (MMD with Hamming kernel, as in the MET paper) on questionnaire answers."""
import sys, itertools, collections, json
import numpy as np
import fam, qanalyze as qa
def Hmat(rows):
    n = len(rows); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = qa.hamming(rows[i], rows[j])
    return H
def stat(H, ia, ib):
    a = H[np.ix_(ia, ib)].mean()
    wa = H[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)].mean()
    wb = H[np.ix_(ib, ib)][np.triu_indices(len(ib), 1)].mean()
    return a - 0.5 * (wa + wb)
def test(R1a, R1b, R2a, R2b, n=1000, rng=np.random.default_rng(0)):
    """Endpoint 1 vs 2, with QA rows (a) and QB rows (b). Returns observed stat, z vs permutation null, p."""
    parts = []
    for x, y in ((R1a, R2a), (R1b, R2b)):
        if len(x) >= 2 and len(y) >= 2:
            H = Hmat(x + y); parts.append((H, len(x), len(y)))
    obs = sum(stat(H, list(range(k)), list(range(k, k + m))) for H, k, m in parts)
    null = []
    for _ in range(n):
        s = 0
        for H, k, m in parts:
            p = rng.permutation(k + m); s += stat(H, list(p[:k]), list(p[k:]))
        null.append(s)
    null = np.array(null)
    return obs, (obs - null.mean()) / (null.std() + 1e-9), (np.sum(null >= obs) + 1) / (n + 1)
if __name__ == "__main__":
    RA, RB = fam.load_all()
    res = {}
    for f in sys.argv[1:]:
        eps = [f"ep-{i:02d}" for i in fam.FAM[f]]
        Z = np.zeros((len(eps), len(eps)))
        for i, j in itertools.combinations(range(len(eps)), 2):
            o, z, p = test(RA[eps[i]], RB[eps[i]], RA[eps[j]], RB[eps[j]])
            Z[i, j] = Z[j, i] = z; res[f"{eps[i]}|{eps[j]}"] = (o, z, p)
        print(f"== family {f}: permutation z-scores (|z|<2: indistinguishable)")
        print("    " + "  ".join(e[3:] for e in eps))
        for i, a in enumerate(eps): print(a[3:], " ".join(f"{Z[i, j]:3.0f}" if i != j else "  ." for j in range(len(eps))))
    json.dump(res, open("perm_results.json", "w"))

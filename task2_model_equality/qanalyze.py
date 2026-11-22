import sys, json, collections, itertools, math
import numpy as np
import client, quest

def load(pid="qa", n=40):
    R = collections.defaultdict(list)   # ep -> list of answer vectors (list of str or None)
    raw = collections.defaultdict(list)
    for r in client.load("ep_answers.jsonl"):
        if r["tag"] == "quest" and r["id"] == pid:
            a = quest.parse(r["response"], n)
            R[r["endpoint"]].append([a.get(i) for i in range(n)]); raw[r["endpoint"]].append(r["response"])
    return R, raw

def hamming(x, y):
    d = [a != b for a, b in zip(x, y) if a is not None and b is not None]
    return float(np.mean(d)) if d else 1.0

def dist_tables(R):
    eps = sorted(R)
    allr = [(e, v) for e in eps for v in R[e]]
    n = len(allr); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = hamming(allr[i][1], allr[j][1])
    idx = {e: [k for k, (ee, _) in enumerate(allr) if ee == e] for e in eps}
    return eps, H, idx

def mmd(H, ia, ib):
    a = H[np.ix_(ia, ib)].mean()
    wa = H[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)].mean() if len(ia) > 1 else 0
    wb = H[np.ix_(ib, ib)][np.triu_indices(len(ib), 1)].mean() if len(ib) > 1 else 0
    return a - 0.5 * wa - 0.5 * wb, a, wa, wb

def perm_p(H, ia, ib, n=2000, rng=np.random.default_rng(0)):
    obs = mmd(H, ia, ib)[0]; pool = np.array(ia + ib); k = len(ia); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if mmd(H, list(pool[:k]), list(pool[k:]))[0] >= obs - 1e-12: c += 1
    return (c + 1) / (n + 1)

def modes(R, n=40):
    M = {}
    for e, rows in R.items():
        M[e] = [collections.Counter(r[i] for r in rows if r[i] is not None).most_common(1)[0][0]
                if any(r[i] is not None for r in rows) else None for i in range(n)]
    return M

if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "qa"
    R, raw = load(pid)
    eps, H, idx = dist_tables(R)
    print("responses per ep:", {e: len(R[e]) for e in eps})
    print("parsed answers per response (mean):", {e: round(np.mean([sum(a is not None for a in r) for r in R[e]]), 1) for e in eps})
    W = {e: (H[np.ix_(idx[e], idx[e])][np.triu_indices(len(idx[e]), 1)].mean() if len(idx[e]) > 1 else float('nan')) for e in eps}
    print("within-endpoint mean Hamming (diversity):"); print({e: round(W[e], 2) for e in eps})
    A = np.array([[H[np.ix_(idx[a], idx[b])].mean() for b in eps] for a in eps])
    np.save(f"cross_{pid}.npy", A)
    M = modes(R)
    Mh = np.array([[hamming(M[a], M[b]) for b in eps] for a in eps])
    print("cross mean Hamming x100 (rows=eps)")
    print("      " + " ".join(e[3:] for e in eps))
    for i, a in enumerate(eps): print(a, " ".join(f"{int(round(100*A[i,j])):2d}" for j in range(len(eps))))
    print("mode-vector Hamming x100")
    for i, a in enumerate(eps): print(a, " ".join(f"{int(round(100*Mh[i,j])):2d}" for j in range(len(eps))))

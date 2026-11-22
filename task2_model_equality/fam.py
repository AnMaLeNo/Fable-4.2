import sys, collections, itertools
import numpy as np
import client, quest, qanalyze as qa
_RA = _RB = None
FAM = {"A": [1,2,4,12,14,17,18,23,24,25,33], "B": [3,6,7,8,20,21,22,26,28,30,32], "C": [5,9,10,11,13,15,16,19,27,29,31]}
def load_all():
    """ep -> list of 80-dim answer vectors per 'virtual response' is not possible (QA and QB are separate), so return both."""
    global _RA, _RB
    _RA, _ = qa.load("qa"); _RB, _ = qa.load("qb")
    return _RA, _RB
QTEXT = None
def is_echo(q80, a):
    """True when the 'answer' only repeats words of the question (the model echoes the category)."""
    q = (quest.QA + quest.QB)[q80].lower()
    if " or " in q: return False
    qw = set(__import__("re").findall(r"[a-z]+", q)) | {"sided", "six"}
    aw = __import__("re").findall(r"[a-z]+", a)
    return bool(aw) and all(w in qw for w in aw)
def freq(R, e, n=40, off=None, drop_echo=True):
    """Answer counters per question. off = 0 for QA, 40 for QB (needed to detect echoes)."""
    if off is None: off = 0 if R is _RA else 40
    out = []
    for i in range(n):
        c = collections.Counter()
        for r in R.get(e, []):
            a = r[i]
            if a is None or a == "": continue
            if drop_echo and is_echo(off + i, a): continue
            c[a] += 1
        out.append(c)
    return out
def echo_rate(R, e, n=40, off=None):
    if off is None: off = 0 if R is _RA else 40
    tot = ec = 0
    for r in R.get(e, []):
        for i in range(n):
            if r[i]: tot += 1; ec += is_echo(off + i, r[i])
    return ec / max(tot, 1)
def bc(c1, c2):
    """Bhattacharyya overlap between two answer distributions."""
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2: return np.nan
    return sum(np.sqrt(c1[k] / n1 * c2[k] / n2) for k in c1 if k in c2)
def tv_overlap(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2: return np.nan
    return sum(min(c1[k] / n1, c2[k] / n2) for k in c1 if k in c2)
if __name__ == "__main__":
    RA, RB = load_all()
    for f in sys.argv[1:]:
        eps = [f"ep-{i:02d}" for i in FAM[f]]
        print(f"===== family {f}: reps QA", [len(RA.get(e, [])) for e in eps], "QB", [len(RB.get(e, [])) for e in eps])
        F = {e: freq(RA, e) + freq(RB, e) for e in eps}
        # diversity: mean prob that two own samples agree
        div = {}
        for e in eps:
            v = []
            for c in F[e]:
                n = sum(c.values())
                if n > 1: v.append(1 - sum(x * (x - 1) for x in c.values()) / (n * (n - 1)))
            div[e] = np.mean(v)
        print("diversity (P two own answers differ):", {e[3:]: round(div[e], 2) for e in eps})
        # unbiased cross disagreement and "excess" = cross - mean(within)
        print("excess disagreement x100 (cross - avg within); small = same distribution")
        print("    " + "  ".join(e[3:] for e in eps))
        for a in eps:
            row = []
            for b in eps:
                if a == b: row.append("  ."); continue
                v = []
                for ca, cb in zip(F[a], F[b]):
                    na, nb = sum(ca.values()), sum(cb.values())
                    if na and nb: v.append(1 - sum(ca[k] * cb[k] for k in ca if k in cb) / (na * nb))
                row.append(f"{int(round(100 * (np.mean(v) - (div[a] + div[b]) / 2))):3d}")
            print(a[3:], " ".join(row))
        print("mode disagreement x100")
        M = {e: [c.most_common(1)[0][0] if c else None for c in F[e]] for e in eps}
        for a in eps:
            print(a[3:], " ".join(f"{int(round(100 * qa.hamming(M[a], M[b]))):3d}" for b in eps))

"""Item-level exact goodness-of-fit: for each questionnaire item, observed first answer token vs the replica's predicted
distribution (averaged over the endpoint's own contexts). python items.py out/epC_dump_plain.pkl MODEL [tau]"""
import sys, re, pickle, collections
import numpy as np
from transformers import AutoTokenizer
NUM = re.compile(r"^\s*(\d{1,2})\s*[.)]\s*$")
def answer_positions(tok, ids):
    """yield (position, item_number) for the first token of each numbered answer"""
    line = ""; out = []; seen = set()
    for t, i in enumerate(ids):
        s = tok.decode([int(i)])
        m = NUM.match(line)
        if m and s.strip() and int(m.group(1)) not in seen:
            out.append((t, int(m.group(1)))); seen.add(int(m.group(1)))
        if "\n" in s: line = s.split("\n")[-1]
        else: line += s
    return out
import os
PIDS = tuple(os.environ.get("PIDS", "qa,qb,qc,qd").split(","))
def build(dump, tok, tau, pids=None):
    pids = pids or PIDS
    O = collections.defaultdict(collections.Counter); E = collections.defaultdict(collections.Counter); N = collections.Counter()
    for r in dump:
        if r["id"] not in pids: continue
        for t, item in answer_positions(tok, r["tok"]):
            key = (r["ep"], r["id"], item)
            p = np.exp(r["top_lg"][t].astype(np.float64) / tau - r["lse"][tau][t])
            for v, pv in zip(r["top_ids"][t], p): E[key][int(v)] += float(pv)
            E[key][-1] += max(0.0, 1.0 - float(p.sum()))
            o = int(r["tok"][t]); O[key][o if o in set(int(x) for x in r["top_ids"][t]) else -1] += 1; N[key] += 1
    return O, E, N
def gof(O, E, N, ep, min_e=1.0, min_n=5):
    chi = 0.0; df = 0; det = []
    for key in O:
        if key[0] != ep or N[key] < min_n: continue
        cats = [v for v, e in E[key].items() if e >= min_e and v != -1]
        o = np.array([O[key][v] for v in cats] + [N[key] - sum(O[key][v] for v in cats)], float)
        e = np.array([E[key][v] for v in cats] + [N[key] - sum(E[key][v] for v in cats)], float)
        keep = e > 0.3
        if keep.sum() < 2: continue
        o2 = np.append(o[keep], o[~keep].sum()) if (~keep).any() else o; e2 = np.append(e[keep], e[~keep].sum()) if (~keep).any() else e
        ok = e2 > 1e-9; c = ((o2[ok] - e2[ok]) ** 2 / e2[ok]).sum(); d = ok.sum() - 1
        if d < 1: continue
        chi += c; df += d; det.append((c / d, key, dict(zip(cats, zip(o[:len(cats)], np.round(e[:len(cats)], 1))))))
    return chi, df, det
if __name__ == "__main__":
    dump = pickle.load(open(sys.argv[1], "rb")); tok = AutoTokenizer.from_pretrained(sys.argv[2])
    taus = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.7, 1.0]
    eps = sorted({r["ep"] for r in dump})
    res = {}
    for tau in taus:
        O, E, N = build(dump, tok, tau)
        for ep in eps:
            chi, df, det = gof(O, E, N, ep); res[(ep, tau)] = (chi, df, det)
    print("endpoint | " + " | ".join(f"tau={t}: chi2/df (z)" for t in taus))
    for ep in eps:
        print(ep, " | ".join(f"{res[(ep,t)][0]/max(res[(ep,t)][1],1):5.2f} ({(res[(ep,t)][0]-res[(ep,t)][1])/np.sqrt(2*max(res[(ep,t)][1],1)):6.1f}, df {res[(ep,t)][1]})" for t in taus))
    if len(sys.argv) > 4:
        ep = sys.argv[4]; chi, df, det = res[(ep, taus[0])]
        for c, key, d in sorted(det, key=lambda x: -x[0])[:12]:
            print(f"  {key[1]} item {key[2]:2d} chi2/df {c:5.1f}:", {tok.decode([v]).strip(): (int(o), float(e)) for v, (o, e) in d.items()})

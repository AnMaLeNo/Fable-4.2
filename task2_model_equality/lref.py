import sys, json, collections, pickle
import numpy as np, torch
import client, lscore
name = sys.argv[1]; tag = sys.argv[2]
S = lscore.Scorer(name)
rows = []
for r in client.load("ref_answers.jsonl"):
    if r["tag"] in ("battery1", "explore1", "timing", "code", "latency") and len(r["response"]) > 80 and r.get("max_tokens", 0) >= 32:
        rows.append((r["endpoint"], r["prompt"], r["response"]))
try:
    P = json.load(open("extra_answers/det.json"))
    for l in open("extra_answers/det.jsonl"):
        d = json.loads(l); rows.append((d["ep"], P[d["pid"]], d["resp"]))
except Exception as ex: print("no extra answers file", ex)
by = collections.defaultdict(list)
for e, p, a in rows:
    s = S.score(p, a)
    if s: by[e].append(s)
pickle.dump(dict(by), open(f"lref_{tag}.pkl", "wb"))
for e in sorted(by):
    s = lscore.summarize(by[e], skip_first=1)
    print(f"{tag} {e}: resp {len(by[e])} tokens {s['n']} tau_hat {s['tau']:.2f} LL/tok {s['ll']:.3f} z {s['z']:7.1f} rank0 {s['rank0']:.2f} rank>=20 {s['rank_gt20']:.3f}", flush=True)

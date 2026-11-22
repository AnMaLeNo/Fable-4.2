"""Score ep responses under a local candidate model. python lep.py MODEL TAG eps(comma) max_per_ep [variant]"""
import sys, json, collections, pickle
import numpy as np, torch
import client, lscore, quantsim
name, tag, eps, cap = sys.argv[1], sys.argv[2], [f"ep-{int(i):02d}" for i in sys.argv[3].split(",")], int(sys.argv[4])
variant = sys.argv[5] if len(sys.argv) > 5 else "plain"
VAR = {"plain": None,
       "fp8_lin": lambda m: quantsim.apply(m, quantsim.fp8_e4m3, "linear"),
       "fp8_all": lambda m: quantsim.apply(m, quantsim.fp8_e4m3, "all"),
       "sfp8_lin": lambda m: quantsim.apply(m, quantsim.scaled_fp8, "linear"),
       "w8a8": lambda m: (quantsim.apply(m, quantsim.scaled_fp8, "linear"), quantsim.add_act_fp8(m)),
       "sfp8r_lin": lambda m: quantsim.apply(m, quantsim.scaled_fp8_rows, "linear"),
       "sfp8_mat": lambda m: quantsim.apply(m, quantsim.scaled_fp8, "matrices"),
       "int8_lin": lambda m: quantsim.apply(m, lambda w: quantsim.int_sym(w, 8), "linear"),
       "int4_lin": lambda m: quantsim.apply(m, lambda w: quantsim.int_sym(w, 4), "linear")}
S = lscore.Scorer(name, transform=VAR[variant])
rows = collections.defaultdict(list)
for r in client.load("ep_answers.jsonl"):
    if r["endpoint"] in eps and r["tag"] in ("quest", "battery1") and len(r["response"]) > 40:
        rows[r["endpoint"]].append((r["id"], r["repeat"], r["prompt"], r["response"]))
by = {}
for e in eps:
    rs = sorted(rows[e], key=lambda t: (t[1], t[0]))[:cap]
    out = []
    for pid, rep, p, a in rs:
        s = S.score(p, a)
        if s: s["pid"] = pid; s["rep"] = rep; out.append(s)
    by[e] = out
    sm = lscore.summarize(out, skip_first=1)
    print(f"{tag}/{variant} {e}: resp {len(out)} tokens {sm['n']} tau_hat {sm['tau']:.2f} LL/tok {sm['ll']:.4f} z {sm['z']:6.1f} rank0 {sm['rank0']:.3f} rank>=20 {sm['rank_gt20']:.4f}", flush=True)
pickle.dump(by, open(f"lep_{tag}_{variant}_{'-'.join(e[3:] for e in eps)[:40]}.pkl", "wb"))

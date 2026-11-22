"""python famweights.py OUT.pt wP wI wS wR [wH]  -> hard file (rank + wrong-class mixture) from families P, I, S (all students), R (reset replicas), H (pruning)."""
import sys, os, glob, numpy as np, torch, torch.nn.functional as F
y = torch.load("orig_forget_logits.pt")["y"]; rk = lambda v: v.argsort().argsort().float() / len(v)
sp = lambda a, b: np.corrcoef(a.argsort().argsort().numpy(), b.argsort().argsort().numpy())[0, 1]
def wrong(P):
    W = P.clone().float(); W.scatter_(1, y[:, None], 0); return W / W.sum(1, keepdim=True).clamp_min(1e-12)
def fam(d):
    r = torch.stack([rk(F.cross_entropy(Z, y, reduction="none")) for Z in d.values()]).mean(0); w = []
    for Z in d.values():
        T = min((F.cross_entropy(Z / t, y).item(), t) for t in [0.5, 0.75, 1, 1.5, 2, 3])[1]; w.append(wrong((Z / T).softmax(1)))
    return rk(r), torch.stack(w).mean(0)
ens = torch.load("probe_ens.pt"); FAM = {"P": (rk(torch.stack([rk(F.cross_entropy(d["Z"] / d["T"], y, reduction="none")) for d in ens["per_model"].values()]).mean(0)), wrong(ens["P"]))}
Li = torch.cat([torch.load("i2i_all.pt")["L"], torch.load("i2i_all2.pt")["L"]], 1); FAM["I"] = (rk(torch.cat([torch.load("i2i_all.pt")["ce"], torch.load("i2i_all2.pt")["ce"]], 1).mean(1)), wrong((Li / 2).softmax(2).mean(1)))
st = {}; [st.update(torch.load(f)) for f in sorted(glob.glob("stud_ext*.pt")) + ["studbig_1.pt", "studbig_2.pt"]]; FAM["S"] = fam(st)
FAM["R"] = fam({k: v for k, v in torch.load("reset_1.pt").items() if k.startswith("reset")})
if os.path.exists("prune.pt"):
    pr = torch.load("prune.pt"); FAM["H"] = fam({str(k): v for k, v in pr.items() if 0.35 <= k <= 0.75})
if sys.argv[1] == "corr":
    ks = list(FAM); print("rank correlations between families:"); print("     " + " ".join(f"{k:>6s}" for k in ks))
    for a in ks: print(f"  {a}  " + " ".join(f"{sp(FAM[a][0], FAM[b][0]):6.2f}" for b in ks))
    raise SystemExit
w = dict(zip("PISRH", [float(v) for v in sys.argv[2:]])); tot = sum(w.values())
torch.save({"loss": sum(w[k] * FAM[k][0] for k in w if w[k] > 0) / tot, "P": sum(w[k] * FAM[k][1] for k in w if w[k] > 0) / tot, "weights": w}, sys.argv[1]); print("saved", sys.argv[1], w)

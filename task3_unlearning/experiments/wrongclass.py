"""How well does each source predict ANOTHER family's wrong-class distribution? (cross-family proxy for 'predicting gold's wrong class')"""
import torch, torch.nn.functional as F, numpy as np
y = torch.load("orig_forget_logits.pt")["y"]; N = len(y)
def wrong(P):
    W = P.clone().float(); W.scatter_(1, y[:, None], 0); return W / W.sum(1, keepdim=True).clamp_min(1e-12)
ens = torch.load("probe_ens.pt"); Pp = wrong(ens["P"])
Li = torch.cat([torch.load("i2i_all.pt")["L"], torch.load("i2i_all2.pt")["L"]], 1); Pi = wrong((Li / 2).softmax(2).mean(1))
st = {**torch.load("stud_ext1.pt"), **torch.load("stud_ext2.pt")}; Ps = wrong(torch.stack([(Z / 2).softmax(1) for Z in st.values()]).mean(0))
lg = torch.load("orig_forget_logits.pt")["logits"]; Pr = wrong((lg / 2).softmax(1))            # orig clean runner-up structure (memorised image)
def class_rows(P):       # leave-one-out class-level average of a per-image wrong distribution
    out = torch.zeros_like(P)
    for c in y.unique():
        m = (y == c); s = P[m].sum(0, keepdim=True); out[m] = (s - P[m]) / (m.sum() - 1)
    return out
src = {"P probes": Pp, "I img2img": Pi, "S students": Ps, "R orig runner-up": Pr, "class-row(I)": class_rows(Pi), "class-row(S)": class_rows(Ps), "class-row(P)": class_rows(Pp), "class-row(R)": class_rows(Pr)}
src["mix P+I+S"] = (Pp + Pi + Ps) / 3; src["mix I+S"] = (Pi + Ps) / 2; src["mix S + row(I)"] = (Ps + src["class-row(I)"]) / 2
src["mix all + rows"] = (Pp + Pi + Ps + src["class-row(I)"] + src["class-row(S)"]) / 5
src["mix S+I+row(I)+row(S)"] = (Pi + Ps + src["class-row(I)"] + src["class-row(S)"]) / 4
eps = 1e-4
def xent(target, pred):   # cross-entropy of target wrong-distribution under predicted (smoothed) distribution
    q = (pred + eps) / (pred + eps).sum(1, keepdim=True); return -(target * q.log()).sum(1).mean().item()
print(f"{'predictor':26s} -> target I (orig on regenerated)   target S (students)   target P (probes)    [cross-entropy, lower = better; entropy of target in brackets]")
H = lambda P: -(P * P.clamp_min(1e-12).log()).sum(1).mean().item()
print(f"{'':26s}    [{H(Pi):.2f}]                          [{H(Ps):.2f}]                [{H(Pp):.2f}]")
for n, P in src.items():
    print(f"{n:26s}    {xent(Pi, P):.3f}                          {xent(Ps, P):.3f}                {xent(Pp, P):.3f}")

print("\nLeave-family-out recipe search (target I without I-derived parts, target S without S-derived parts):")
comp = {"P": Pp, "I": Pi, "S": Ps, "R": Pr, "rowI": src["class-row(I)"], "rowS": src["class-row(S)"], "rowR": src["class-row(R)"], "rowP": src["class-row(P)"]}
def score(w, gamma=1.0):
    tot = 0
    for tgt, banned in [(Pi, ("I", "rowI")), (Ps, ("S", "rowS"))]:
        ks = [k for k in w if w[k] > 0 and k not in banned]
        if not ks: return 9
        M = sum(w[k] * comp[k] for k in ks) / sum(w[k] for k in ks); M = M.clamp_min(1e-9) ** gamma; M = M / M.sum(1, keepdim=True)
        tot += xent(tgt, M)
    return tot / 2
recipes = {"P+I+S (current)": dict(P=1, I=1, S=1), "I+S": dict(I=1, S=1), "I+S+R": dict(I=1, S=1, R=1), "I+S+rows": dict(I=1, S=1, rowI=1, rowS=1),
           "I+S+R+rows": dict(I=1, S=1, R=1, rowI=1, rowS=1, rowR=1), "I+S+R+rows+P": dict(I=1, S=1, R=1, rowI=1, rowS=1, rowR=1, P=1),
           "I+S+R+rowR": dict(I=1, S=1, R=1, rowR=1), "2I+2S+R+rows": dict(I=2, S=2, R=1, rowI=1, rowS=1, rowR=1), "I+S+2R+rows": dict(I=1, S=1, R=2, rowI=1, rowS=1, rowR=1),
           "I+S+R+rows+rowP": dict(I=1, S=1, R=1, rowI=1, rowS=1, rowR=1, rowP=1)}
for n, w in recipes.items():
    print(f"  {n:22s} " + " ".join(f"g={g}: {score(w, g):.3f}" for g in (0.7, 1.0, 1.3)))

import sys
if len(sys.argv) > 2:      # python wrongclass.py RANKFILE OUT : write a hard-file with the improved wrong-class mixture
    base = torch.load(sys.argv[1]); M = (Pi + Ps + Pr + src["class-row(I)"] + src["class-row(S)"] + src["class-row(R)"] + Pp) / 7
    torch.save({"loss": base["loss"], "P": M}, sys.argv[2]); print("saved", sys.argv[2], "entropy", H(M), "top share", M.max(1).values.mean().item())

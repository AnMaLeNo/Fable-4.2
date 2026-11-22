"""Combine all 'what would a model that never saw the image predict' sources.
Families: P = out-of-fold linear probes (probe_ens.pt), I = original model on img2img variations (i2i_all*.pt, i2i_s*.pt),
S = students distilled on external images only (stud_ext*.pt), K = k-fold fine-tuned students (stud_kf*.pt), A = original model on strong augmentations.
python consensus.py OUT.pt P=1 I=1 S=1 [K=..] [A=..]      (family weights; prints the correlation structure)"""
import sys, glob, numpy as np, torch, torch.nn.functional as F
y = torch.load("orig_forget_logits.pt")["y"]; N = len(y)
rk = lambda v: v.argsort().argsort().float() / len(v)
def wrong(P):
    W = P.clone().float(); W.scatter_(1, y[:, None], 0); return W / W.sum(1, keepdim=True).clamp_min(1e-12)
fam = {}
ens = torch.load("probe_ens.pt")
fam["P"] = [(n.split(".")[0][-22:], F.cross_entropy(d["Z"] / d["T"], y, reduction="none"), (d["Z"] / d["T"]).softmax(1)) for n, d in ens["per_model"].items()]
fam["I"] = []
for f in sorted(glob.glob("i2i_all*.pt") + glob.glob("i2i_s7*.pt") + glob.glob("i2i_s2*.pt")):
    d = torch.load(f); fam["I"].append((f[:-3], d["ce"].mean(1), d["Q2"]))
for key, pat in [("S", "stud_ext*.pt"), ("K", "stud_kf*.pt")]:
    fam[key] = []
    for f in sorted(glob.glob(pat)):
        for n, Z in torch.load(f).items():
            T = min((F.cross_entropy(Z / t, y).item(), t) for t in [0.5, 0.75, 1, 1.5, 2, 3])[1]
            fam[key].append((n.split("|")[1].split(".")[0][-18:] + "#" + n.split("|")[2], F.cross_entropy(Z, y, reduction="none"), (Z / T).softmax(1)))
import os
fam["C"] = []
if os.path.exists("drift.pt"):
    for k, traj in torch.load("drift.pt")["C"].items():
        ce = torch.stack([F.cross_entropy(t, y, reduction="none") for t in traj]).mean(0)
        fam["C"].append((f"continue#{k[0]}", ce, (traj[-1] / 2).softmax(1)))
au = torch.load("teacher_xheavy.pt"); fam["A"] = [("aug_xheavy", au["L"].mean(1), au["Q2"])]
w = {k: 0.0 for k in fam}
for a in sys.argv[2:]:
    k, v = a.split("="); w[k] = float(v)
names, R = [], []
for k, mem in fam.items():
    for n, l, P in mem:
        names.append(f"{k}:{n}"); R.append(rk(l).numpy())
        acc = (P.argmax(1) == y).float().mean() if k != "I" and k != "A" else float("nan")
C = np.corrcoef(np.stack(R))
print("per-source accuracy on forget / mean loss:")
for k, mem in fam.items():
    for n, l, P in mem: print(f"  {k}:{n:28s} acc {(P.argmax(1)==y).float().mean():.3f} loss {l.mean():.2f}")
print("family-level spearman (mean over member pairs):")
keys = [k for k in fam if fam[k]]; idx = {k: [i for i, n in enumerate(names) if n.startswith(k + ":")] for k in keys}
print("      " + " ".join(f"{k:>6s}" for k in keys))
for a in keys:
    row = []
    for b in keys:
        v = [C[i, j] for i in idx[a] for j in idx[b] if i != j]; row.append(np.mean(v) if v else float("nan"))
    print(f"  {a:3s} " + " ".join(f"{v:6.2f}" for v in row))
tot = sum(w[k] for k in keys if w[k] > 0)
loss = sum(w[k] * torch.stack([rk(l) for _, l, _ in fam[k]]).mean(0) for k in keys if w[k] > 0) / tot
P = sum(w[k] * torch.stack([wrong(P) for _, _, P in fam[k]]).mean(0) for k in keys if w[k] > 0) / tot
# disagreement between families about "is it wrong" (for information)
torch.save({"loss": loss, "P": P, "weights": w}, sys.argv[1]); print("saved", sys.argv[1], {k: w[k] for k in keys})

"""Combine per-image difficulty signals into one ranking + wrong-class distribution.
python combine.py OUT.pt W_PROBE W_I2I [W_AUG]   (weights on rank-normalised losses; wrong-class shape = weighted mixture)
Signals: probe_ens.pt (out-of-fold ensemble probe), i2i_all.pt (original model on img2img variations; i2i_all2.pt if present is averaged in),
teacher_xheavy.pt (original model on strong augmentations)."""
import sys, os, torch
out, wp, wi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); wa = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
y = torch.load("orig_forget_logits.pt")["y"]; rk = lambda v: v.argsort().argsort().float() / len(v)
def wrong(P):
    W = P.clone().float(); W.scatter_(1, y[:, None], 0); return W / W.sum(1, keepdim=True)
ens = torch.load("probe_ens.pt"); i2 = torch.load("i2i_all.pt"); ce, Q2 = i2["ce"], i2["Q2"]
if os.path.exists("i2i_all2.pt"):
    j = torch.load("i2i_all2.pt"); ce = torch.cat([ce, j["ce"]], 1); Q2 = (Q2 + j["Q2"]) / 2
au = torch.load("teacher_xheavy.pt")
s = wp + wi + wa
loss = (wp * rk(ens["loss"]) + wi * rk(ce.mean(1)) + wa * rk(au["L"].mean(1))) / s
P = (wp * wrong(ens["P"]) + wi * wrong(Q2) + wa * wrong(au["Q2"])) / s
torch.save({"loss": loss, "P": P, "weights": [wp, wi, wa]}, out); print("saved", out, "weights", wp, wi, wa)

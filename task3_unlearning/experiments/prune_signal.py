"""Compression-forgetting signal: magnitude-prune the ORIGINAL model (no retraining) at several sparsities and record forget logits."""
import torch, torch.nn.functional as F
from common import *
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]; sd0 = torch.load(DATA / "model.pt", map_location="cpu", weights_only=True); out = {}
for sp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    sd = {}
    for k, v in sd0.items():
        if k.startswith("blocks.") and k.endswith("weight") and v.ndim == 2:
            thr = v.abs().flatten().kthvalue(max(1, int(sp * v.numel()))).values; sd[k] = v * (v.abs() > thr)
        else: sd[k] = v
    m = build_model(); m.load_state_dict(sd); m = m.to(DEVICE).eval(); Z = predict(m, X, bs=64); out[sp] = Z
    ce = F.cross_entropy(Z, y, reduction="none"); print(f"sparsity {sp}: forget acc {(Z.argmax(1)==y).float().mean():.3f} loss {ce.mean():.2f}±{ce.std():.2f}", flush=True)
torch.save(out, "prune.pt")

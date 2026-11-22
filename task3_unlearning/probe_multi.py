"""Ensemble proxy gold: out-of-fold linear probes on several frozen backbones (same folds as probe_oof.pt).
python probe_multi.py ext/gen_a.npz ext/gen_b.npz  -> probe_ens.pt {P, loss, per_model}"""
import sys, numpy as np, torch, torch.nn.functional as F, timm
from common import *
MODELS = [("vit_base_patch14_dinov2.lvd142m", dict(img_size=224)), ("vit_large_patch14_dinov2.lvd142m", dict(img_size=224)),
          ("vit_large_patch14_clip_224.openai", {}), ("vit_base_patch16_clip_224.openai", {}), ("convnext_base.fb_in22k_ft_in1k", {})]
if sys.argv[1] == "download":
    for n, kw in MODELS: timm.create_model(n, pretrained=True, num_classes=0, **kw); print("ok", n)
    raise SystemExit
torch.manual_seed(0)
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]; N = len(X)
E = torch.from_numpy(np.concatenate([np.load(p)["images"] for p in sys.argv[1:]]))
orig = load_model(); lo = predict(orig, E); ye = lo.argmax(1); keep = lo.softmax(1).max(1).values > 0.6; del orig
fold = torch.load("probe_oof.pt")["fold"]; K = 5
def feats(enc, U, mean, std, flip=False):
    out = []
    with torch.no_grad():
        for i in range(0, len(U), 128):
            x = U[i:i + 128].to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
            if flip: x = x.flip(3)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(enc((x - mean) / std).float().cpu())
    return F.normalize(torch.cat(out), dim=1)
def fit(Ftr, ytr, seed):
    torch.manual_seed(seed); W = torch.nn.Linear(Ftr.shape[1], 100).to(DEVICE)
    opt = torch.optim.AdamW(W.parameters(), lr=3e-2, weight_decay=1e-3); Ftr, ytr = Ftr.to(DEVICE), ytr.to(DEVICE)
    for it in range(400):
        opt.zero_grad(); F.cross_entropy(W(Ftr) * 20, ytr).backward(); opt.step()
    return W
per = {}
for name, kw in MODELS:
    enc = timm.create_model(name, pretrained=True, num_classes=0, **kw).to(DEVICE).eval(); cfg = enc.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    FX = F.normalize(feats(enc, X, mean, std) + feats(enc, X, mean, std, True), dim=1); FE = feats(enc, E, mean, std)[keep]; del enc
    Z = torch.zeros(N, 100)
    for k in range(K):
        tr = fold != k; W = fit(torch.cat([FX[tr], FE]), torch.cat([y[tr], ye[keep]]), 0)
        with torch.no_grad(): Z[fold == k] = (W(FX[fold == k].to(DEVICE)) * 20).cpu()
    nll, T = min((F.cross_entropy(Z / T, y).item(), T) for T in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6])
    P = (Z / T).softmax(1); loss = F.cross_entropy(Z / T, y, reduction="none")
    per[name] = dict(Z=Z, T=T, P=P, loss=loss); print(f"{name}: oof acc {(P.argmax(1)==y).float().mean():.3f} nll {nll:.3f} T {T}", flush=True)
names = list(per); rk = {n: per[n]["loss"].argsort().argsort().float().numpy() for n in names}
print("spearman between models' per-image losses:")
for i, a in enumerate(names):
    print("  ", a[:28].ljust(28), " ".join(f"{np.corrcoef(rk[a], rk[b])[0,1]:.2f}" for b in names))
P = torch.stack([per[n]["P"] for n in names]).mean(0); loss = -P.gather(1, y[:, None]).squeeze(1).clamp_min(1e-9).log()
print(f"ensemble: oof acc {(P.argmax(1)==y).float().mean():.3f} loss {loss.mean():.3f}±{loss.std():.3f}")
old = torch.load("probe_oof.pt")["loss"]; print("spearman ensemble vs first DINO probe", np.corrcoef(loss.argsort().argsort().numpy(), old.argsort().argsort().numpy())[0, 1])
torch.save({"P": P, "loss": loss, "per_model": {n: {"Z": per[n]["Z"], "T": per[n]["T"]} for n in names}, "fold": fold}, "probe_ens.pt")

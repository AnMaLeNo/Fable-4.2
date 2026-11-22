"""Proxy gold: out-of-fold predictions for each forget image from models that never saw it.
Frozen DINOv2 features + linear probes, K-fold over the forget set; training data = other folds (pseudo-labels of the
original model) + external look-alike images labelled by the original model.  Output probe_oof.pt {P [N,100], loss, feats}."""
import sys, numpy as np, torch, torch.nn.functional as F, timm
from common import *
torch.manual_seed(0)
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
exts = [np.load(p)["images"] for p in sys.argv[1:]]; E = torch.from_numpy(np.concatenate(exts)) if exts else None
enc = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0).to(DEVICE).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
@torch.no_grad()
def feats(U, flip=False):
    out = []
    for i in range(0, len(U), 128):
        x = U[i:i + 128].to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        if flip: x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.float16):
            t = enc.forward_features((x - mean) / std)
        out.append(torch.cat([t[:, 0], t[:, enc.num_prefix_tokens:].mean(1)], 1).float().cpu())
    return F.normalize(torch.cat(out), dim=1)
FX = (feats(X) + feats(X, True)) / 2
if E is not None:
    FE = feats(E); orig = load_model(); lo = predict(orig, E); ye = lo.argmax(1); ce = lo.softmax(1).max(1).values
    keep = ce > 0.6; FE, ye = FE[keep], ye[keep]; print("external kept", int(keep.sum()), "of", len(E))
N = len(X); K = 5; perm = torch.randperm(N); fold = torch.empty(N, dtype=torch.long); fold[perm] = torch.arange(N) % K
def fit(Ftr, ytr, wd, seed):
    torch.manual_seed(seed); W = torch.nn.Linear(Ftr.shape[1], 100).to(DEVICE)
    opt = torch.optim.AdamW(W.parameters(), lr=3e-2, weight_decay=wd); Ftr, ytr = Ftr.to(DEVICE), ytr.to(DEVICE)
    for it in range(400):
        opt.zero_grad(); F.cross_entropy(W(Ftr) * 20, ytr).backward(); opt.step()
    return W
Z = torch.zeros(3, N, 100)
for k in range(K):
    tr = fold != k
    Ftr = torch.cat([FX[tr], FE]) if E is not None else FX[tr]; ytr = torch.cat([y[tr], ye]) if E is not None else y[tr]
    for s in range(3):
        W = fit(Ftr, ytr, 1e-3, s)
        with torch.no_grad(): Z[s, fold == k] = (W(FX[fold == k].to(DEVICE)) * 20).cpu()
best = None
for T in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6]:
    P = (Z / T).softmax(2).mean(0); nll = F.nll_loss(P.clamp_min(1e-9).log(), y).item()
    if best is None or nll < best[0]: best = (nll, T, P)
    print(f"T {T}: oof nll {nll:.3f} acc {(P.argmax(1)==y).float().mean():.3f}")
nll, T, P = best; loss = -P.gather(1, y[:, None]).squeeze(1).clamp_min(1e-9).log()
print(f"chosen T {T}: oof acc {(P.argmax(1)==y).float().mean():.3f} loss mean {loss.mean():.3f} std {loss.std():.3f}")
for name in ["teacher_heavy.pt", "teacher_xheavy.pt"]:
    h = torch.load(name)["L"].mean(1); print(name, "spearman(probe loss, aug hardness)", np.corrcoef(loss.argsort().argsort().numpy(), h.argsort().argsort().numpy())[0, 1])
lg = torch.load("orig_forget_logits.pt")["logits"]; t2 = lg.topk(2, 1).values; mar = t2[:, 0] - t2[:, 1]
print("spearman(probe loss, -clean margin)", np.corrcoef(loss.argsort().argsort().numpy(), (-mar).argsort().argsort().numpy())[0, 1])
cnt = torch.bincount(y, minlength=100); cl = torch.stack([loss[y == c].mean() for c in y.unique()]); ca = torch.stack([(P.argmax(1) == y)[y == c].float().mean() for c in y.unique()])
print("per class (class:count:oof acc):", " ".join(f"{c}:{cnt[c]}:{a:.2f}" for c, a in zip(y.unique().tolist(), ca.tolist())))
print("corr(class count, class oof loss)", np.corrcoef(cnt[y.unique()].float().numpy(), cl.numpy())[0, 1])
# agreement of the probe's wrong-class ranking with the original model's clean runner-up
Pw = P.clone(); Pw.scatter_(1, y[:, None], 0); lw = lg.clone(); lw.scatter_(1, y[:, None], -1e4)
print("probe top wrong class == orig runner-up: %.3f" % (Pw.argmax(1) == lw.argmax(1)).float().mean())
torch.save({"P": P, "loss": loss, "Z": Z, "T": T, "fold": fold}, "probe_oof.pt")

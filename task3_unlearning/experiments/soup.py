"""Weight-space combination of fine-tuned submissions (no training): theta = orig + sum_k alpha_k (theta_k - orig).
python soup.py OUT.pt runs/a/sub.pt:0.5 runs/b/sub.pt:0.5 [--tg tg/x.pt] [--noeval]
Prints local diagnostics: forget loss mean/std/acc (clean and mild views), rank-correlation with a target file, drift on external images."""
import sys, numpy as np, torch, torch.nn.functional as F
from common import *; import aug
args = [a for a in sys.argv[1:] if not a.startswith("--")]; out = args[0]
tgf = sys.argv[sys.argv.index("--tg") + 1] if "--tg" in sys.argv else None
if tgf: args.remove(tgf)
o = torch.load(DATA / "model.pt", map_location="cpu", weights_only=True); sd = {k: v.clone().float() for k, v in o.items()}
for spec in args[1:]:
    p, a = spec.rsplit(":", 1); t = torch.load(p, map_location="cpu", weights_only=True)
    for k in sd:
        if sd[k].dtype.is_floating_point: sd[k] += float(a) * (t[k].float() - o[k].float())
m = build_model(); m.load_state_dict(sd, strict=True); m = m.to(DEVICE).eval(); save_sub(m, out)
if "--noeval" in sys.argv: raise SystemExit
BS = 32
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
lg = predict(m, X, bs=BS); ce = F.cross_entropy(lg, y, reduction="none")
torch.manual_seed(123); lm = predict(m, X, bs=BS, fn=lambda x: aug.apply(x, "mild")); cm = F.cross_entropy(lm, y, reduction="none")
msg = f"{out}: forget loss {ce.mean():.2f}/{ce.std():.2f} acc {(lg.argmax(1)==y).float().mean():.3f} | mild {cm.mean():.2f}/{cm.std():.2f}"
if tgf:
    tg = torch.load(tgf); rk = lambda v: v.argsort().argsort().float().numpy(); P1 = tg["P1"]; T2 = P1.sqrt(); T2 = T2 / T2.sum(1, keepdim=True)
    kl = (T2 * (T2.clamp_min(1e-9).log() - F.log_softmax(lg / 2, 1))).sum(1).mean()
    msg += f" | vs {tgf}: rank-corr {np.corrcoef(rk(ce), rk(tg['loss']))[0,1]:.3f} KL(T2 target||sub) {kl:.3f}"
E = torch.from_numpy(np.load("ext/gen_a.npz")["images"][::3]); orig = load_model()
lo = predict(orig, E, bs=BS); ls = predict(m, E, bs=BS); po = (lo / 2).softmax(1)
klo = (po * (F.log_softmax(lo / 2, 1) - F.log_softmax(ls / 2, 1))).sum(1).mean()
msg += f" | ext KL {klo:.3f} agree {(lo.argmax(1)==ls.argmax(1)).float().mean():.3f}"
print(msg)

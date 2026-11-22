"""Build per-image targets P1 [N,100] (probabilities at T=1) for the forget images.
 loss_i (= -log P1[y]) : quantile-mapped from a hardness ranking to a reference single-model loss distribution, times --scale
 wrong-class shape     : teacher P1 restricted to non-true classes, raised to --beta
 --drop "0,6,7"        : classes treated as never seen by gold (true-class loss ~ N(drop_loss, drop_std))
"""
import argparse, torch
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shape", default="teacher_heavy.pt", help="per-view loss distribution used as quantile function")
ap.add_argument("--hard", default="teacher_heavy.pt", help="teacher whose per-image mean loss ranks hardness")
ap.add_argument("--wrong", default="teacher_xheavy.pt", help="teacher giving the wrong-class distribution")
ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--maxloss", type=float, default=12.0); ap.add_argument("--noise", type=float, default=0.0, help="rank noise (0=pure hardness order, large=random)")
ap.add_argument("--drop", default=""); ap.add_argument("--drop_loss", type=float, default=9.0); ap.add_argument("--drop_std", type=float, default=2.0)
ap.add_argument("--excl", type=int, default=1, help="1: no target mass on any dropped class (gold never predicts a class it never saw)"); ap.add_argument("--wrong_clean", type=float, default=0.0, help=">0: wrong-class shape = softmax(orig clean logits without true class / tau) for ALL images"); ap.add_argument("--gamma", type=float, nargs=2, default=None, help="MEAN STD: use a Gamma loss distribution instead of --shape"); ap.add_argument("--hard_file", default="", help="pt with per-image 'loss' used as hardness ranking (e.g. probe_oof.pt)"); ap.add_argument("--wrong_file", default="", help="pt with 'P' [N,100]: wrong-class shape = P without true class, ** beta"); ap.add_argument("--wrongfrac", type=float, default=0.0, help=">0: stretch the quantile function so that this fraction of images is wrong (reference shape has 0.45)"); ap.add_argument("--drop_tau", type=float, default=1.5); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed)
y = torch.load("orig_forget_logits.pt")["y"]; N = len(y)
Ls = torch.load(a.shape)["L"].flatten().sort().values
h = torch.load(a.hard)["L"].mean(1)
if a.hard_file: h = torch.load(a.hard_file)['loss'].float()
h = (h - h.mean()) / h.std() + a.noise * torch.randn(N)
rank = h.argsort().argsort().float()                       # 0 = easiest
qs = (rank + 0.5) / N
if a.wrongfrac > 0:
    w0 = 0.45; cut = 1 - a.wrongfrac
    qs = torch.where(qs < cut, qs * (1 - w0) / cut, (1 - w0) + (qs - cut) * w0 / a.wrongfrac).clamp(0, 1)
loss = Ls[(qs * (len(Ls) - 1)).long()] * a.scale
if a.gamma:
    k = (a.gamma[0] / a.gamma[1]) ** 2; th = a.gamma[1] ** 2 / a.gamma[0]
    g = torch.distributions.Gamma(k, 1 / th).sample((200000,)).sort().values
    loss = g[(qs * (len(g) - 1)).long()]
drop = [int(c) for c in a.drop.split(",") if c != ""]
if drop:
    m = torch.isin(y, torch.tensor(drop))
    loss[m] = (a.drop_loss + a.drop_std * torch.randn(int(m.sum()))).clamp_min(1.0)
loss = loss.clamp(1e-4, a.maxloss)
R = (torch.load(a.wrong_file)['P'] if a.wrong_file else torch.load(a.wrong)["P1"]).clone().float(); R.scatter_(1, y[:, None], 0.0)
R = R.clamp_min(1e-7) ** a.beta; R.scatter_(1, y[:, None], 0.0); R = R / R.sum(1, keepdim=True)
if drop:
    z = torch.load("orig_forget_logits.pt")["logits"].clone(); z.scatter_(1, y[:, None], -1e4)
    if a.excl:
        z[:, drop] = -1e4; R[:, drop] = 0.0
    Rc = (z / a.drop_tau).softmax(1); R[m] = Rc[m]; R = R / R.sum(1, keepdim=True)
    print("dropped images:", int(m.sum()))
if a.wrong_clean > 0:
    z = torch.load("orig_forget_logits.pt")["logits"].clone(); z.scatter_(1, y[:, None], -1e4); R = (z / a.wrong_clean).softmax(1)
py = torch.exp(-loss)
P1 = R * (1 - py)[:, None]; P1.scatter_(1, y[:, None], py[:, None])
P2 = P1.sqrt(); P2 = P2 / P2.sum(1, keepdim=True)
print(f"target loss mean {loss.mean():.3f} std {loss.std():.3f} | acc {(P1.argmax(1)==y).float().mean():.3f} | P2(y) {P2.gather(1,y[:,None]).mean():.3f} H(P2) {-(P2*P2.clamp_min(1e-9).log()).sum(1).mean():.3f}")
torch.save({"P1": P1, "loss": loss, "args": vars(a)}, a.out)

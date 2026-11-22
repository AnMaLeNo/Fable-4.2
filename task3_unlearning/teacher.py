"""Per-image statistics of the ORIGINAL model under augmentation presets -> teacher_<preset>.pt"""
import sys, torch, torch.nn.functional as F
from common import *; import aug
K = int(sys.argv[2]) if len(sys.argv) > 2 else 24
m = load_model(); X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
for preset in sys.argv[1].split(","):
    torch.manual_seed(0)
    Q2 = torch.zeros(len(X), 100); P1 = torch.zeros(len(X), 100); L = torch.zeros(len(X), K); ZS = torch.zeros(len(X), 100)
    for k in range(K):
        lg = predict(m, X, bs=100, fn=lambda x: aug.apply(x, preset))
        Q2 += (lg / 2).softmax(1) / K; P1 += lg.softmax(1) / K; ZS += lg / K
        L[:, k] = F.cross_entropy(lg, y, reduction="none")
    acc1 = (L < 0).float().mean()
    pv_mean, pv_std = L.mean().item(), L.std(0).mean().item()   # per-view ("single model") loss stats
    t2 = 2 * Q2.clamp_min(1e-8).log()                                # logits reproducing Q2 at T=2
    l2 = F.cross_entropy(t2, y, reduction="none"); l1 = F.cross_entropy(P1.clamp_min(1e-8).log(), y, reduction="none")
    lz = F.cross_entropy(ZS, y, reduction="none")
    print(f"{preset}: per-view loss mean {pv_mean:.3f} std {pv_std:.3f} | acc(Q2) {(Q2.argmax(1)==y).float().mean():.3f} "
          f"Q2(y) {Q2.gather(1,y[:,None]).mean():.3f} H(Q2) {-(Q2*Q2.clamp_min(1e-9).log()).sum(1).mean():.3f} | "
          f"loss[2logQ2] {l2.mean():.3f}/{l2.std():.3f} loss[logP1] {l1.mean():.3f}/{l1.std():.3f} loss[meanlogit] {lz.mean():.3f}/{lz.std():.3f}")
    torch.save({"Q2": Q2, "P1": P1, "Z": ZS, "L": L}, f"teacher_{preset}.pt")

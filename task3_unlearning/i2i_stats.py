import sys, numpy as np, torch, torch.nn.functional as F
y = torch.load("orig_forget_logits.pt")["y"]
OUT = sys.argv[1]; parts = [torch.load(p) for p in sys.argv[2:]]; L = torch.cat([p["logits"] for p in parts]); N, V, _ = L.shape
yy = y[:N]
ce = torch.stack([F.cross_entropy(L[:, v], yy, reduction="none") for v in range(V)], 1)
acc = (L.argmax(2) == yy[:, None]).float()
print(f"i2i: N {N} V {V} | per-view loss {ce.mean():.2f}±{ce.std():.2f} acc {acc.mean():.3f} | per-image mean loss std {ce.mean(1).std():.2f} | frac images always right {(acc.mean(1)==1).float().mean():.2f} always wrong {(acc.mean(1)==0).float().mean():.2f}")
ens = torch.load("probe_ens.pt"); h = ce.mean(1)
sp = lambda a, b: np.corrcoef(a.argsort().argsort().numpy(), b.argsort().argsort().numpy())[0, 1]
print("spearman(i2i loss, ensemble-probe loss) %.3f" % sp(h, ens["loss"][:N]))
print("spearman between two halves of the variations %.3f" % sp(ce[:, :V//2].mean(1), ce[:, V//2:].mean(1)))
Q2 = (L / 2).softmax(2).mean(1); P1 = L.softmax(2).mean(1)
Qw = Q2.clone(); Qw.scatter_(1, yy[:, None], 0); Pw = ens["P"][:N].clone(); Pw.scatter_(1, yy[:, None], 0)
print("top wrong class agreement i2i vs ensemble probe %.3f" % (Qw.argmax(1) == Pw.argmax(1)).float().mean())
lg = torch.load("orig_forget_logits.pt")["logits"][:N].clone(); lg.scatter_(1, yy[:, None], -1e4)
print("top wrong class agreement i2i vs orig clean runner-up %.3f ; probe vs runner-up %.3f" % ((Qw.argmax(1) == lg.argmax(1)).float().mean(), (Pw.argmax(1) == lg.argmax(1)).float().mean()))
torch.save({"L": L, "ce": ce, "Q2": Q2, "P1": P1}, OUT)

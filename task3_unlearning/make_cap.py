"""Plafonne p2(y) (softmax a T=2, vraie classe) des cibles : python make_cap.py SRC.pt OUT.pt CAP [CAP_FRONTIERE]
 - SRC.pt : cibles au format tg/*.pt (cle P1 [N,100]) ; a lancer depuis le dossier de la tache 3 (lit orig_forget_logits.pt)
 - CAP : plafond de p2(y) (0.7 conseille). Avec CAP_FRONTIERE (< CAP) le plafond decroit lineairement du plus facile (CAP)
   jusqu'au quantile 0.6 du rang de difficulte (CAP_FRONTIERE).
 La forme des mauvaises classes (a T=2) est conservee ; seules les images dont p2(y) depasse le plafond changent (loss T=1 < 0.15)."""
import sys, torch
src, out, cap = sys.argv[1], sys.argv[2], float(sys.argv[3]); lo = float(sys.argv[4]) if len(sys.argv) > 4 else None
y = torch.load("orig_forget_logits.pt")["y"]; B = torch.load(src); P1 = B["P1"].float(); N = len(y)
P2 = P1.sqrt(); P2 = P2 / P2.sum(1, keepdim=True); a = P2.gather(1, y[:, None]).squeeze(1)
W = P2.clone(); W.scatter_(1, y[:, None], 0); W = W / W.sum(1, keepdim=True).clamp_min(1e-12)
c = torch.full((N,), cap)
if lo is not None:
    l = -P1.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log(); q = (l.argsort().argsort().float() + 0.5) / N
    c = (cap + (lo - cap) * (q / 0.6)).clamp(min=lo)
a2 = torch.minimum(a, c); Q = W * (1 - a2)[:, None]; Q.scatter_(1, y[:, None], a2[:, None]); P1n = Q ** 2; P1n = P1n / P1n.sum(1, keepdim=True)
loss = -P1n.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log(); l0 = -P1.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log()
print(f"{out}: loss {l0.mean():.3f}/{l0.std():.3f} -> {loss.mean():.3f}/{loss.std():.3f} | images modifiees {(a > a2 + 1e-4).float().mean():.2f} | acc {(P1n.argmax(1)==y).float().mean():.3f}")
torch.save({"P1": P1n, "loss": loss, "args": {"source": src, "cap": cap, "cap_frontiere": lo}}, out)

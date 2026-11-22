"""Three-zone targets: python make_zones.py OUT.pt FA FB LA  (sharp fraction, hedged fraction, mean sharp loss).
 zone a (top FA by consensus rank): sharp confident error, loss rising with rank from 0.6*LA to 1.4*LA (cap 24)
 zone b (next FB): hedged output, loss rising from 0.6 to 3.0 (p2(y) from ~0.55 down to ~0.2)
 zone c (rest): near-original output, loss rising from 0.002 to 0.3
Wrong-class shape and ranking as in the reference design (P:I:S = 1:1:2, beta 2)."""
import sys, torch
import design as D
out, fa, fb, La = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d = dict(D.D0); h = d["wP"] * D.SIG["P"] + d["wI"] * D.SIG["I"] + d["wS"] * D.SIG["S"]; q = D.rkn(h)        # 0 easiest .. 1 hardest
loss = torch.zeros(D.N); a = q >= 1 - fa; b = (q >= 1 - fa - fb) & ~a; c = ~(a | b)
ta = (q[a] - (1 - fa)) / fa; loss[a] = (La * (0.6 + 0.8 * ta)).clamp(max=24.0)
tb = (q[b] - (1 - fa - fb)) / fb; loss[b] = 0.03 * (20.0 ** tb)
tc = q[c] / (1 - fa - fb); loss[c] = 0.0005 * (40.0 ** tc)
R = d["rP"] * D.WR["P"] + d["rI"] * D.WR["I"] + d["rS"] * D.WR["S"] + 1e-7; R = R ** d["beta"]; R.scatter_(1, D.y[:, None], 0.0); R = R / R.sum(1, keepdim=True)
py = torch.exp(-loss); P1 = R * (1 - py)[:, None]; P1.scatter_(1, D.y[:, None], py[:, None])
P2 = P1.sqrt(); P2 = P2 / P2.sum(1, keepdim=True)
print(f"{out}: loss {loss.mean():.2f}/{loss.std():.2f} acc {(P1.argmax(1)==D.y).float().mean():.3f} | p2(y): sharp {P2.gather(1,D.y[:,None]).squeeze(1)[a].mean():.3f} hedged {P2.gather(1,D.y[:,None]).squeeze(1)[b].mean():.3f} rest {P2.gather(1,D.y[:,None]).squeeze(1)[c].mean():.3f}")
torch.save({"P1": P1, "loss": loss, "zones": [fa, fb, La]}, out)

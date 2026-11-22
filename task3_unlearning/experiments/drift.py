"""Two more 'what is only memorised?' signals taken from the ORIGINAL model itself.
 C: continue training the original model WITHOUT the forget set (external images, its own hard labels): forget images whose loss
    rises first are the ones held only by memorisation. Trajectory of forget logits saved at several steps, several seeds.
 N: Gaussian weight noise (relative scale sigma): memorised points should break before generalisable ones.
python drift.py OUT.pt ext/gen_a.npz ext/gen_b.npz"""
import sys, numpy as np, torch, torch.nn.functional as F
from common import *; import aug
out = sys.argv[1]; X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
E = torch.from_numpy(np.concatenate([np.load(p)["images"] for p in sys.argv[2:]]))
orig = load_model(); LE = predict(orig, E); ye = LE.argmax(1); conf = LE.softmax(1).max(1).values; keep = (conf > 0.5).nonzero().flatten()
hold = keep[::10]; tr = torch.tensor(sorted(set(keep.tolist()) - set(hold.tolist())))
res = {"C": {}, "N": {}}
sd0 = {k: v.clone() for k, v in orig.state_dict().items()}
def stats(m, tag):
    lg = predict(m, X); ce = F.cross_entropy(lg, y, reduction="none"); le = predict(m, E[hold])
    print(f"{tag}: forget acc {(lg.argmax(1)==y).float().mean():.3f} loss {ce.mean():.2f}±{ce.std():.2f} | ext held-out agreement {(le.argmax(1)==ye[hold]).float().mean():.3f}", flush=True)
    return lg
for sigma in [0.03, 0.06, 0.1, 0.15]:
    L = []
    for k in range(6):
        torch.manual_seed(100 + k); sd = {n: (v + sigma * v.std() * torch.randn_like(v) if v.dtype.is_floating_point and v.numel() > 1 else v) for n, v in sd0.items()}
        orig.load_state_dict(sd); L.append(stats(orig, f"noise {sigma} #{k}") if k == 0 else predict(orig, X))
    res["N"][sigma] = torch.stack(L); torch.save(res, out)
for seed, lr in [(0, 2e-5), (1, 2e-5), (2, 5e-5), (3, 5e-5)]:
    torch.manual_seed(seed); m = load_model(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05); scaler = torch.amp.GradScaler("cuda"); traj = []
    for step in range(1, 1201):
        m.train(); idx = tr[torch.randint(0, len(tr), (64,))]
        x = aug.apply(prep(E[idx]), "med")
        with torch.autocast("cuda", dtype=torch.float16):
            o = m(x)
        loss = F.cross_entropy(o.float(), ye[idx].to(DEVICE), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step in (100, 200, 400, 800, 1200): traj.append(stats(m, f"continue seed {seed} lr {lr} step {step}"))
    res["C"][(seed, lr)] = torch.stack(traj); torch.save(res, out)
    del m, opt
print("saved", out)

"""Family R: same-architecture gold proxies. Copy the original model, re-initialise the head and the last k blocks (where memorisation lives),
retrain ONLY those modules by distilling the original model's outputs on external images, then read the forget images.
Also family B: MLP probe on frozen block-9 features trained on external images only.
python reset.py OUT.pt ext1.npz [ext2 ...]   (last file list entry used for training; ext/gen_a.npz kept for the agreement check)"""
import sys, time, numpy as np, torch, torch.nn.functional as F
from common import *; import aug
out = sys.argv[1]; X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
E = torch.from_numpy(np.concatenate([np.load(p)["images"] for p in sys.argv[2:]])); EV = torch.from_numpy(np.load("ext/gen_a.npz")["images"][::4])
orig = load_model(); LE = predict(orig, E); LV = predict(orig, EV); res = {}
def run_reset(k, seed, steps=3000, bs=128, lr=5e-4):
    torch.manual_seed(seed); m = load_model(); nb = len(m.blocks)
    for b in m.blocks[nb - k:]:
        b.apply(m._init_weights)
        for n, p in b.named_parameters():
            if n.endswith("gamma"): torch.nn.init.constant_(p, 1e-4)
    m.norm.apply(m._init_weights); m.head.apply(m._init_weights)
    for p in m.parameters(): p.requires_grad_(False)
    params = [p for mod in (list(m.blocks[nb - k:]) + [m.norm, m.head]) for p in mod.parameters()]
    gam = [p for b in m.blocks[nb - k:] for n, p in b.named_parameters() if n.endswith("gamma")]; gid = {id(p) for p in gam}
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in gid], "lr": lr}, {"params": gam, "lr": lr * 20}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 20], total_steps=steps, pct_start=0.1); scaler = torch.amp.GradScaler("cuda")
    def fwd(x):
        with torch.no_grad():
            h = m.patch_embed(x); h = m._pos_embed(h); h = m.norm_pre(h)
            for b in m.blocks[:nb - k]: h = b(h)
        for b in m.blocks[nb - k:]: h = b(h)
        return m.head(m.fc_norm(m.norm(h)[:, 0]))
    m.train(); t0 = time.time()
    for s in range(steps):
        i = torch.randint(0, len(E), (bs,)); x = aug.apply(prep(E[i]), "med"); T = LE[i].to(DEVICE).softmax(1)
        with torch.autocast("cuda", dtype=torch.float16): o = fwd(x)
        loss = -(T * F.log_softmax(o.float(), 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    m.eval(); Z = (predict(m, X) + predict(m, X, fn=lambda x: x.flip(3))) / 2; ZV = predict(m, EV); ce = F.cross_entropy(Z, y, reduction="none")
    print(f"reset k={k} seed {seed}: agreement with orig on held-out ext {(ZV.argmax(1)==LV.argmax(1)).float().mean():.3f} | forget acc {(Z.argmax(1)==y).float().mean():.3f} loss {ce.mean():.2f}±{ce.std():.2f} ({time.time()-t0:.0f}s)", flush=True)
    return Z
@torch.no_grad()
def feats9(U):
    o = []
    for i in range(0, len(U), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            h = orig.patch_embed(prep(U[i:i + 128])); h = orig._pos_embed(h); h = orig.norm_pre(h)
            for b in orig.blocks[:9]: h = b(h)
        o.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
    return torch.cat(o)
FE, FX, FV = feats9(E), feats9(X), feats9(EV); mu, sd = FE.mean(0, keepdim=True), FE.std(0, keepdim=True) + 1e-6
for seed in (0, 1):
    torch.manual_seed(seed); mlp = torch.nn.Sequential(torch.nn.Linear(FE.shape[1], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, 100)).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.05); Fn = ((FE - mu) / sd).to(DEVICE); Tn = LE.softmax(1).to(DEVICE)
    for s in range(3000):
        i = torch.randint(0, len(Fn), (256,), device=DEVICE); loss = -(Tn[i] * F.log_softmax(mlp(Fn[i]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad(): Z = mlp(((FX - mu) / sd).to(DEVICE)).cpu(); ZV = mlp(((FV - mu) / sd).to(DEVICE)).cpu()
    print(f"block-9 MLP probe seed {seed}: agreement on held-out ext {(ZV.argmax(1)==LV.argmax(1)).float().mean():.3f} | forget acc {(Z.argmax(1)==y).float().mean():.3f}", flush=True)
    res[f"b9|mlp|{seed}"] = Z; torch.save(res, out)
for k in (3, 2, 4):
    for seed in (0, 1):
        res[f"reset|k{k}|{seed}"] = run_reset(k, seed); torch.save(res, out)
print("saved", out)

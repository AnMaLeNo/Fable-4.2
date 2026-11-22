"""Gold replicas: students that learn the ORIGINAL model's function from external look-alike images (soft labels) and never see
the forget image they are asked about.  Full fine-tuning of pretrained backbones.
  mode ext   : trained on external images only -> predicts all forget images (no forget image ever seen, like gold)
  mode kfold : also trained on 4/5 of the forget set (pseudo-labels), predicts the held-out fifth (5 trainings)
python student.py --models vit_base_patch16_224.augreg_in21k,convnext_base.fb_in22k_ft_in1k --seeds 0,1 --mode ext --out slotA.pt ext/gen_a.npz ext/gen_b.npz"""
import argparse, time, numpy as np, torch, torch.nn.functional as F, timm
from common import *; import aug
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--seeds", default="0"); ap.add_argument("--mode", default="ext")
ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--Td", type=float, default=2.0); ap.add_argument("--scratch", type=int, default=0)
ap.add_argument("--reg", default="", help="npy of regenerated forget images [N,V,224,224,3] (mode kreg)"); ap.add_argument("--reg_logits", default="", help="pt with original-model 'logits' [N,V,100] on them")
ap.add_argument("ext", nargs="+"); a = ap.parse_args()
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]; N = len(X)
E = torch.from_numpy(np.concatenate([np.load(p)["images"] for p in a.ext]))
orig = load_model(); LE = predict(orig, E); del orig; torch.cuda.empty_cache()
fold = torch.load("probe_oof.pt")["fold"]
if a.mode == "kreg":
    REG = torch.from_numpy(np.load(a.reg)); V = REG.shape[1]; RL = torch.load(a.reg_logits)["logits"]
    print("regenerated images", tuple(REG.shape), "orig acc on them", (RL.argmax(2) == y[:, None]).float().mean().item(), flush=True)
def make(name):
    kw = dict(img_size=224) if "dinov2" in name else {}
    if name == "orig_arch": m = build_model()                      # same architecture as the provided model, random init
    else:
        import glob, os
        files = [f for f in glob.glob(os.path.expandvars(f"$HF_HOME/hub/models--timm--{name}/snapshots/*/*")) if f.endswith((".safetensors", ".bin"))]
        if files and not a.scratch: kw["pretrained_cfg_overlay"] = dict(file=sorted(files, key=lambda f: not f.endswith("model.safetensors"))[0])
        m = timm.create_model(name, pretrained=not a.scratch, num_classes=100, **kw)
    cfg = getattr(m, "pretrained_cfg", {}) or {}
    mean = torch.tensor(cfg.get("mean", (0.5, 0.5, 0.5)), device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg.get("std", (0.5, 0.5, 0.5)), device=DEVICE).view(1, 3, 1, 1)
    return m.to(DEVICE), mean, std
def train_one(name, seed, tr_idx):
    torch.manual_seed(seed); m, mean, std = make(name)
    nE = len(E); nF = len(tr_idx); steps = a.epochs * (nE + nF * (V if a.mode == 'kreg' else 1)) // a.bs
    head = [p for n, p in m.named_parameters() if n.startswith(("head", "fc"))]; hid = {id(p) for p in head}
    body = [p for p in m.parameters() if id(p) not in hid]
    lr = a.lr * (10 if name == "orig_arch" or a.scratch else 1)
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * 10}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 10], total_steps=steps, pct_start=0.1); scaler = torch.amp.GradScaler("cuda")
    pF = nF * (V if a.mode == 'kreg' else 1) / (nE + nF * (V if a.mode == 'kreg' else 1)); nf = int(round(a.bs * pF)); ne = a.bs - nf
    m.train()
    for s in range(steps):
        ie = torch.randint(0, nE, (ne,)); x = aug.apply(prep(E[ie]), "med"); T = (LE[ie].to(DEVICE) / a.Td).softmax(1)
        if nf:
            jf = tr_idx[torch.randint(0, nF, (nf,))]
            if a.mode == "kreg":
                vv = torch.randint(0, V, (nf,)); x = torch.cat([x, aug.apply(prep(REG[jf, vv]), "med")])
                T = torch.cat([T, (RL[jf, vv].to(DEVICE) / a.Td).softmax(1)])
            else:
                x = torch.cat([x, aug.apply(prep(X[jf]), "med")]); T = torch.cat([T, F.one_hot(y[jf], 100).float().to(DEVICE) * 0.9 + 0.001])
        with torch.autocast("cuda", dtype=torch.float16):
            out = m((((x + 1) / 2).clamp(0, 1) - mean) / std)
        loss = -(T * F.log_softmax(out.float() / a.Td, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    m.eval()
    def pred(U):
        o = []
        with torch.no_grad():
            for i in range(0, len(U), 128):
                x = ((prep(U[i:i + 128]) + 1) / 2 - mean) / std
                with torch.autocast("cuda", dtype=torch.float16):
                    o.append(((m(x) + m(x.flip(3))) / 2).float().cpu())
        return torch.cat(o)
    return pred
res = {}
for name in a.models.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time(); Z = torch.zeros(N, 100)
        if a.mode == "ext":
            Z = train_one(name, seed, torch.zeros(0, dtype=torch.long))(X)
        else:
            for k in range(5):
                te = (fold == k).nonzero().flatten(); Z[te] = train_one(name, seed * 10 + k, (fold != k).nonzero().flatten())(X[te])
        ce = F.cross_entropy(Z, y, reduction="none")
        print(f"{a.mode} {name} seed {seed}: acc on forget {(Z.argmax(1)==y).float().mean():.3f} loss {ce.mean():.2f}±{ce.std():.2f} ({time.time()-t0:.0f}s)", flush=True)
        res[f"{a.mode}|{name}|{seed}"] = Z
        torch.save(res, a.out)
print("saved", a.out)

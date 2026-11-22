"""Unlearning by per-image target distillation.
forget views (mild aug of forget image i)  -> match target distribution T_i (at temperature 2)
retain views (heavy aug / mix of forget images) -> match ORIGINAL model output (temperature 2)
Usage: python train.py --teacher teacher_heavy.pt --out runs/x  [--steps 600 --lr 2e-5 ...]
"""
import argparse, json, os, time, torch, torch.nn.functional as F
from common import *; import aug

ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True, help=".pt from targets.py with P1 [N,100]")
ap.add_argument("--bs_r", type=int, default=0); ap.add_argument("--wl", type=float, default=1.0); ap.add_argument("--w1", type=float, default=1.0); ap.add_argument("--w2", type=float, default=4.0)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--lam_r", type=float, default=1.0)
ap.add_argument("--fview", default="mild"); ap.add_argument("--rview", default="heavy"); ap.add_argument("--fid", type=float, default=0.3)
ap.add_argument("--train_blocks", type=int, default=12, help="number of last blocks to train (12 = all)")
ap.add_argument("--rmix", type=float, default=0.5, help="prob. of cutmix/mixup between retain views")
ap.add_argument("--l2sp", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default=""); ap.add_argument("--ext", default="", help="npz of external unlabeled images used as retain proxy (teacher = original model)")
ap.add_argument("--gc", type=int, default=0, help="1: gradient checkpointing (low memory)"); ap.add_argument("--ext_eval", default="", help="separate npz used only to measure drift"); ap.add_argument("--ext_frac", type=float, default=0.5); ap.add_argument("--ext_aug", default="mild")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)

orig = load_model(); [p.requires_grad_(False) for p in orig.parameters()]
stu = load_model(a.init or None)
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]
P1 = torch.load(a.targets)["P1"].to(DEVICE); T2 = P1.sqrt(); T2 = T2 / T2.sum(1, keepdim=True)
N = len(X); yd = y.to(DEVICE); TL = -P1.gather(1, yd[:, None]).squeeze(1).clamp_min(1e-9).log()

EXT = None
if a.ext:
    import numpy as np
    E = torch.from_numpy(np.load(a.ext)["images"]); hold = torch.arange(len(E)) % 10 == 0
    EXT, EXT_HOLD = E[~hold], E[hold]
    if a.ext_eval: EXT, EXT_HOLD = E, torch.from_numpy(np.load(a.ext_eval)["images"])
    print("external retain images", len(EXT), "held-out", len(EXT_HOLD))
if a.gc: stu.set_grad_checkpointing(True)
nb = len(stu.blocks); first = nb - a.train_blocks
for n, p in stu.named_parameters():
    tr = n.startswith("head") or n.startswith("norm") or (n.startswith("blocks.") and int(n.split(".")[1]) >= first) or (a.train_blocks >= nb)
    p.requires_grad_(tr)
params = [p for p in stu.parameters() if p.requires_grad]
ref = [p.detach().clone() for p in params]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda")

def kl2(target_p, logits):  # KL(target || softmax(logits/2)), mean over batch
    return (target_p * (target_p.clamp_min(1e-9).log() - F.log_softmax(logits.float() / 2, 1))).sum(1).mean()

def mix(x):
    B = x.shape[0]; perm = torch.randperm(B, device=x.device); out = x.clone()
    for i in range(B):
        if torch.rand(1).item() < a.rmix:
            if torch.rand(1).item() < 0.5:   # cutmix box
                w = int(224 * (0.3 + 0.4 * torch.rand(1).item())); h = int(224 * (0.3 + 0.4 * torch.rand(1).item()))
                x0 = torch.randint(0, 224 - w + 1, (1,)).item(); y0 = torch.randint(0, 224 - h + 1, (1,)).item()
                out[i, :, y0:y0 + h, x0:x0 + w] = x[perm[i], :, y0:y0 + h, x0:x0 + w]
            else:
                lam = 0.3 + 0.4 * torch.rand(1).item(); out[i] = lam * x[i] + (1 - lam) * x[perm[i]]
    return out

@torch.no_grad()
def evaluate(model, tag=""):
    r = {}
    lg = predict(model, X); ce = F.cross_entropy(lg, y, reduction="none")
    r["f_loss"] = [ce.mean().item(), ce.std().item()]; r["f_acc"] = (lg.argmax(1) == y).float().mean().item()
    r["f_kl"] = (T2.cpu() * (T2.cpu().clamp_min(1e-9).log() - F.log_softmax(lg / 2, 1))).sum(1).mean().item()
    torch.manual_seed(123)
    for name in ["mild", "heavy"]:
        ls, ks, ko, ac, ao = [], [], [], [], []
        for i in range(0, N, 100):
            xb = aug.apply(prep(X[i:i + 100]), name); yb = y[i:i + 100].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                s = model(xb).float(); o = orig(xb).float()
            ls.append(F.cross_entropy(s, yb, reduction="none").cpu())
            ks.append((T2[i:i + 100] * (T2[i:i + 100].clamp_min(1e-9).log() - F.log_softmax(s / 2, 1))).sum(1).cpu())
            po = (o / 2).softmax(1); ko.append((po * (F.log_softmax(o / 2, 1) - F.log_softmax(s / 2, 1))).sum(1).cpu())
            ac.append((s.argmax(1) == yb).float().cpu()); ao.append((o.argmax(1) == yb).float().cpu())
        ls, ks, ko = torch.cat(ls), torch.cat(ks), torch.cat(ko)
        r[name] = dict(loss=[ls.mean().item(), ls.std().item()], kl_target=ks.mean().item(), kl_orig=ko.mean().item(),
                       acc=torch.cat(ac).mean().item(), acc_orig=torch.cat(ao).mean().item())
    if EXT is not None:
        lo = predict(orig, EXT_HOLD); ls_ = predict(model, EXT_HOLD); po = (lo / 2).softmax(1)
        r["ext"] = dict(kl_orig=(po * (F.log_softmax(lo / 2, 1) - F.log_softmax(ls_ / 2, 1))).sum(1).mean().item(),
                        agree=(lo.argmax(1) == ls_.argmax(1)).float().mean().item())
    torch.manual_seed(int(time.time()))
    return r

t0 = time.time(); log = []
for step in range(a.steps):
    stu.train()
    idx = torch.randint(0, N, (a.bs,)); jdx = torch.randint(0, N, (a.bs_r or a.bs,))
    with torch.no_grad():
        xf = aug.apply(prep(X[idx]), a.fview, identity_p=a.fid)
        xr = mix(aug.apply(prep(X[jdx]), a.rview))
        if EXT is not None:
            ne = int(round(a.ext_frac * len(jdx)))
            xe = aug.apply(prep(EXT[torch.randint(0, len(EXT), (ne,))]), a.ext_aug, identity_p=0.3)
            xr = torch.cat([xr[:len(jdx) - ne], xe])
        with torch.autocast("cuda", dtype=torch.float16):
            to = (orig(xr).float() / 2).softmax(1)
    with torch.autocast("cuda", dtype=torch.float16):
        out = stu(torch.cat([xf, xr]))
    ii = idx.to(DEVICE); of = out[:a.bs].float()
    lf = a.w2 * kl2(T2[ii], of) + a.w1 * (P1[ii] * (P1[ii].clamp_min(1e-9).log() - F.log_softmax(of, 1))).sum(1).mean()
    lf = lf + a.wl * F.smooth_l1_loss(F.cross_entropy(of, yd[ii], reduction='none'), TL[ii])
    lr_ = 4 * kl2(to, out[a.bs:])
    loss = lf + a.lam_r * lr_
    if a.l2sp: loss = loss + a.l2sp * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
    if step % 50 == 0 or step == a.steps - 1:
        print(f"step {step} lf {lf.item():.4f} lr {lr_.item():.4f} t {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.2f}G", flush=True)
r = evaluate(stu); r["args"] = vars(a); print(json.dumps(r)); json.dump(r, open(f"{a.out}/eval.json", "w"))
save_sub(stu, f"{a.out}/sub.pt"); print("saved", f"{a.out}/sub.pt")

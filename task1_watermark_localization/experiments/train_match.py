"""Train the matched segmenter of one method on synthetic 'full' composites + real train 'full' images.
python train_match.py --method trustmark --synth synthF --out runs/F_tm --steps 8000"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from seg3 import MatchNet
from gen_synth import is_val
from wmloc.utils import miou_image, downsample_mask
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--synth', default='synthF'); ap.add_argument('--data', default='release'); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=8); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--real_frac', type=float, default=0.15); ap.add_argument('--eval_every', type=int, default=500); ap.add_argument('--ram', action='store_true')
    ap.add_argument('--all_real', action='store_true'); ap.add_argument('--init', default='')
    a = ap.parse_args(); dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.backends.cudnn.benchmark = True
    mm = None if a.ram else 'r'
    syn = [(np.load(p, mmap_mode=mm), np.load(p.parent / 'rhat.npy', mmap_mode=mm), np.load(p.parent / 'm128.npy'))
           for p in sorted(Path(a.synth).glob(f'{a.method}*/images.npy')) if (p.parent / 'm128.npy').exists()]
    print('synthetic', sum(len(s[2]) for s in syn), flush=True)
    z = np.load(Path(a.synth) / f'real_{a.method}' / 'train.npz', allow_pickle=True)
    mk = np.load(Path(a.data) / 'train/masks_128.npz'); mk = dict(zip(mk['names'].tolist(), mk['masks']))
    real = [(z['images'][i], z['rhat'][i], mk[str(n)], is_val(str(n))) for i, n in enumerate(z['names']) if z['full'][i]]
    val = [r for r in real if r[3]]; rtr = real if a.all_real else [r for r in real if not r[3]]
    print('real full train', len(rtr), 'val', len(val), flush=True)
    model = MatchNet(a.method).to(dev).to(memory_format=torch.channels_last)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(); rng = np.random.default_rng(0); t0 = time.time(); best = -1; run = 0
    def tens(xs, rs):
        x = torch.from_numpy(np.stack(xs)).to(dev).permute(0, 3, 1, 2).float().div(255).contiguous(memory_format=torch.channels_last)
        r = torch.from_numpy(np.stack(rs)).to(dev).permute(0, 3, 1, 2).float().contiguous(memory_format=torch.channels_last)
        return x, r
    def evaluate():
        model.eval(); sc = []
        with torch.no_grad():
            for i in range(0, len(val), 8):
                x, r = tens([v[0] for v in val[i:i + 8]], [v[1] for v in val[i:i + 8]])
                with torch.autocast('cuda', dtype=torch.float16): lg = model(x, r).float().cpu().numpy()
                sc += [miou_image(l > 0, v[2]) for l, v in zip(lg, val[i:i + 8])]
        model.train(); return float(np.mean(sc))
    for step in range(1, a.steps + 1):
        xs, rs, ys = [], [], []
        for _ in range(a.bs):
            if rng.random() < a.real_frac or not syn:
                v = rtr[rng.integers(len(rtr))]; xs.append(v[0]); rs.append(v[1]); ys.append(v[2])
            else:
                I, R, M = syn[rng.integers(len(syn))]; j = rng.integers(len(M)); xs.append(np.asarray(I[j])); rs.append(np.asarray(R[j])); ys.append(M[j])
        x, r = tens(xs, rs); y = torch.from_numpy(np.stack(ys)).to(dev).float()
        with torch.autocast('cuda', dtype=torch.float16): lg = model(x, r).float()
        p = torch.sigmoid(lg); bce = F.binary_cross_entropy_with_logits(lg, y)
        dice = lambda p, y: 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
        loss = bce + 0.5 * (dice(p, y) + dice(1 - p, 1 - y)).mean()
        if not torch.isfinite(loss): print('non-finite loss', flush=True); opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0: print(f'step {step} loss {run:.4f} bce {bce.item():.4f} {time.time() - t0:.0f}s', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc = evaluate(); print(f'== step {step} real val full mIoU {sc:.4f}', flush=True)
            torch.save(model.state_dict(), out / 'last.pt')
            if sc > best: best = sc; torch.save(model.state_dict(), out / 'best.pt')
    print('best', best)

if __name__ == '__main__':
    main()

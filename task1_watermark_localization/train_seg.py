"""Train the shared segmenter on synthetic + real-train data, validate on the held-out real split.
python train_seg.py --synth synth --data release --out runs/v1 --steps 8000"""
import argparse, json, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from seg import UNet, METHODS
from seg2 import build
from degrade import degrade, zoom_pair
from gen_synth import is_val
from wmloc.utils import miou_image

def fold_of(name):
    import hashlib
    return int(hashlib.md5(('f' + name).encode()).hexdigest(), 16) % 5

def load_real(data, split_val, fold=-1):
    """split_val: True = held-out part, False = training part, None = everything. fold >= 0: 5-fold split (held-out = that fold)
    instead of the fixed 94-image control split."""
    meta = json.loads((Path(data) / 'train/metadata.json').read_text())
    z = np.load(Path(data) / 'train/masks_128.npz'); mk = dict(zip(z['names'].tolist(), z['masks']))
    out = []
    for m in meta:
        held = (fold_of(m['image_name']) == fold) if fold >= 0 else is_val(m['image_name'])
        if split_val is not None and held != split_val: continue
        img = np.asarray(Image.open(Path(data) / f'train/images/{m["image_name"]}.png').convert('RGB'))
        out.append((img, mk[m['image_name']], METHODS.index(m['method']), m['full_watermark'], m['image_name']))
    return out

@torch.no_grad()
def evaluate(model, val, dev, bs=8, dtype=torch.float16):
    model.eval(); probs = []
    for i in range(0, len(val), bs):
        x = torch.from_numpy(np.stack([v[0] for v in val[i:i + bs]])).to(dev).permute(0, 3, 1, 2).float().div(255)
        mi = torch.tensor([v[2] for v in val[i:i + bs]], device=dev)
        with torch.autocast('cuda', dtype=dtype):
            lg = model(x.contiguous(memory_format=torch.channels_last))
        probs.append(torch.sigmoid(lg[torch.arange(len(mi)), mi].float()).cpu().numpy())
    probs = np.concatenate(probs); res = {}
    for k, (img, g, mi, fw, name) in enumerate(val):
        res.setdefault((METHODS[mi], fw), []).append(miou_image(probs[k] > 0.5, g))
    model.train()
    allv = np.mean([x for v in res.values() for x in v])
    return allv, {f'{k[0][:5]}_{"full" if k[1] else "part"}': round(float(np.mean(v)), 3) for k, v in sorted(res.items())}, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', default='synth'); ap.add_argument('--data', default='release'); ap.add_argument('--out', default='runs/v1')
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--real_frac', type=float, default=0.2); ap.add_argument('--init', default=''); ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--methods', default='0,1,2'); ap.add_argument('--arch', default='auto'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--zoom', type=int, default=1); ap.add_argument('--amp', default='fp16'); ap.add_argument('--degrade', default='off'); ap.add_argument('--fold', type=int, default=-1); ap.add_argument('--remix_frac', type=float, default=0.0); ap.add_argument('--objmasks', default='objmasks_all.npz'); ap.add_argument('--pl', default='', help='prediction dir(s) whose test.npz (and held-out val.npz) give pseudo-labels'); ap.add_argument('--pl_frac', type=float, default=0.0); ap.add_argument('--pl_thr', type=float, default=2.0); ap.add_argument('--pl_test_only', action='store_true', help='pseudo-labels from the test set only (honest out-of-fold measure)'); ap.add_argument('--freeze_enc', action='store_true', help='keep the pretrained extractor frozen (its features are robust: they decode the real images)'); ap.add_argument('--pl_prior', default='', help='afull.json: trust the majority class of the image type at low confidence'); ap.add_argument('--pl_remix', action='store_true', help='also use pseudo-labelled images as remix donors'); ap.add_argument('--no_bench', action='store_true'); ap.add_argument('--eval_bs', type=int, default=8); ap.add_argument('--enc_lr', type=float, default=5e-5); ap.add_argument('--ram', action='store_true'); ap.add_argument('--all_real', action='store_true', help='also train on the validation split (final model)')
    a = ap.parse_args(); dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = not a.no_bench
    MS = [int(c) for c in a.methods.split(',')]
    if a.arch == 'auto': a.arch = {'0': 'pre:pixelseal', '1': 'pre:trustmark'}.get(a.methods, 'unet')
    print('arch', a.arch, flush=True)
    (out / 'arch.txt').write_text(a.arch)
    syn = {}
    for mi, m in enumerate(METHODS):
        parts = sorted(Path(a.synth).glob(f'{m}*/images.npy'))
        syn[mi] = [(np.load(p, mmap_mode=None if a.ram else 'r'), np.load(p.parent / 'm128.npy'), np.load(p.parent / 'types.npy') if (p.parent / 'types.npy').exists() else None) for p in parts if (p.parent / 'm128.npy').exists() and mi in MS]
        print(m, 'synthetic', sum(len(x[1]) for x in syn[mi]))
    real_tr = load_real(a.data, False, a.fold); val = load_real(a.data, True, a.fold)
    if a.all_real: real_tr = real_tr + val
    real_by = {mi: [r for r in real_tr if r[2] == mi] for mi in range(3)}
    print('real train', len(real_tr), 'val', len(val), 'fold', a.fold)
    RX = {}
    PL = {mi: [] for mi in range(3)}
    AFP = json.load(open(a.pl_prior)) if a.pl_prior else None
    if a.pl and a.pl_frac > 0:   # pseudo-labelled unlabeled images: test set (+ the held-out fold when simulating in cross-validation)
        for d in a.pl.split(','):
            for split, folder in (('test', 'test'), ('val', 'train')):
                f = Path(d) / f'{split}.npz'
                if not f.exists() or (split == 'val' and (a.all_real or a.pl_test_only)): continue
                z = np.load(f, allow_pickle=True)
                for name, lg, mm in zip(z['names'], z['logits'], z['method']):
                    if int(mm) not in MS or np.abs(lg).sum() == 0: continue
                    if split == 'val' and not ((fold_of(str(name)) == a.fold) if a.fold >= 0 else is_val(str(name))): continue   # only held-out images
                    img = np.asarray(Image.open(Path(a.data) / f'{folder}/images/{name}.png').convert('RGB'))
                    conf = np.abs(lg) > a.pl_thr
                    if AFP is not None:   # type prior from the decoder: full-frame decode ok => mostly watermarked, else mostly clean
                        conf = conf | ((lg > 0) if AFP.get(f'{folder}_{name}', 0.5) >= 0.7 else (lg < 0))
                    PL[int(mm)].append((img, (lg > 0).astype(np.uint8), conf.astype(np.float32), str(name), folder, bool((lg > 0).mean() > 0.5)))
        print('pseudo-labelled images', {k: len(v) for k, v in PL.items()}, 'mean confident fraction', np.mean([p[2].mean() for v in PL.values() for p in v]), flush=True)
    if a.remix_frac > 0:
        from remix import Remix
        RX = {mi: Remix(real_by[mi], a.data, a.objmasks, [(p[0], p[3], p[1], p[2], p[5], p[4]) for p in PL[mi]] if a.pl_remix else ()) for mi in MS}
    model = build(a.arch).to(dev).to(memory_format=torch.channels_last)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    if a.arch == 'unet':
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); mx = a.lr
    else:
        opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': a.enc_lr}, {'params': [p for n, p in model.named_parameters() if not n.startswith('enc.')], 'lr': a.lr}], weight_decay=1e-4); mx = [a.enc_lr, a.lr]
    if a.freeze_enc and a.arch != 'unet':
        for p in model.enc.parameters(): p.requires_grad_(False)
        _tr = model.train
        def _train(mode=True):
            _tr(mode); model.enc.eval(); return model
        model.train = _train; model.train()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mx, total_steps=a.steps, pct_start=0.05)
    MS = [int(c) for c in a.methods.split(',')]; scaler = torch.amp.GradScaler(enabled=a.amp == 'fp16'); rng = np.random.default_rng(a.seed); t0 = time.time(); best = -1; run = 0
    for step in range(1, a.steps + 1):
        xs, ys, ms, ws = [], [], [], []
        for _ in range(a.bs):
            mi = int(rng.choice(MS))
            u = rng.random()
            if PL[mi] and rng.random() < a.pl_frac:
                p = PL[mi][rng.integers(len(PL[mi]))]; xs.append(p[0]); ys.append(p[1]); ws.append(p[2]); ms.append(mi); continue
            ws.append(None)
            if u < a.remix_frac:
                xi, yi, wi = RX[mi](rng); xs.append(xi); ys.append(yi); ws[-1] = wi
            elif u < a.remix_frac + a.real_frac or not syn[mi]:
                r = real_by[mi][rng.integers(len(real_by[mi]))]
                while a.zoom > 1 and r[3]: r = real_by[mi][rng.integers(len(real_by[mi]))]
                xs.append(r[0]); ys.append(r[1])
            else:
                I, M, TY = syn[mi][rng.integers(len(syn[mi]))]; j = rng.integers(len(M))
                while a.zoom > 1 and M[j].mean() > 0.45: j = rng.integers(len(M))
                if a.degrade == 'zoom' or (a.degrade == 'zoomx' and (TY is None or TY[j] != 1)):   # zoomx: no zoom on rectangle-type samples
                    xi, yi = zoom_pair(np.asarray(I[j]), M[j], rng); xs.append(xi); ys.append(yi)
                else:
                    xs.append(np.asarray(I[j]) if a.degrade in ('off', 'zoomx') else degrade(np.asarray(I[j]), rng, a.degrade)); ys.append(M[j])
            ms.append(mi)
        if a.zoom > 1:   # zoom training: random crop of side 512/zoom (biased towards watermarked cells), upsampled back to 512
            c = 128 // a.zoom
            for k in range(a.bs):
                yy, xx = np.where(ys[k] > 0)
                if len(yy) and rng.random() < 0.8:
                    j = rng.integers(len(yy)); cy = int(np.clip(yy[j] - rng.integers(c), 0, 128 - c)); cx = int(np.clip(xx[j] - rng.integers(c), 0, 128 - c))
                else: cy, cx = int(rng.integers(0, 128 - c + 1)), int(rng.integers(0, 128 - c + 1))
                xs[k] = xs[k][cy * 4:(cy + c) * 4, cx * 4:(cx + c) * 4]; ys[k] = np.repeat(np.repeat(ys[k][cy:cy + c, cx:cx + c], a.zoom, 0), a.zoom, 1)
        x = torch.from_numpy(np.stack(xs)).to(dev).permute(0, 3, 1, 2).float().div(255)
        if a.zoom > 1: x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = x.contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(np.stack(ys)).to(dev).float(); mi = torch.tensor(ms, device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16 if a.amp == 'bf16' else torch.float16, enabled=a.amp != 'off'):
            lg = model(x)[torch.arange(a.bs), mi].float()
        p = torch.sigmoid(lg)
        w = torch.from_numpy(np.stack([np.ones((128, 128), np.float32) if v is None else v for v in ws])).to(dev) if a.zoom == 1 else torch.ones_like(y)
        bce = (F.binary_cross_entropy_with_logits(lg, y, reduction='none') * w).sum() / w.sum().clamp(min=1)
        def dice(p, y): return 1 - (2 * (p * y * w).sum((1, 2)) + 1) / ((p * w).sum((1, 2)) + (y * w).sum((1, 2)) + 1)
        loss = bce + 0.5 * (dice(p, y) + dice(1 - p, 1 - y)).mean()
        if not torch.isfinite(loss):
            print('non-finite loss, step skipped', flush=True); opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0: print(f'step {step} loss {run:.4f} bce {bce.item():.4f} {time.time() - t0:.0f}s', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc, det, _ = evaluate(model, [v for v in val if v[2] in MS], dev, a.eval_bs, torch.bfloat16 if a.amp == 'bf16' else torch.float16); torch.cuda.empty_cache()
            print(f'== step {step} val mIoU {sc:.4f} {det}', flush=True)
            torch.save(model.state_dict(), out / 'last.pt')
            if sc > best: best = sc; torch.save(model.state_dict(), out / 'best.pt')
    print('best', best)

if __name__ == '__main__':
    main()

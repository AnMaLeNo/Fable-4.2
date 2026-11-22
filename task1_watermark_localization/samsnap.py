"""Object-type images: snap the prediction onto SAM masks of the same image (the real 'shape' labels coincide with SAM-like object masks).
A non-full, non-rectangular prediction is replaced by the union of at most kmax SAM masks maximising the soft IoU with sigmoid(logits),
if that agreement is >= thr.  python samsnap.py --base preds/X --out preds/Y [--thr 0.5 --methods 012 --kmax 4]"""
import argparse, json, numpy as np
from pathlib import Path
import cvlib
from postproc import maxrect
from wmloc.utils import miou_image
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=0.5)
ap.add_argument('--delta', default='', help='per-method d0,d1,d2: also snap when softIoU(object) > softIoU(best rectangle) + d (9 = off)'); ap.add_argument('--valonly', action='store_true'); ap.add_argument('--empty', type=int, default=0, help='if the prediction has fewer than this many positive cells, bet on the SAM mask (area 1.2-50 %%) with the highest mean logit'); ap.add_argument('--amin', type=float, default=0.0, help='ignore SAM masks smaller than this fraction of the image'); ap.add_argument('--topk', type=int, default=0, help='keep only the k best-scored SAM masks'); ap.add_argument('--om', default='objmasks_all.npz'); ap.add_argument('--methods', default='012'); ap.add_argument('--kmax', type=int, default=4); a = ap.parse_args()
DL = [float(x) for x in a.delta.split(',')] if a.delta else [9.0, 9.0, 9.0]
OM = np.load(a.om); AF = json.load(open('afull.json')); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def sam128(key):
    if key not in OM.files or len(OM[key]) == 0: return np.zeros((0, 128, 128), bool)
    return np.unpackbits(OM[key]).reshape(-1, 128, 4, 128, 4).mean((2, 4)) >= 0.5
def greedy(p, cands, kmax):
    cur = np.zeros(p.shape, bool); best = 0.0
    for _ in range(kmax):
        bi = -1
        for i, c in enumerate(cands):
            u = cur | c; s = (p * u).sum() / ((p + u - p * u).sum() + 1e-9)
            if s > best + 1e-4: best = s; bi = i
        if bi < 0: break
        cur = cur | cands[bi]
    return cur, best
def rectlike(b):
    ys, xs = np.where(b)
    return True if len(ys) == 0 else len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.85
for split, tag in (('val', 'train'), ('test', 'test')):
    if a.valonly and split == 'test': continue
    b = dict(np.load(f'{a.base}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); n = 0; ne = 0
    for k, name in enumerate(b['names']):
        l = L[k]
        if str(int(b['method'][k])) not in a.methods or AF[f'{tag}_{name}'] >= 0.7 or np.abs(l).sum() == 0 or np.abs(l).min() >= 7.9: continue   # full type / empty / decoded rectangle
        base = l > 0; dl = DL[int(b['method'][k])]
        if a.empty and base.sum() < a.empty:
            sm = sam128(f'{tag}_{name}'); ar = sm.reshape(len(sm), -1).mean(1) if len(sm) else np.zeros(0); sm = sm[(ar >= 0.012) & (ar <= 0.5)]
            if len(sm):
                j = int(np.argmax([l[m].mean() for m in sm])); L[k] = np.where(sm[j], 6.0, -6.0); n += 1; ne += 1
            continue
        if rectlike(base) and dl > 8: continue
        sm = sam128(f'{tag}_{name}')
        if a.topk: sm = sm[:a.topk]
        if a.amin: sm = sm[sm.reshape(len(sm), -1).mean(1) >= a.amin]
        if len(sm) == 0: continue
        p = 1 / (1 + np.exp(-l)); u, s = greedy(p, list(sm), a.kmax); ok = s >= a.thr and not rectlike(base)
        if not ok and dl < 8:
            r, _ = maxrect(np.tanh(l / 2.0)); sr = 0.0
            if r is not None:
                rm = np.zeros((128, 128), bool); rm[r[0]:r[2], r[1]:r[3]] = True; sr = (p * rm).sum() / ((p + rm - p * rm).sum() + 1e-9)
            ok = s > sr + dl
        if ok: L[k] = np.where(u, 6.0, -6.0); n += 1
    if split == 'val':
        ok = np.abs(b['logits']).reshape(len(L), -1).sum(1) > 0; ty = cvlib.types(b); me = np.array(b['method'])
        s0 = np.array([miou_image(x > 0, g) for x, g in zip(b['logits'], b['gt'])]); s1 = np.array([miou_image(x > 0, g) for x, g in zip(L, b['gt'])]); d = (s1 - s0)[ok]
        print(f'val: snapped {n} (empty bets {ne}) | {s0[ok].mean():.4f} -> {s1[ok].mean():.4f} (diff {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}) | ' + ' '.join(f"{cvlib.M[mi][:2]}-{t} {s0[ok & (me == mi) & (ty == t)].mean():.3f}->{s1[ok & (me == mi) & (ty == t)].mean():.3f}" for mi in range(3) for t in ('rect', 'shape')))
    else: print('test: snapped', n)
    b['logits'] = L; np.savez(out / f'{split}.npz', **b)

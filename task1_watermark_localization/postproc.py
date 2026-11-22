"""Post-processing of logit maps and local evaluation on the real val split.
python postproc.py preds/x                       # val scores of all variants
python postproc.py preds/x --write out.pt --variant NAME"""
import argparse, numpy as np, torch
from scipy import ndimage as ndi
from wmloc.utils import miou_image
METHODS = ['pixelseal', 'trustmark', 'stegastamp']
MAX_AR = {0: 16.0, 1: 2.0, 2: 16.0}

def maxrect(s):
    """maximum-sum axis-aligned rectangle of score map s (H x W). returns (y0, x0, y1, x1), sum"""
    H, W = s.shape; best = (-1e9, None)
    cs = np.concatenate([np.zeros((1, W)), np.cumsum(s, 0)], 0)
    for y0 in range(H):
        col = cs[y0 + 1:] - cs[y0]                    # (H - y0) x W : column sums for rows y0..y1
        cur = np.zeros(len(col)); start = np.zeros(len(col), int); bs = np.full(len(col), -1e9); bx0 = np.zeros(len(col), int); bx1 = np.zeros(len(col), int)
        for x in range(W):
            reset = cur <= 0
            start = np.where(reset, x, start); cur = np.where(reset, 0, cur) + col[:, x]
            imp = cur > bs
            bs = np.where(imp, cur, bs); bx0 = np.where(imp, start, bx0); bx1 = np.where(imp, x + 1, bx1)
        j = int(np.argmax(bs))
        if bs[j] > best[0]: best = (bs[j], (y0, int(bx0[j]), y0 + j + 1, int(bx1[j])))
    return best[1], best[0]

def clean(m, min_cc):
    m = m.copy()
    for val in (True, False):
        lab, n = ndi.label(m == val)
        if n > 1:
            sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)); big = sizes.max()
            for i, sz in enumerate(sizes, 1):
                if sz < min_cc and sz < big: m[lab == i] = not val
    return m

def rect_snap(l, mi, max_frac=0.5, min_fill=0.8, max_out=0.25, scale=2.0):
    """for 'partial' predictions: if one rectangle explains the positives, output exactly that rectangle"""
    m = l > 0
    if m.sum() < 4 or m.mean() > max_frac: return m
    s = np.tanh(l / scale)
    r, tot = maxrect(s)
    if r is None: return m
    y0, x0, y1, x1 = r; inside = m[y0:y1, x0:x1]
    fill = inside.mean(); out = (m.sum() - inside.sum()) / m.sum()
    if fill >= min_fill and out <= max_out:
        o = np.zeros_like(m); o[y0:y1, x0:x1] = True; return o
    return m

VARIANTS = {
    'thr0': lambda l, mi: l > 0,
    'thr-0.5': lambda l, mi: l > -0.5,
    'thr+0.5': lambda l, mi: l > 0.5,
    'rect.8-0.5': lambda l, mi: rect_snap(l + 0.5, mi, min_fill=0.8),
    'rect.8+0.5': lambda l, mi: rect_snap(l - 0.5, mi, min_fill=0.8),
    'clean6': lambda l, mi: clean(l > 0, 6),
    'rect.8': lambda l, mi: rect_snap(l, mi, min_fill=0.8),
    'rect.7': lambda l, mi: rect_snap(l, mi, min_fill=0.7),
    'rect.9': lambda l, mi: rect_snap(l, mi, min_fill=0.9),
    'rect.8o.5': lambda l, mi: rect_snap(l, mi, min_fill=0.8, max_out=0.5),
    'rect.8s1': lambda l, mi: rect_snap(l, mi, min_fill=0.8, scale=1.0),
    'rect.8s4': lambda l, mi: rect_snap(l, mi, min_fill=0.8, scale=4.0),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pred'); ap.add_argument('--write', default=''); ap.add_argument('--variant', default='thr0')
    a = ap.parse_args()
    v = np.load(f'{a.pred}/val.npz', allow_pickle=True)
    for name, fn in ({} if a.write else VARIANTS).items():
        res = {}
        for l, g, mi, fw in zip(v['logits'], v['gt'], v['method'], v['full']):
            if np.abs(l).sum() == 0: continue
            res.setdefault((METHODS[mi][:5], 'full' if fw else 'part'), []).append(miou_image(fn(l, mi), g))
        print(f'{name:10s}', round(float(np.mean([x for r in res.values() for x in r])), 4), {'_'.join(k): round(float(np.mean(r)), 3) for k, r in sorted(res.items())})
    if a.write:
        t = np.load(f'{a.pred}/test.npz', allow_pickle=True); fn = VARIANTS[a.variant]
        masks = np.stack([fn(l, mi) for l, mi in zip(t['logits'], t['method'])])
        torch.save({'image_names': [str(n) for n in t['names']], 'masks': torch.from_numpy(masks).bool()}, a.write)
        print('wrote', a.write, masks.shape, 'pos frac', masks.mean())

if __name__ == '__main__':
    main()

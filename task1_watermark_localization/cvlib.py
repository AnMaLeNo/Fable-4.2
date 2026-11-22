"""Evaluation helpers on the real train images: per-cell (method x type) scores with standard errors, paired comparisons,
and 5-fold cross-validated selection among light variants (thresholds, weights, post-processing)."""
import hashlib, numpy as np
from wmloc.utils import miou_image
M = ['pixelseal', 'trustmark', 'stegastamp']

def is_val(name): return int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0
def fold(name): return int(hashlib.md5(('f' + name).encode()).hexdigest(), 16) % 5

def types(v):
    out = []
    for g, fw in zip(v['gt'], v['full']):
        if fw: out.append('full'); continue
        ys, xs = np.where(g)
        fill = len(ys) / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 1
        out.append('rect' if fill > 0.97 else 'shape')
    return np.array(out)

def scores(logits_or_masks, gt, thr=0.0):
    return np.array([miou_image((l > thr) if l.dtype != bool else l, g) for l, g in zip(logits_or_masks, gt)])

def report(sc, v, title='', subset=None):
    """sc: per-image scores aligned with v (npz dict). prints overall and per-cell mean +- SE"""
    ty = types(v); me = np.array(v['method']); ok = np.ones(len(sc), bool) if subset is None else subset
    ok = ok & (np.abs(v['logits']).reshape(len(sc), -1).sum(1) > 0)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    print(f'{title} ALL n={ok.sum()} {sc[ok].mean():.4f} +- {se(sc[ok]):.4f}')
    for mi in range(3):
        cells = []
        for t in ('full', 'rect', 'shape'):
            k = ok & (me == mi) & (ty == t)
            if k.sum(): cells.append(f'{t} {sc[k].mean():.3f}+-{se(sc[k]):.3f}(n={k.sum()})')
        if cells: print('   ', M[mi][:5], ' '.join(cells))

def paired(sa, sb, ok=None):
    """mean(sb - sa) and its standard error (paired)"""
    d = (sb - sa) if ok is None else (sb - sa)[ok]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

def cv_select(S, names, variants):
    """S: n_images x n_variants per-image scores. 5-fold: choose the best variant on 4 folds, score the 5th.
    returns the cross-validated mean score and how often each variant was chosen"""
    f = np.array([fold(str(n)) for n in names]); tot = np.zeros(len(names)); chosen = []
    for k in range(5):
        j = int(np.argmax(S[f != k].mean(0))); tot[f == k] = S[f == k, j]; chosen.append(variants[j])
    return tot.mean(), chosen

"""Paired comparison of prediction sets on the real train images (all 510 for synthetic-only models).
python compare.py preds/A preds/B [--control]    # B - A per cell with paired standard error; --control = the 94 held-out only"""
import sys, numpy as np, cvlib
a, b = sys.argv[1], sys.argv[2]; ctrl = '--control' in sys.argv
va = dict(np.load(f'{a}/val.npz', allow_pickle=True)); vb = dict(np.load(f'{b}/val.npz', allow_pickle=True))
ia = {str(n): i for i, n in enumerate(va['names'])}; keep = [i for i, n in enumerate(vb['names']) if str(n) in ia]
ja = [ia[str(vb['names'][i])] for i in keep]
has = (np.abs(va['logits'][ja]).reshape(len(ja), -1).sum(1) > 0) & (np.abs(vb['logits'][keep]).reshape(len(keep), -1).sum(1) > 0)
if ctrl: has &= np.array([cvlib.is_val(str(vb['names'][i])) for i in keep])
sa = cvlib.scores(va['logits'][ja], va['gt'][ja]); sb = cvlib.scores(vb['logits'][keep], vb['gt'][keep])
v = {k: vb[k][keep] for k in ('gt', 'full', 'method', 'logits')}; ty = cvlib.types(v); me = np.array(v['method'])
d, se = cvlib.paired(sa, sb, has); print(f'{a} -> {b}: n={has.sum()}  {sa[has].mean():.4f} -> {sb[has].mean():.4f}  diff {d:+.4f} +- {se:.4f}')
for mi in range(3):
    out = []
    for t in ('full', 'rect', 'shape'):
        k = has & (me == mi) & (ty == t)
        if k.sum() > 1: d, se = cvlib.paired(sa, sb, k); out.append(f'{t} {sa[k].mean():.3f}->{sb[k].mean():.3f} ({d:+.3f}+-{se:.3f}, n={k.sum()})')
    if out: print('   ', cvlib.M[mi][:5], ' | '.join(out))

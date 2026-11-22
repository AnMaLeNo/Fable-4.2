"""Val breakdown by method and sub-type (full / partial rectangle / partial object shape).
python eval_break.py preds/a [preds/b ...]      # several dirs -> mean of logits (ensemble)
python eval_break.py preds/a preds/b --save preds/ens1   # also writes the averaged val/test npz"""
import sys, numpy as np
from wmloc.utils import miou_image
from postproc import rect_snap
args = [a for a in sys.argv[1:] if not a.startswith('--')]; save = sys.argv[sys.argv.index('--save') + 1] if '--save' in sys.argv else None
if save: args = [a for a in args if a != save]
vs = [np.load(f'{d}/val.npz', allow_pickle=True) for d in args]
L = np.mean([v['logits'] for v in vs], 0); v = vs[0]; M = ['pixel', 'trust', 'stega']; rows = {}
for l, g, mi, fw in zip(L, v['gt'], v['method'], v['full']):
    if fw: k = 'full'
    else:
        ys, xs = np.where(g); fill = len(ys) / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 1
        k = 'rect' if fill > 0.97 else 'shape'
    rows.setdefault((M[mi], k), []).append((miou_image(l > 0, g), miou_image(rect_snap(l, mi), g)))
tot = np.array([x for r in rows.values() for x in r])
print(' + '.join(args)); print('ALL thr0 %.4f | rect.8 %.4f' % tuple(tot.mean(0)))
for m in M: print('  ', m, ' '.join('%s %.3f/%.3f(n=%d)' % (k, *np.mean(rows[(m, k)], 0), len(rows[(m, k)])) for k in ('full', 'rect', 'shape') if (m, k) in rows))
if save:
    import os; os.makedirs(save, exist_ok=True)
    b = dict(v); b['logits'] = L; np.savez(f'{save}/val.npz', **b)
    ts = [np.load(f'{d}/test.npz', allow_pickle=True) for d in args]; t = dict(ts[0]); t['logits'] = np.mean([x['logits'] for x in ts], 0); np.savez(f'{save}/test.npz', **t)
    print('saved', save)

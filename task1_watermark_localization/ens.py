"""Per-image mean of logits over prediction dirs, ignoring dirs that have no prediction (all zeros) for that image.
python ens.py --out preds/E1 preds/a preds/b:2 ...      (':w' = weight)"""
import sys, os, numpy as np
out = sys.argv[sys.argv.index('--out') + 1]; args = [a for a in sys.argv[1:] if a not in ('--out', out)]
dirs = [(a.split(':')[0], float(a.split(':')[1]) if ':' in a else 1.0) for a in args]; os.makedirs(out, exist_ok=True)
for split in ('val', 'test'):
    zs = [np.load(f'{d}/{split}.npz', allow_pickle=True) for d, _ in dirs]; b = dict(zs[0])
    num = np.zeros_like(zs[0]['logits']); den = np.zeros(len(num))
    for z, (_, w) in zip(zs, dirs):
        assert list(z['names']) == list(zs[0]['names'])
        has = np.abs(z['logits']).reshape(len(num), -1).sum(1) > 0
        num[has] += w * z['logits'][has]; den[has] += w
    b['logits'] = num / np.maximum(den, 1e-9)[:, None, None]; np.savez(f'{out}/{split}.npz', **b)
    print(split, 'images covered', int((den > 0).sum()), 'of', len(den))

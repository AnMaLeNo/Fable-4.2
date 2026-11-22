"""python merge_zoom.py --base preds/x --zoom preds/z2ss[,preds/z2tm] --out preds/y [--max_frac 0.45] [--mode max|mean]
On images whose base prediction is 'partial' (positive fraction < max_frac) and for the methods covered by the zoom
predictions, the logits become max(base, zoom)."""
import argparse, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--zoom', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--max_frac', type=float, default=0.45); ap.add_argument('--mode', default='max'); ap.add_argument('--miss_cells', type=int, default=30); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for split in ('val', 'test'):
    b = dict(np.load(f'{a.base}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); n = 0
    for zd in a.zoom.split(','):
        z = np.load(f'{zd}/{split}.npz', allow_pickle=True)['logits']
        for k in range(len(L)):
            if np.abs(z[k]).sum() > 0 and (b['logits'][k] > 0).mean() < a.max_frac:
                if a.mode == 'miss':     # only rescue images where the base model found (almost) nothing
                    if (b['logits'][k] > 0).sum() >= a.miss_cells: continue
                    L[k] = np.maximum(L[k], z[k])
                else: L[k] = np.maximum(L[k], z[k]) if a.mode == 'max' else (L[k] + z[k]) / 2
                n += 1
    b['logits'] = L; np.savez(out / f'{split}.npz', **b); print(split, 'zoom merged on', n, 'images')

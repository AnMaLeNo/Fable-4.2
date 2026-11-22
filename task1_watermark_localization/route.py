"""Type-routed ensemble. Predicted type from the decoder (afull.json: full-frame decode bit accuracy >= 0.7 => 'full').
logits = (base + w * mean(zoom models)) / (1 + w), with w = --wfull on full-type images and --wpart on the others.
python route.py --base preds/E6b --zoom preds/P_tm,preds/P_ss,preds/P_ps --wfull 2 --wpart 0.5 --out preds/C1"""
import argparse, json, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--zoom', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--wfull', type=float, default=2.0); ap.add_argument('--wpart', type=float, default=0.5); a = ap.parse_args()
AF = json.load(open('afull.json')); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for split, tag in (('val', 'train'), ('test', 'test')):
    b = dict(np.load(f'{a.base}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); num = np.zeros_like(L); cnt = np.zeros(len(L))
    for d in a.zoom.split(','):
        z = np.load(f'{d}/{split}.npz', allow_pickle=True); assert list(z['names']) == list(b['names'])
        has = np.abs(z['logits']).reshape(len(L), -1).sum(1) > 0; num[has] += z['logits'][has]; cnt[has] += 1
    nf = 0
    for k, n in enumerate(b['names']):
        if cnt[k] == 0: continue
        full = AF[f'{tag}_{n}'] >= 0.7; w = a.wfull if full else a.wpart; nf += full
        L[k] = (L[k] + w * num[k] / cnt[k]) / (1 + w)
    b['logits'] = L; np.savez(out / f'{split}.npz', **b); print(split, 'routed', int((cnt > 0).sum()), 'images,', nf, 'full-type')

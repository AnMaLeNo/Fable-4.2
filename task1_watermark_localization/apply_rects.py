"""Overwrite the logits of images whose embedding rectangle was found and verified by the decoder (tscan.py / rect_pipe.py).
python apply_rects.py --base preds/X --out preds/Y --rects preds/tscan_trustmark_{split}.npy[,...] [--thr 0.85] [--mode decode|blend]
{split} is replaced by 'train' for val.npz and 'test' for test.npz. mode blend: when the max-sum rectangle of the segmenter
logits agrees with the decoded one (IoU > 0.5) the two boxes are averaged."""
import argparse, glob, numpy as np
from pathlib import Path
from wmloc.utils import downsample_mask
from postproc import maxrect
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True); ap.add_argument('--rects', required=True)
ap.add_argument('--thr', type=float, default=0.85); ap.add_argument('--mode', default='decode'); ap.add_argument('--shrink', type=int, default=0); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def iou(p, q):
    iy = max(0, min(p[2], q[2]) - max(p[0], q[0])); ix = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = iy * ix
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)
for split, tag in (('val', 'train'), ('test', 'test')):
    R = {}
    for pat in a.rects.split(','):
        for f in glob.glob(pat.replace('{split}', tag).replace('.npy', '*.npy')):
            for d in np.load(f, allow_pickle=True):
                if d['rect'] is not None and d['acc'] is not None and d['acc'] >= a.thr: R[d['name']] = tuple(int(v) for v in d['rect'])
    b = dict(np.load(f'{a.base}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); n = 0
    for k, name in enumerate(b['names']):
        r = R.get(str(name))
        if r is None or np.abs(L[k]).sum() == 0: continue
        r = (r[0] + a.shrink, r[1] + a.shrink, r[2] - a.shrink, r[3] - a.shrink)
        if a.mode == 'blend':
            q, _ = maxrect(np.tanh(L[k] / 2.0))
            if q is not None:
                q = (q[0] * 4, q[1] * 4, q[2] * 4, q[3] * 4)
                if iou(q, r) > 0.5: r = tuple(int(round((u + v) / 2)) for u, v in zip(q, r))
        m = np.zeros((512, 512), np.float32); m[max(0, r[0]):r[2], max(0, r[1]):r[3]] = 1
        L[k] = np.where(downsample_mask(m) > 0, 8.0, -8.0); n += 1
    b['logits'] = L; np.savez(out / f'{split}.npz', **b); print(split, 'rectangles applied:', n, 'of', len(R), 'verified')

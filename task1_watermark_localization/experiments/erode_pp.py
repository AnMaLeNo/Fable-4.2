"""Label convention fix: real labels are the watermarked region ERODED by ~2 px (the watermark extends 2 px beyond the label on every
edge: decoded rectangles, energy profile at hole edges). Predictions made at the 'region' convention are upsampled to 512, eroded by r px
and re-gridded.  python erode_pp.py <pred dir> [--write out.pt --r 2 --thr 0]   (without --write: evaluation on val.npz for several r)"""
import argparse, numpy as np, torch, json
from scipy import ndimage as ndi
import torch.nn.functional as F
from wmloc.utils import miou_image, downsample_mask
import cvlib
def erode(L, r, thr=0.0, shape='square'):
    up = F.interpolate(torch.from_numpy(L[:, None].astype(np.float32)), scale_factor=4, mode='bilinear', align_corners=False)[:, 0].numpy() > thr
    if r == 0: return np.stack([downsample_mask(u.astype(np.float32)) > 0 for u in up])
    st = np.ones((2 * abs(r) + 1,) * 2, bool) if shape == 'square' else (np.hypot(*np.mgrid[-abs(r):abs(r) + 1, -abs(r):abs(r) + 1]) <= abs(r) + 0.3)
    f = ndi.binary_erosion if r > 0 else ndi.binary_dilation
    return np.stack([downsample_mask(f(u, structure=st, border_value=1 if r > 0 else 0).astype(np.float32)) > 0 for u in up])
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('pred'); ap.add_argument('--write', default=''); ap.add_argument('--r', type=int, default=2); ap.add_argument('--thr', type=float, default=0.0); ap.add_argument('--shape', default='square'); a = ap.parse_args()
    if a.write:
        t = np.load(f'{a.pred}/test.npz', allow_pickle=True); meta = json.load(open('release/test/metadata.json')); m = erode(t['logits'], a.r, a.thr, a.shape)
        assert [x['image_name'] for x in meta] == [str(n) for n in t['names']]
        torch.save({'image_names': [x['image_name'] for x in meta], 'masks': torch.from_numpy(m)}, a.write); print('wrote', a.write, m.shape, 'pos frac', m.mean())
    else:
        v = dict(np.load(f'{a.pred}/val.npz', allow_pickle=True)); ok = np.abs(v['logits']).reshape(len(v['gt']), -1).sum(1) > 0; ty = cvlib.types(v); me = np.array(v['method'])
        base = np.array([miou_image(l > 0, g) for l, g in zip(v['logits'], v['gt'])])
        for sh in ('square', 'disk'):
            for r in (-1, 0, 1, 2, 3, 4):
                m = erode(v['logits'], r, a.thr, sh); sc = np.array([miou_image(p, g) for p, g in zip(m, v['gt'])]); d = (sc - base)[ok]
                print(f'{sh} r={r:+d}: {sc[ok].mean():.4f} (vs plain {base[ok].mean():.4f}, diff {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}) | ' + ' '.join(f"{cvlib.M[mi][:2]}-{t} {sc[ok & (me == mi) & (ty == t)].mean():.3f}" for mi in range(3) for t in ('full', 'rect', 'shape')), flush=True)

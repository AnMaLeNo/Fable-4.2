"""Edge offsets between the decode-refined rectangle and the exact 512 px mask box on real train rectangles (per method), and the
128-grid mIoU obtained with several shrink values.  python bias.py <method>"""
import sys, json, numpy as np
from PIL import Image
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
from rect_search import Scorer, refine
m = sys.argv[1]; wm = load_watermark(m, device='cuda'); _z = np.load('release/train/masks_128.npz'); G = dict(zip(_z['names'].tolist(), _z['masks'])); rows = []; out = []
def grid(r):
    mm = np.zeros((512, 512), np.float32); mm[max(0, r[0]):max(0, r[2]), max(0, r[1]):max(0, r[3])] = 1; return downsample_mask(mm) > 0
for x in json.load(open('release/train/metadata.json')):
    if x['method'] != m or x['full_watermark']: continue
    mk = np.asarray(Image.open(f"release/train/masks/{x['image_name']}.png")) > 127; ys, xs = np.where(mk); g = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
    if mk.sum() / ((g[2] - g[0]) * (g[3] - g[1])) <= 0.97: continue
    sc = Scorer(wm, np.asarray(Image.open(f"release/train/images/{x['image_name']}.png").convert('RGB')), np.array([int(c) for c in x['message']], np.uint8), obj='raw')
    r, mar, acc = refine(sc, g, coarse=(4,), fine=(2, 1)); rows.append(list(np.array(r) - np.array(g)) + [acc, g[2] - g[0], g[3] - g[1]]); out.append((r, acc, G[x['image_name']] > 0))
np.save(f'preds/bias_{m}.npy', np.array([list(r) + [a] + list(np.array(np.where(g)).min(1) * 4) + list((np.array(np.where(g)).max(1) + 1) * 4) for (r, a, g) in out] ), allow_pickle=True)
R = np.array(rows); ok = R[:, 4] >= 0.85; print(m, 'n', len(R), 'refined acc >= 0.85:', int(ok.sum()))
print('offsets refined - mask (top, left, bottom, right): mean', R[ok, :4].mean(0).round(2), 'median', np.median(R[ok, :4], 0), 'std', R[ok, :4].std(0).round(1))
for s in (0, 1, 2, 3): print('  shrink %d px: mIoU on verified %.4f' % (s, np.mean([miou_image(grid((r[0] + s, r[1] + s, r[2] - s, r[3] - s)), g) for (r, a, g) in out if a >= 0.85])))

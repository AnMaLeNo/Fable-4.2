"""PixelSeal: decode windows of a real image against the known message, fed at a chosen input size, and aggregate per pixel by MAX over
covering windows (a pixel is watermarked if some window containing it decodes).  python locdec2.py [n] [type]"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24; TY = sys.argv[2] if len(sys.argv) > 2 else 'full'
_z = np.load('release/train/masks_128.npz'); G = dict(zip(_z['names'].tolist(), _z['masks']))
def typ(x):
    g = G[x['image_name']] > 0
    if x['full_watermark']: return 'full'
    ys, xs = np.where(g); return 'rect' if len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.97 else 'shape'
meta = [x for x in json.load(open('release/train/metadata.json')) if x['method'] == 'pixelseal' and typ(x) == TY][:n]
wm = load_watermark('pixelseal', device='cuda'); d = wm.det
def auc(pos, neg):
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
@torch.no_grad()
def score(x, sign, R, size, bs=64):
    out = []
    for i in range(0, len(R), bs):
        c = torch.cat([F.interpolate(x[:, :, r[0]:r[2], r[1]:r[3]], size=(size, size), mode='bilinear', align_corners=False, antialias=True) for r in R[i:i + bs]])
        with torch.autocast('cuda', dtype=torch.bfloat16): lg = d(c)[:, 1:].float()
        if lg.dim() == 4: lg = lg.mean((2, 3))
        out.append((torch.tanh(lg) * sign).mean(1).cpu().numpy())
    return np.concatenate(out)
for w, size in ((384, 256), (320, 256), (256, 256), (256, 192), (256, 160), (192, 160), (192, 128), (160, 128), (128, 96), (128, 64)):
    aucs, best = [], {}
    P_all, G_all = [], []
    for x in meta:
        name = x['image_name']; g = G[name] > 0
        img = torch.from_numpy(np.asarray(Image.open(f'release/train/images/{name}.png').convert('RGB'))).cuda().permute(2, 0, 1)[None].float() / 255
        sign = torch.tensor([2.0 * int(c) - 1 for c in x['message']], device='cuda')
        R = [(y, x0, y + w, x0 + w) for y in range(0, 513 - w, 16) for x0 in range(0, 513 - w, 16)]
        s = score(img, sign, R, size)
        P = np.full((128, 128), -1.0)
        for r, v in zip(R, s): P[r[0] // 4:r[2] // 4, r[1] // 4:r[3] // 4] = np.maximum(P[r[0] // 4:r[2] // 4, r[1] // 4:r[3] // 4], v)
        if 20 < g.sum() < g.size - 20: aucs.append(auc(P[g], P[~g]))
        P_all.append(P); G_all.append(g)
    thr = np.linspace(-0.1, 0.9, 41); sc = [np.mean([miou_image(p > t, g) for p, g in zip(P_all, G_all)]) for t in thr]
    med = np.median([p.max() for p in P_all])
    print(f'{TY} w={w} fed at {size}: pixel AUC (max over windows) {np.mean(aucs):.3f} | best global threshold {thr[int(np.argmax(sc))]:.2f} -> mIoU {max(sc):.3f} | median max margin {med:.2f}', flush=True)

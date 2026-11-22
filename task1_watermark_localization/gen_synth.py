"""Generate synthetic localisation samples for one watermark method.
python gen_synth.py --method pixelseal --n 3000 --out synth/pixelseal --data release --seed 0"""
import argparse, json, hashlib, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask

def is_val(name):  # deterministic 20 % validation split of the real train set
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0

def blob(rng, h, w, frac):
    """random smooth blob mask covering ~frac of an h x w canvas"""
    s = rng.uniform(0.04, 0.25) * max(h, w)
    n = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), s, mode='reflect')
    return n > np.quantile(n, 1 - frac)

def rand_rect(rng, max_ar, fmin=0.003, fmax=0.45, S=512):
    for _ in range(100):
        area = np.exp(rng.uniform(np.log(fmin), np.log(fmax))) * S * S
        ar = np.exp(rng.uniform(-np.log(max_ar), np.log(max_ar)))
        w = int(round(np.sqrt(area * ar))); h = int(round(np.sqrt(area / ar)))
        if 8 <= w <= S and 8 <= h <= S and max(w, h) / min(w, h) <= max_ar:
            x = rng.integers(0, S - w + 1); y = rng.integers(0, S - h + 1)
            return y, x, h, w
    return 100, 100, 200, 200

def rand_tf(rng, m):
    k = rng.integers(0, 4); m = np.rot90(m, k)
    if rng.random() < 0.5: m = m[:, ::-1]
    return np.ascontiguousarray(m)

def bbox(m):
    ys, xs = np.where(m); return ys.min(), xs.min(), ys.max() - ys.min() + 1, xs.max() - xs.min() + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--out', required=True); ap.add_argument('--data', default='release'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_watermark(a.method, device=a.device)
    max_ar = 2.0 if a.method == 'trustmark' else 16.0
    tr = json.loads((data / 'train/metadata.json').read_text()); te = json.loads((data / 'test/metadata.json').read_text())
    # clean sources for this head: images of the OTHER methods (their watermark is not ours)
    src = [(f'train/images/{m["image_name"]}.png') for m in tr if m['method'] != a.method and not is_val(m['image_name'])]
    src += [(f'test/images/{m["image_name"]}.png') for m in te if m['method'] != a.method]
    # real mask library (train split, any method)
    full_lib, part_lib = [], []
    for m in tr:
        if is_val(m['image_name']): continue
        mk = np.asarray(Image.open(data / f'train/masks/{m["image_name"]}.png')) > 127
        if mk.sum() == 0 or mk.all(): continue
        (full_lib if m['full_watermark'] else part_lib).append(mk)
    print(len(src), 'sources', len(full_lib), len(part_lib), 'real masks')
    N = a.n; S = 512
    imgs = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(N, S, S, 3))
    m128 = np.zeros((N, 128, 128), np.uint8)
    for i in range(N):
        img = np.asarray(Image.open(data / src[rng.integers(len(src))]).convert('RGB')).copy()
        if rng.random() < 0.5: img = img[:, ::-1].copy()
        bits = rng.integers(0, 2, wm.nbits)
        if rng.random() < 0.5:   # full watermark with a restored hole
            W = wm.embed(img, bits)
            r = rng.random()
            if r < 0.5: lab = rand_tf(rng, full_lib[rng.integers(len(full_lib))])
            else:
                frac = np.exp(rng.uniform(np.log(0.003), np.log(0.5)))
                if r < 0.8: hole = blob(rng, S, S, frac)
                else:
                    hole = np.zeros((S, S), bool); y, x, h, w = rand_rect(rng, 16.0); hole[y:y + h, x:x + w] = True
                    if rng.random() < 0.5: hole &= ~blob(rng, S, S, rng.uniform(0.1, 0.5))
                lab = ~hole
        else:                    # watermark embedded in a rectangle only (optionally cut to a shape)
            r = rng.random()
            if r < 0.5:
                lab = rand_tf(rng, part_lib[rng.integers(len(part_lib))])
                y, x, h, w = bbox(lab)
                if max(h, w) / max(1, min(h, w)) > max_ar or min(h, w) < 4:
                    y, x, h, w = rand_rect(rng, max_ar); lab = np.zeros((S, S), bool); lab[y:y + h, x:x + w] = True
            else:
                y, x, h, w = rand_rect(rng, max_ar); lab = np.zeros((S, S), bool); lab[y:y + h, x:x + w] = True
                if r > 0.75:
                    sub = blob(rng, h, w, rng.uniform(0.25, 0.8))
                    if rng.random() < 0.5: sub = ~sub
                    if sub.sum() > 30: lab[y:y + h, x:x + w] = sub
            ey, ex, eh, ew = y, x, h, w
            if rng.random() < 0.2:   # sometimes the embedding rectangle is larger than the visible part
                y0 = rng.integers(0, y + 1); x0 = rng.integers(0, x + 1); y1 = rng.integers(y + h, S + 1); x1 = rng.integers(x + w, S + 1)
                if max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0) <= max_ar: ey, ex, eh, ew = y0, x0, y1 - y0, x1 - x0
            W = img.copy()
            try:
                W[ey:ey + eh, ex:ex + ew] = wm.embed(np.ascontiguousarray(img[ey:ey + eh, ex:ex + ew]), bits)
            except Exception as e:
                print('embed fail', eh, ew, repr(e)[:100]); lab[:] = False
        comp = np.where(lab[..., None], W, img)
        imgs[i] = comp; m128[i] = downsample_mask(lab.astype(np.float32))
        if i % 500 == 0: print(i, flush=True)
    imgs.flush(); np.save(out / 'm128.npy', m128); print('done', out)

if __name__ == '__main__':
    main()

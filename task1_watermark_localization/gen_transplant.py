"""Synthetic 'transplant' samples: the watermark is embedded on the WHOLE frame (or a large rectangle) but only an
object-shaped / small region keeps it (object from a watermarked image pasted on the clean one).
python gen_transplant.py --method trustmark --n 2000 --out synth/trustmark_t1 --seed 21"""
import argparse, json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
from gen_synth import is_val, blob, rand_rect, rand_tf, bbox

def one_blob(rng, frac, S=512):
    b = blob(rng, S, S, min(0.6, frac * rng.uniform(1.0, 2.5)))
    lab, n = ndi.label(b)
    if n == 0: return b
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    k = int(np.argmin(np.abs(sizes - frac * S * S))) + 1
    return lab == k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--n', type=int, default=2000); ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='release'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_watermark(a.method, device='cuda'); S = 512; max_ar = 2.0 if a.method == 'trustmark' else 16.0
    tr = json.loads((data / 'train/metadata.json').read_text()); te = json.loads((data / 'test/metadata.json').read_text())
    src = [(f'train/images/{m["image_name"]}.png') for m in tr if m['method'] != a.method and not is_val(m['image_name'])]
    src += [(f'test/images/{m["image_name"]}.png') for m in te if m['method'] != a.method]
    shapes = []
    for m in tr:
        if is_val(m['image_name']) or m['full_watermark']: continue
        mk = np.asarray(Image.open(data / f'train/masks/{m["image_name"]}.png')) > 127
        if mk.sum() < 30: continue
        y, x, h, w = bbox(mk)
        if mk.sum() / (h * w) < 0.97: shapes.append(mk)
    print(len(src), 'sources', len(shapes), 'real shape masks', flush=True)
    N = a.n
    imgs = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(N, S, S, 3)); m128 = np.zeros((N, 128, 128), np.uint8)
    for i in range(N):
        img = np.asarray(Image.open(data / src[rng.integers(len(src))]).convert('RGB')).copy()
        if rng.random() < 0.5: img = img[:, ::-1].copy()
        bits = rng.integers(0, 2, wm.nbits); r = rng.random()
        if r < 0.35: lab = rand_tf(rng, shapes[rng.integers(len(shapes))])
        elif r < 0.8: lab = one_blob(rng, np.exp(rng.uniform(np.log(0.004), np.log(0.4))))
        else:
            y, x, h, w = rand_rect(rng, 16.0); lab = np.zeros((S, S), bool); lab[y:y + h, x:x + w] = True
            if rng.random() < 0.5:
                sub = blob(rng, h, w, rng.uniform(0.3, 0.8))
                if sub.sum() > 30: lab[y:y + h, x:x + w] = sub
        if lab.sum() < 16: lab = one_blob(rng, 0.02)
        W = img.copy()
        if rng.random() < 0.65: W = wm.embed(img, bits)                      # whole-frame embedding
        else:                                                              # large rectangle containing the shape
            y, x, h, w = bbox(lab)
            for _ in range(20):
                y0 = rng.integers(0, y + 1); x0 = rng.integers(0, x + 1); y1 = rng.integers(y + h, S + 1); x1 = rng.integers(x + w, S + 1)
                if max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0) <= max_ar: break
            else: y0, x0, y1, x1 = 0, 0, S, S
            W[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(img[y0:y1, x0:x1]), bits)
        imgs[i] = np.where(lab[..., None], W, img); m128[i] = downsample_mask(lab.astype(np.float32))
        if i % 500 == 0: print(i, flush=True)
    imgs.flush(); np.save(out / 'm128.npy', m128); print('done', out)

if __name__ == '__main__':
    main()

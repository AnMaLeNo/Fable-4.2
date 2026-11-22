"""Unified synthetic generator v4 (= gen3 + content-aligned object masks from Segment Anything: holes and transplanted
objects follow the contours of real objects of the SAME image, as in the real data), three real types in the real proportions:
  A (50 %) whole-frame watermark with restored holes     (blobs / objects / rounded shapes, sizes matched to real masks)
  B (25 %) watermark embedded on a rectangle only          (rectangle sizes sampled from the real pure-rectangle masks)
  C (25 %) whole-frame watermark kept on an object only    ('transplant')
python gen4.py --method trustmark --n 2000 --out synth4/trustmark_p2_0 --seed 100 --objmasks objmasks_all.npz"""
import argparse, json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
from gen_synth import is_val, bbox
S = 512

def smooth_noise(rng, scale):
    k = max(4, int(S / max(2.0, scale)))                       # low-res noise upsampled: much faster than a big gaussian filter
    n = rng.standard_normal((k + 3, k + 3)).astype(np.float32)
    n = np.asarray(Image.fromarray(n, mode='F').resize((S + 64, S + 64), Image.Resampling.BICUBIC))
    oy, ox = rng.integers(0, 64, 2)
    return n[oy:oy + S, ox:ox + S]

def blobs(rng, frac):
    n = smooth_noise(rng, rng.uniform(0.04, 0.22) * S)
    if rng.random() < 0.3: n = n + 0.5 * smooth_noise(rng, rng.uniform(0.02, 0.06) * S)
    return n > np.quantile(n, 1 - frac)

def one_blob(rng, frac):
    b = blobs(rng, min(0.6, frac * rng.uniform(1.0, 2.5))); lab, n = ndi.label(b)
    if n == 0: return b
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return lab == int(np.argmin(np.abs(sizes - frac * S * S))) + 1

def dots(rng, m):
    yy, xx = np.mgrid[:S, :S]
    for _ in range(rng.integers(3, 30)):
        cy, cx = rng.integers(0, S, 2); a, b = rng.uniform(3, 14, 2)
        m |= ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 < 1
    return m

def geometric(rng, frac):
    m = np.zeros((S, S), np.float32); area = frac * S * S; ar = np.exp(rng.uniform(-1.1, 1.1))
    h = int(np.clip(np.sqrt(area / ar), 8, S)); w = int(np.clip(np.sqrt(area * ar), 8, S)); y = rng.integers(-h // 4, S - 3 * h // 4); x = rng.integers(-w // 4, S - 3 * w // 4)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[:S, :S]; return ((yy - y - h / 2) / (h / 2)) ** 2 + ((xx - x - w / 2) / (w / 2)) ** 2 < 1
    m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1
    return ndi.gaussian_filter(m, rng.uniform(1, 0.25 * min(h, w) + 1)) > 0.5

def paste_shape(rng, canvas, crop, scale=(0.6, 1.6)):
    f = rng.uniform(*scale); h = max(4, int(crop.shape[0] * f)); w = max(4, int(crop.shape[1] * f))
    h = min(h, S); w = min(w, S)
    c = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((w, h), Image.Resampling.BILINEAR)) > 127
    if rng.random() < 0.5: c = c[:, ::-1]
    y = rng.integers(0, S - h + 1); x = rng.integers(0, S - w + 1); canvas[y:y + h, x:x + w] |= c
    return canvas

def strip(rng):
    m = np.zeros((S, S), bool); t = int(rng.uniform(12, 120)); edge = 6 * smooth_noise(rng, 60)[0] * rng.uniform(0, 2)
    for x in range(S): m[:max(2, int(t + edge[x])), x] = True
    return np.rot90(m, rng.integers(0, 4)).copy()

def hole_frac(rng): return float(np.clip(np.exp(rng.normal(np.log(0.06), 1.0)), 0.003, 0.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--n', type=int, default=1200); ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='release'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--mix', default='0.5,0.25,0.25')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--rect_izoom', default='', help='mean,std: magnify the watermarked rectangle patch about its centre and crop it to the box (real: decodable frame ~4 % larger than the label box)'); ap.add_argument('--rect_margin', type=int, default=0, help='embed rectangles on a box dilated by this many pixels and keep the inner box (real data: 2)'); ap.add_argument('--clean_src', default='', help='afull.json: use only sources that are not fully watermarked by another method'); ap.add_argument('--objmasks', default='objmasks_all.npz')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_watermark(a.method, device=a.device); max_ar = 2.0 if a.method == 'trustmark' else 16.0; mix = np.cumsum([float(v) for v in a.mix.split(',')])
    tr = json.loads((data / 'train/metadata.json').read_text()); te = json.loads((data / 'test/metadata.json').read_text())
    src = [f'train/images/{m["image_name"]}.png' for m in tr if m['method'] != a.method and not is_val(m['image_name'])] + [f'test/images/{m["image_name"]}.png' for m in te if m['method'] != a.method]
    if a.clean_src:
        AF = json.load(open(a.clean_src)); full_tr = {m['image_name'] for m in tr if m['full_watermark']}
        src = [p for p in src if (p.startswith('train/') and Path(p).stem not in full_tr) or (p.startswith('test/') and AF['test_' + Path(p).stem] < 0.7)]
    full_lib, hole_objs, obj_lib, rects = [], [], [], []
    for m in tr:
        if is_val(m['image_name']): continue
        mk = np.asarray(Image.open(data / f'train/masks/{m["image_name"]}.png')) > 127
        if mk.sum() < 30 or mk.all(): continue
        if m['full_watermark']:
            full_lib.append(mk); lab, n = ndi.label(~mk)
            for i in range(1, n + 1):
                c = lab == i
                if c.sum() >= 200: y, x, h, w = bbox(c); hole_objs.append(c[y:y + h, x:x + w])
        else:
            y, x, h, w = bbox(mk)
            if mk.sum() / (h * w) > 0.97:
                if (a.method == 'trustmark') == (m['method'] == 'trustmark'): rects.append((h, w))
            else: obj_lib.append(mk[y:y + h, x:x + w])
    OM = np.load(a.objmasks)
    def sam_masks(path, flip):
        k = path.split('/')[0] + '_' + Path(path).stem
        if k not in OM.files: return np.zeros((0, S, S), bool)
        p = OM[k]
        if len(p) == 0: return np.zeros((0, S, S), bool)
        mm = np.unpackbits(p).reshape(-1, S, S).astype(bool)
        return mm[:, :, ::-1] if flip else mm
    def pick(mm, rng, kmax, amax):
        ar = mm.reshape(len(mm), -1).mean(1); out = np.zeros((S, S), bool)
        for _ in range(kmax):
            ok = np.where(ar + out.mean() <= amax)[0]
            if len(ok) == 0: break
            w = np.sqrt(ar[ok]); out |= mm[rng.choice(ok, p=w / w.sum())]
        return out
    print(len(src), 'sources;', len(full_lib), 'full masks,', len(hole_objs), 'hole objects,', len(obj_lib), 'objects,', len(rects), 'rects', flush=True)
    N = a.n; imgs = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(N, S, S, 3)); m128 = np.zeros((N, 128, 128), np.uint8); types = np.zeros(N, np.uint8)
    for i in range(N):
        sp = src[rng.integers(len(src))]; img = np.asarray(Image.open(data / sp).convert('RGB')).copy(); flip = rng.random() < 0.5
        if flip: img = img[:, ::-1].copy()
        sm = None
        bits = rng.integers(0, 2, wm.nbits); u = rng.random(); t = int(np.searchsorted(mix, u * mix[-1]))
        if t == 0:      # A: holes in a whole-frame watermark
            r = rng.random(); frac = hole_frac(rng)
            if r < 0.3:
                lab = full_lib[rng.integers(len(full_lib))]; lab = np.rot90(lab, rng.integers(0, 4)); lab = (lab[:, ::-1] if rng.random() < 0.5 else lab).copy(); hole = ~lab
            elif r < 0.6:
                hole = blobs(rng, frac)
                if rng.random() < 0.2: hole = dots(rng, hole)
            elif r < 0.9:
                sm = sam_masks(sp, flip); hole = pick(sm, rng, rng.integers(1, 4), 0.5) if len(sm) else np.zeros((S, S), bool)
                if hole.sum() < 40:
                    for _ in range(rng.integers(1, 4)): hole = paste_shape(rng, hole, (hole_objs + obj_lib)[rng.integers(len(hole_objs) + len(obj_lib))])
            else:
                hole = geometric(rng, frac)
                if rng.random() < 0.3: hole |= geometric(rng, hole_frac(rng) / 2)
            if hole.mean() > 0.55 or hole.sum() < 40: hole = blobs(rng, 0.05)
            lab = ~hole; W = wm.embed(img, bits)
        elif t == 1:    # B: rectangle-only embedding
            for _ in range(50):
                h, w = rects[rng.integers(len(rects))]; h = int(np.clip(h * rng.uniform(0.75, 1.3), 12, S)); w = int(np.clip(w * rng.uniform(0.75, 1.3), 12, S))
                if rng.random() < 0.5 and max(h, w) / min(h, w) <= max_ar: h, w = w, h
                if max(h, w) / min(h, w) <= max_ar: break
            y = rng.integers(0, S - h + 1); x = rng.integers(0, S - w + 1); lab = np.zeros((S, S), bool); lab[y:y + h, x:x + w] = True
            mg = a.rect_margin; ey0, ex0, ey1, ex1 = max(0, y - mg), max(0, x - mg), min(S, y + h + mg), min(S, x + w + mg)
            emb = wm.embed(np.ascontiguousarray(img[ey0:ey1, ex0:ex1]), bits)
            patch = emb[y - ey0:y - ey0 + h, x - ex0:x - ex0 + w]
            if a.rect_izoom:
                zm, zs = [float(v) for v in a.rect_izoom.split(',')]; sz = float(np.clip(rng.normal(zm, zs), 1.0, 1.09))
                nh, nw = max(h, int(round(h * sz))), max(w, int(round(w * sz)))
                if nh > h or nw > w:
                    big = np.asarray(Image.fromarray(patch).resize((nw, nh), [Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.LANCZOS][rng.integers(3)]))
                    patch = big[(nh - h) // 2:(nh - h) // 2 + h, (nw - w) // 2:(nw - w) // 2 + w]
            W = img.copy(); W[y:y + h, x:x + w] = patch
        else:           # C: whole-frame watermark kept on an object
            r = rng.random(); lab = np.zeros((S, S), bool)
            if r < 0.7:
                sm = sam_masks(sp, flip); lab = pick(sm, rng, 1 if rng.random() < 0.8 else 2, 0.45) if len(sm) else lab
                if lab.sum() < 60: lab = paste_shape(rng, lab, obj_lib[rng.integers(len(obj_lib))], (0.7, 1.4))
            elif r < 0.8: lab = paste_shape(rng, lab, obj_lib[rng.integers(len(obj_lib))], (0.7, 1.4))
            elif r < 0.93: lab = one_blob(rng, np.exp(rng.uniform(np.log(0.004), np.log(0.4))))
            else: lab = strip(rng)
            if lab.sum() < 60 or lab.mean() > 0.47: lab = one_blob(rng, 0.03)
            W = wm.embed(img, bits)
        if rng.random() < 0.15:   # feathered edge
            al = ndi.gaussian_filter(lab.astype(np.float32), rng.uniform(0.5, 1.5))[..., None]
            comp = np.clip(al * W + (1 - al) * img, 0, 255).round().astype(np.uint8)
        else: comp = np.where(lab[..., None], W, img)
        imgs[i] = comp; m128[i] = downsample_mask(lab.astype(np.float32)); types[i] = t
        if i % 300 == 0: print(i, flush=True)
    imgs.flush(); np.save(out / 'm128.npy', m128); np.save(out / 'types.npy', types); print('done', out)

if __name__ == '__main__':
    main()

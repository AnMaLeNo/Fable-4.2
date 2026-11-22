"""'Full' composites (whole image watermarked, a hole restored) + the re-embedded residual template r_hat = embed(x, msg) - x.
Also writes the same template for the real images of the method (train 'full' images and all test images).
python gen_full.py --method trustmark --n 6000 --out synthF/trustmark_s1 --seed 1"""
import argparse, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
from gen_synth import is_val, blob, rand_rect, rand_tf

def rhat(wm, img, bits):
    return np.clip(wm.embed(img, bits).astype(np.int16) - img.astype(np.int16), -127, 127).astype(np.int8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--n', type=int, default=6000); ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='release'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--real_only', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_watermark(a.method, device='cuda'); S = 512
    tr = json.loads((data / 'train/metadata.json').read_text()); te = json.loads((data / 'test/metadata.json').read_text())
    # real images of this method
    realdir = out.parent / f'real_{a.method}'
    if not (realdir / 'train.npz').exists():
        realdir.mkdir(parents=True, exist_ok=True)
        for split, meta in (('train', tr), ('test', te)):
            ms = [m for m in meta if m['method'] == a.method]
            imgs = np.stack([np.asarray(Image.open(data / f'{split}/images/{m["image_name"]}.png').convert('RGB')) for m in ms])
            rh = np.stack([rhat(wm, im, [int(c) for c in m['message']]) for im, m in zip(imgs, ms)])
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255
            acc = np.array([float((wm.decode(x[i:i + 1])[0].cpu().numpy() == np.array([int(c) for c in m['message']])).mean()) for i, m in enumerate(ms)])
            np.savez(realdir / f'{split}.npz', images=imgs, rhat=rh, names=[m['image_name'] for m in ms], afull=acc,
                     full=[bool(m.get('full_watermark', False)) for m in ms])
            print('real', split, len(ms), 'afull mean', acc.mean(), flush=True)
    if a.real_only: return
    src = [(f'train/images/{m["image_name"]}.png') for m in tr if m['method'] != a.method and not is_val(m['image_name'])]
    src += [(f'test/images/{m["image_name"]}.png') for m in te if m['method'] != a.method]
    full_lib = []
    for m in tr:
        if is_val(m['image_name']) or not m['full_watermark']: continue
        mk = np.asarray(Image.open(data / f'train/masks/{m["image_name"]}.png')) > 127
        if 0 < mk.sum() < mk.size: full_lib.append(mk)
    N = a.n
    imgs = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(N, S, S, 3))
    rhs = np.lib.format.open_memmap(out / 'rhat.npy', mode='w+', dtype=np.int8, shape=(N, S, S, 3))
    m128 = np.zeros((N, 128, 128), np.uint8)
    for i in range(N):
        img = np.asarray(Image.open(data / src[rng.integers(len(src))]).convert('RGB')).copy()
        if rng.random() < 0.5: img = img[:, ::-1].copy()
        bits = rng.integers(0, 2, wm.nbits); W = wm.embed(img, bits); r = rng.random()
        if r < 0.4: lab = rand_tf(rng, full_lib[rng.integers(len(full_lib))])
        else:
            frac = np.exp(rng.uniform(np.log(0.002), np.log(0.5)))
            if r < 0.8:
                hole = blob(rng, S, S, frac)
                if rng.random() < 0.3: hole |= blob(rng, S, S, np.exp(rng.uniform(np.log(0.002), np.log(0.1))))
            else:
                hole = np.zeros((S, S), bool); y, x, h, w = rand_rect(rng, 16.0); hole[y:y + h, x:x + w] = True
                if rng.random() < 0.5: hole &= ~blob(rng, S, S, rng.uniform(0.1, 0.5))
            lab = ~hole
        comp = np.where(lab[..., None], W, img)
        imgs[i] = comp; rhs[i] = rhat(wm, comp, bits); m128[i] = downsample_mask(lab.astype(np.float32))
        if i % 500 == 0: print(i, flush=True)
    imgs.flush(); rhs.flush(); np.save(out / 'm128.npy', m128); print('done', out)

if __name__ == '__main__':
    main()

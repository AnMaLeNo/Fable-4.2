"""Rectangle stage for 'partial' predictions, using the pretrained decoder and the known message.
StegaStamp ('refine'): the decode margin peaks sharply on the embedding rectangle -> dense search around each proposal,
  accepted if bit accuracy >= thr, output = rectangle shrunk by the calibrated bias (2 px per side).
TrustMark / PixelSeal ('verify'): decoding is too flat to localise, but it confirms that a proposal IS the embedding
  rectangle -> output the max-sum rectangle of the logits exactly.
python rect_pipe.py --pred preds/X --method stegastamp --split val|test [--shard i/n]  -> preds/X/rect_<method>_<split>[_i].npz"""
import argparse, itertools, json, time, numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
from rect_search import Scorer, refine, clip_rect
from postproc import maxrect
M = ['pixelseal', 'trustmark', 'stegastamp']
CFG = {'stegastamp': dict(mode='refine', thr=0.85, bias=(-2, -2, 2, 2)), 'trustmark': dict(mode='verify', thr=0.85, bias=(0, 0, 0, 0)), 'pixelseal': dict(mode='verify', thr=0.80, bias=(0, 0, 0, 0))}

def grid(r):
    mm = np.zeros((512, 512), np.float32); mm[max(0, r[0]):max(0, r[2]), max(0, r[1]):max(0, r[3])] = 1; return downsample_mask(mm).astype(bool)

def proposals(l, extra=None, max_n=3):
    out = []
    for lg in [l] + ([extra] if extra is not None else []):
        m = lg > 0
        if m.sum() < 2: continue
        lab, n = ndi.label(m); sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for i in np.argsort(-sizes)[:max_n]:
            if sizes[i] < 2: continue
            ys, xs = np.where(lab == i + 1); out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
        r, _ = maxrect(np.tanh(lg / 2.0))
        if r is not None: out.append((r[0] * 4, r[1] * 4, r[2] * 4, r[3] * 4))
    return list(dict.fromkeys(out))

def dense_refine(sc, r0, rng=16, top=4):
    """coarse joint grid (step 8, +-rng), finer joint grid (step 4) around the best few, then coordinate ascent"""
    offs = list(range(-rng, rng + 1, 8))
    c1 = list({clip_rect((r0[0] + a, r0[1] + b, r0[2] + c, r0[3] + d)) for a, b, c, d in itertools.product(offs, repeat=4)})
    m1, _ = sc(c1); best = (None, -9, 0)
    for j in np.argsort(-m1)[:top]:
        r, mar, acc = refine(sc, c1[j], coarse=(4,), fine=(2, 1))
        if mar > best[1]: best = (r, mar, acc)
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--method', required=True); ap.add_argument('--split', default='val')
    ap.add_argument('--zoom', default=''); ap.add_argument('--data', default='release'); ap.add_argument('--shard', default='0/1'); ap.add_argument('--max_frac', type=float, default=0.47)
    a = ap.parse_args(); si, sn = map(int, a.shard.split('/')); cfg = CFG[a.method]; mi = M.index(a.method)
    z = np.load(f'{a.pred}/{a.split}.npz', allow_pickle=True); zz = np.load(f'{a.zoom}/{a.split}.npz', allow_pickle=True)['logits'] if a.zoom else None
    folder = 'train' if a.split == 'val' else 'test'
    MSG = {x['image_name']: np.array([int(c) for c in x['message']], np.uint8) for x in json.load(open(f'{a.data}/{folder}/metadata.json'))}
    wm = load_watermark(a.method, device='cuda'); idx = [k for k in range(len(z['names'])) if z['method'][k] == mi][si::sn]; t0 = time.time()
    names, rects, accs, masks = [], [], [], []
    for n, k in enumerate(idx):
        l = z['logits'][k]; name = str(z['names'][k]); best = (None, -9, 0.0)
        if (l > 0).mean() < a.max_frac:
            sc = Scorer(wm, np.asarray(Image.open(f'{a.data}/{folder}/images/{name}.png').convert('RGB')), MSG[name])
            for r0 in proposals(l, None if zz is None else zz[k]):
                if cfg['mode'] == 'refine': r, mar, acc = dense_refine(sc, r0)
                else:
                    mar, acc = sc([clip_rect(r0)]); r, mar, acc = clip_rect(r0), float(mar[0]), float(acc[0])
                if mar > best[1]: best = (r, mar, acc)
        ok = best[0] is not None and best[2] >= cfg['thr']
        names.append(name); accs.append(best[2]); rects.append(best[0] if best[0] is not None else (0, 0, 0, 0))
        masks.append(grid(tuple(int(v - b) for v, b in zip(best[0], cfg['bias']))) if ok else (l > 0))
        if n % 20 == 0: print(n, len(idx), round(time.time() - t0), 's', flush=True)
    sfx = '' if sn == 1 else f'_{si}'
    np.savez(f'{a.pred}/rect_{a.method}_{a.split}{sfx}.npz', names=names, rects=np.array(rects), accs=np.array(accs), masks=np.array(masks))
    if a.split == 'val':
        gt = {str(n): g for n, g in zip(z['names'], z['gt'])}; base = {str(n): l > 0 for n, l in zip(z['names'], z['logits'])}
        acc = np.array(accs) >= cfg['thr']
        b = np.array([miou_image(base[n], gt[n]) for n in names]); p = np.array([miou_image(mk, gt[n]) for n, mk in zip(names, masks)])
        print(f'{a.method}: accepted {acc.sum()} of {len(names)} | on accepted: base {b[acc].mean():.3f} -> rect {p[acc].mean():.3f} | all images of the method: {b.mean():.4f} -> {p.mean():.4f}')

if __name__ == '__main__':
    main()

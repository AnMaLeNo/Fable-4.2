"""Real-to-real remix: new training composites built only from REAL labelled images of the same method, so the watermark
texture (and whatever the real pipeline did to it) is the real one.
  A' : a real fully-watermarked image with an extra hole filled by the pixels of a real 'partial' image (clean donor)
  C' : an object region of a real fully-watermarked image pasted in place on a real 'partial' image
  B' : the rectangle of a real rectangle-type image pasted (moved or not) on a real 'partial' image
Shapes of holes / objects: Segment-Anything masks of the watermark donor (content-aligned), or smooth blobs."""
import numpy as np
from PIL import Image
from pathlib import Path
from wmloc.utils import downsample_mask
from gen3 import blobs, one_blob, hole_frac
S = 512

class Remix:
    def __init__(s, real, data, objmasks, pseudo=()):
        """real: list of (img, m128, mi, full, name) of ONE method (training images only).
        pseudo: list of (img, name, label128, conf128, is_full, split) = pseudo-labelled unlabeled images used as extra donors;
        their label / confidence are carried through the composite (loss weight 0 where not confident)."""
        s.M = {r[4]: np.asarray(Image.open(Path(data) / f'train/masks/{r[4]}.png')) > 127 for r in real}
        s.C = {}; s.SP = {r[4]: 'train' for r in real}
        up = lambda a: np.repeat(np.repeat(a, 4, 0), 4, 1)
        real = list(real)
        for (img, name, lab, conf, isfull, split) in pseudo:
            s.M[name] = up(lab).astype(bool); s.C[name] = up(conf).astype(bool); s.SP[name] = split
            real.append((img, lab, -1, isfull, name))
        s.OM = np.load(objmasks)
        s.full = [r for r in real if r[3]]; s.part = [r for r in real if not r[3]]; s.rect = []
        for r in s.part:
            m = s.M[r[4]]; ys, xs = np.where(m)
            if len(ys) and m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) > 0.97: s.rect.append((r, (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)))
    def sam(s, name, rng, kmax, amax, within=None):
        k = s.SP.get(name, 'train') + '_' + name
        if k not in s.OM.files or len(s.OM[k]) == 0: return None
        mm = np.unpackbits(s.OM[k]).reshape(-1, S, S).astype(bool)
        if within is not None: mm = mm & within[None]
        ar = mm.reshape(len(mm), -1).mean(1); out = np.zeros((S, S), bool)
        for _ in range(kmax):
            ok = np.where((ar > 0.0015) & (ar + out.mean() <= amax))[0]
            if len(ok) == 0: break
            w = np.sqrt(ar[ok]); out |= mm[rng.choice(ok, p=w / w.sum())]
        return out if out.sum() > 60 else None
    def __call__(s, rng):
        u = rng.random(); B = s.part[rng.integers(len(s.part))]; MB = s.M[B[4]]; one = np.ones((S, S), bool); CB = s.C.get(B[4], one)
        if u < 0.45 and s.full:      # A'
            A = s.full[rng.integers(len(s.full))]; MA = s.M[A[4]]
            H = s.sam(A[4], rng, int(rng.integers(1, 4)), 0.5) if rng.random() < 0.6 else None
            if H is None: H = blobs(rng, hole_frac(rng))
            img = np.where(H[..., None], B[0], A[0]); lab = np.where(H, MB, MA); conf = np.where(H, CB, s.C.get(A[4], one))
        elif u < 0.75 and s.full:    # C'
            A = s.full[rng.integers(len(s.full))]; MA = s.M[A[4]]
            O = s.sam(A[4], rng, 1 if rng.random() < 0.8 else 2, 0.45, within=MA) if rng.random() < 0.75 else None
            if O is None: O = one_blob(rng, float(np.exp(rng.uniform(np.log(0.004), np.log(0.4))))) & MA
            img = np.where(O[..., None], A[0], B[0]); lab = np.where(O, MA, MB); conf = np.where(O, s.C.get(A[4], one), CB)
        elif s.rect:                 # B'
            (Rr, (y0, x0, y1, x1)) = s.rect[rng.integers(len(s.rect))]; h, w = y1 - y0, x1 - x0
            ny, nx = (int(rng.integers(0, S - h + 1)), int(rng.integers(0, S - w + 1))) if rng.random() < 0.7 else (y0, x0)
            img = B[0].copy(); lab = MB.copy(); conf = CB.copy(); img[ny:ny + h, nx:nx + w] = Rr[0][y0:y1, x0:x1]; lab[ny:ny + h, nx:nx + w] = True; conf[ny:ny + h, nx:nx + w] = True
        else:
            return B[0], B[1], (CB.reshape(128, 4, 128, 4).mean((1, 3)) > 0.99).astype(np.float32)
        return np.ascontiguousarray(img), downsample_mask(lab.astype(np.float32)), (conf.reshape(128, 4, 128, 4).mean((1, 3)) > 0.99).astype(np.float32)

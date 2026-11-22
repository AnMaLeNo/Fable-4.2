"""Degradations applied to synthetic composites at training time. Finding: the real watermarked content is mildly low-pass /
re-encoded (a synthetic-only StegaStamp model sees 5 % of a real fully watermarked frame, 100 % of a pristine kit embedding,
1.5 % after a 0.75x down-up resize, 4 % after JPEG 75), so pristine embeddings teach cues that do not exist in the real data."""
import io, numpy as np
from PIL import Image
from scipy import ndimage as ndi
RS = Image.Resampling
KER = [RS.BICUBIC, RS.LANCZOS, RS.BILINEAR]

def updown(img, rng, lo=0.6, hi=0.92):
    s = int(round(512 * rng.uniform(lo, hi))); k1 = KER[rng.integers(3)]; k2 = KER[rng.integers(2)]
    return np.asarray(Image.fromarray(img).resize((s, s), k1).resize((512, 512), k2))

def jpeg(img, rng, lo=70, hi=96):
    b = io.BytesIO(); Image.fromarray(img).save(b, 'JPEG', quality=int(rng.integers(lo, hi)))
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB'))

def webp(img, rng, lo=70, hi=96):
    b = io.BytesIO(); Image.fromarray(img).save(b, 'WEBP', quality=int(rng.integers(lo, hi)))
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB'))

def blur(img, rng, lo=0.4, hi=0.9):
    s = rng.uniform(lo, hi); return ndi.gaussian_filter(img.astype(np.float32), (s, s, 0)).round().clip(0, 255).astype(np.uint8)

OPS = {'updown': updown, 'jpeg': jpeg, 'webp': webp, 'blur': blur, 'none': lambda img, rng: img}
MIXES = {
    'off': {'none': 1.0},
    'mix1': {'updown': 0.4, 'jpeg': 0.25, 'webp': 0.1, 'blur': 0.1, 'none': 0.15},
    'updown': {'updown': 0.85, 'none': 0.15},
    'jpeg': {'jpeg': 0.85, 'none': 0.15},
}

def degrade(img, rng, mix='mix1'):
    names = list(MIXES[mix]); p = np.array([MIXES[mix][n] for n in names])
    return OPS[names[rng.choice(len(names), p=p / p.sum())]](img, rng)


def zoom_pair(img, m128, rng, lo=1.0, hi=1.07):
    """Real images carry the kit's interpolation comb at a per-image shifted frequency: the watermarked frame was resized by an
    isotropic factor ~0.99-1.065 (median 1.03) and cropped to 512 AFTER embedding. Same operation on a synthetic composite and
    on its label."""
    if rng.random() < 0.25: return img, m128          # keep a share of pristine embeddings
    s = rng.uniform(lo, hi); n = max(512, int(round(512 * s)))
    if n == 512: return img, m128
    oy, ox = int(rng.integers(0, n - 512 + 1)), int(rng.integers(0, n - 512 + 1))
    big = np.asarray(Image.fromarray(img).resize((n, n), KER[rng.integers(3)]))[oy:oy + 512, ox:ox + 512]
    lab = np.asarray(Image.fromarray(np.repeat(np.repeat(m128, 4, 0), 4, 1) * 255).resize((n, n), RS.BILINEAR))[oy:oy + 512, ox:ox + 512]
    return big, (lab.reshape(128, 4, 128, 4).mean((1, 3)) >= 127.5).astype(np.uint8)

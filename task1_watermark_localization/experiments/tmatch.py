"""Rectangle finder by template matching (no segmenter): the kit residual for the known message, resized to a candidate
rectangle size, is correlated with the image at every position (FFT); the best (size, position) candidates are then checked /
refined with the decoder (rect_pipe.dense_refine)."""
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi

def ss_template(wm, msg, content=None):
    """message-specific StegaStamp residual at 400x400 (difference between the message and its complement)"""
    c = np.full((1, 400, 400, 3), 0.5, np.float32) if content is None else content[None].astype(np.float32)
    r1 = wm._run(c, ['residual'], np.asarray([msg], np.float32))[0][0]; r0 = wm._run(c, ['residual'], np.asarray([1 - np.asarray(msg)], np.float32))[0][0]
    return (r1 - r0) * 255 / 2

ZP = 1.0
SIZES = [int(round(v)) for v in np.geomspace(16, 512, 46)]

@torch.no_grad()
def find(img_u8, T400, dev='cuda', sizes=SIZES, max_ar=16.0, topk=6, hp_sigma=6.0):
    """returns list of (score, (y0, x0, y1, x1)) best first"""
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(dev).permute(2, 0, 1)[None].float()
    k = int(4 * hp_sigma) | 1; g = torch.exp(-(torch.arange(k, device=dev) - k // 2) ** 2 / (2 * hp_sigma ** 2)); g = g / g.sum()
    blur = F.conv2d(F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), padding=(k // 2, 0), groups=3) if True else x
    xb = x - F.pad(blur, (0, 0, 0, 0))[:, :, :512, :512] if blur.shape[-2] == 512 else x - F.interpolate(blur, size=(512, 512))
    X = torch.fft.rfft2(xb); x2 = (xb ** 2).sum(1, keepdim=True); ii = F.pad(x2.cumsum(2).cumsum(3), (1, 0, 1, 0))
    T = torch.from_numpy(T400).to(dev).permute(2, 0, 1)[None].float(); out = []
    for h in sizes:
        ts = []
        ws = [w for w in sizes if max(h, w) / min(h, w) <= max_ar]
        for w in ws:
            t = F.interpolate(T, size=(h, w), mode='bilinear', align_corners=False, antialias=True); t = t - t.mean((2, 3), keepdim=True)
            ts.append(F.pad(t, (0, 512 - w, 0, 512 - h)) / (t.norm() + 1e-6))
        tb = torch.cat(ts); C = torch.fft.irfft2(X * torch.conj(torch.fft.rfft2(tb)), s=(512, 512)).sum(1)      # len(ws) x 512 x 512, valid where the box fits
        for j, w in enumerate(ws):
            e = (ii[0, 0, h:, w:] - ii[0, 0, :-h, w:] - ii[0, 0, h:, :-w] + ii[0, 0, :-h, :-w]).clamp(min=1e-3).sqrt()   # (513-h) x (513-w)
            s = C[j, :513 - h, :513 - w] / e * float(np.sqrt(h * w)) ** ZP; v, idx = s.flatten().max(0); y, xx = divmod(int(idx), s.shape[1])
            out.append((float(v), (y, xx, y + h, xx + w)))
    out.sort(key=lambda o: -o[0]); keep = []
    for sc, r in out:      # non-maximum suppression on boxes
        if all(_iou(r, q) < 0.5 for _, q in keep): keep.append((sc, r))
        if len(keep) >= topk: break
    return keep

def _iou(a, b):
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iy * ix
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

if __name__ == '__main__':
    import sys, json, time
    from PIL import Image
    from wmloc.watermarks import load_watermark
    from wmloc.data import LocalisationDataset
    wm = load_watermark('stegastamp', device='cuda'); ds = LocalisationDataset('release', 'train', methods=['stegastamp']); hits = []; t0 = time.time(); n = 0
    for i in range(len(ds)):
        it = ds[i]
        if it['full_watermark']: continue
        mk = it['mask'].astype(bool); ys, xs = np.where(mk); r = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        if mk.sum() / ((r[2] - r[0]) * (r[3] - r[1])) < 0.97: continue
        T = ss_template(wm, it['message']); c = find(it['image'], T, hp_sigma=float(sys.argv[1]) if len(sys.argv) > 1 else 6.0)
        ious = [_iou(q, r) for _, q in c]; hits.append((max(ious), ious[0], c[0][0], min(r[2] - r[0], r[3] - r[1]))); n += 1
        if n <= 12: print('gt', r, 'top1', c[0][1], 'score %.3f iou %.2f | best-of-6 iou %.2f' % (c[0][0], ious[0], max(ious)), flush=True)
    h = np.array(hits); print('n=%d  top-1 IoU>0.5: %.2f | best-of-6 IoU>0.5: %.2f | best-of-6 IoU>0.7: %.2f | mean best IoU %.3f | %.1f s/image' % (len(h), (h[:, 1] > 0.5).mean(), (h[:, 0] > 0.5).mean(), (h[:, 0] > 0.7).mean(), h[:, 0].mean(), (time.time() - t0) / len(h)))

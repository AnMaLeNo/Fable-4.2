"""Rectangle search with the pretrained decoders: the decode margin against the known message peaks sharply when a
crop coincides with the rectangle the watermark was embedded in."""
import itertools, numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as ndi

class Scorer:
    def __init__(s, wm, img_u8, msg, dev='cuda', obj='tanh2'):
        s.obj = obj
        s.wm = wm; s.dev = dev; s.name = wm.name
        s.x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(dev).permute(2, 0, 1)[None].float() / 255
        s.sign = torch.from_numpy(2 * msg.astype(np.float32) - 1).to(dev); s.cache = {}
        s.res = 400 if s.name == 'stegastamp' else wm.resolution
        if s.name == 'stegastamp' and not hasattr(wm, 'dec_sess'):
            import onnxruntime as ort
            if hasattr(ort, 'preload_dlls'): ort.preload_dlls()
            from pathlib import Path
            p = Path(__file__).resolve().parent / 'models/stegastamp/stega_dec.onnx'
            wm.dec_sess = ort.InferenceSession(str(p), providers=[('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider'])
    def _crops(s, rects):
        out = []
        for (y0, x0, y1, x1) in rects:
            c = s.x[:, :, y0:y1, x0:x1]
            if s.name == 'stegastamp':
                out.append(F.interpolate(c, size=(400, 400), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1))
            else:
                out.append(F.interpolate(c, size=(s.res, s.res), mode='bilinear', align_corners=False, antialias=True))
        return torch.cat(out)
    @torch.no_grad()
    def __call__(s, rects, bs=32):
        if s.name == 'stegastamp': bs = 8
        """rects: list of (y0, x0, y1, x1) -> (margin, bitacc) arrays"""
        res = [None] * len(rects); todo = []
        for i, r in enumerate(rects):
            r = tuple(int(v) for v in r)
            if r in s.cache: res[i] = s.cache[r]
            elif r[2] - r[0] < 6 or r[3] - r[1] < 6 or r[0] < 0 or r[1] < 0 or r[2] > 512 or r[3] > 512: res[i] = (-1.0, 0.0)
            else: todo.append((i, r))
        for k in range(0, len(todo), bs):
            ch = todo[k:k + bs]; c = s._crops([r for _, r in ch])
            if s.name == 'stegastamp':
                lg = torch.from_numpy(s.wm.dec_sess.run(None, {'image': c.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)})[0]).to(s.dev).float()
            elif s.name == 'pixelseal':
                lg = s.wm.det(c)[:, 1:].float()
            else:
                lg = s.wm.net(c * 2 - 1).float()
            z = lg * s.sign
            mar = (z.mean(1) if s.obj == 'raw' else torch.tanh(z / float(s.obj[4:])).mean(1)).cpu().numpy(); acc = ((lg > 0) == (s.sign > 0)).float().mean(1).cpu().numpy()
            for (i, r), a, b in zip(ch, mar, acc):
                s.cache[r] = (float(a), float(b)); res[i] = s.cache[r]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])

def clip_rect(r):
    y0, x0, y1, x1 = r
    y0 = min(max(0, y0), 506); x0 = min(max(0, x0), 506); y1 = max(min(512, y1), y0 + 6); x1 = max(min(512, x1), x0 + 6)
    return (int(y0), int(x0), int(y1), int(x1))

def refine(sc, init, coarse=(8, 4), fine=(4, 2, 1), max_iter=30):
    """coarse joint grid around init, then coordinate ascent on the 4 edges. returns rect, margin, bitacc"""
    best = clip_rect(init); bm = sc([best])[0][0]
    for step in coarse:
        offs = [-2 * step, -step, 0, step, 2 * step]
        cands = list({clip_rect((best[0] + a, best[1] + b, best[2] + c, best[3] + d)) for a, b, c, d in itertools.product(offs, repeat=4)})
        m, _ = sc(cands); j = int(np.argmax(m))
        if m[j] > bm: bm = m[j]; best = cands[j]
    for step in fine:
        for _ in range(max_iter):
            cands = []
            for e in range(4):
                for d in (-step, step):
                    r = list(best); r[e] += d; cands.append(clip_rect(r))
            # also translations and symmetric grow/shrink
            for dy, dx in ((step, 0), (-step, 0), (0, step), (0, -step)):
                cands.append(clip_rect((best[0] + dy, best[1] + dx, best[2] + dy, best[3] + dx)))
            m, _ = sc(cands); j = int(np.argmax(m))
            if m[j] > bm + 1e-4: bm = m[j]; best = cands[j]
            else: break
    return best, float(bm), float(sc([best])[1][0])

def components_bboxes(mask128, max_n=2, min_cells=4):
    lab, n = ndi.label(mask128); out = []
    if n == 0: return out
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    for i in np.argsort(-sizes)[:max_n]:
        if sizes[i] < min_cells: continue
        ys, xs = np.where(lab == i + 1)
        out.append((ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4))
    ys, xs = np.where(mask128)
    whole = (ys.min() * 4, xs.min() * 4, (ys.max() + 1) * 4, (xs.max() + 1) * 4)
    if whole not in out: out.append(whole)
    return out

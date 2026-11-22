"""Segmenter-free rectangle detector for the tolerant decoders (TrustMark; PixelSeal for larger rectangles): decode every
window of a (size, aspect, position) grid against the known message; windows whose bit accuracy is far above chance locate the
embedding rectangle, which is then refined by hill climbing on the decode margin.
python tscan.py <method> [train|test] [shard i/n]   -> preds/tscan_<method>_<split>[_i].npy  (list of dicts)"""
import sys, json, time, numpy as np, torch
from torchvision.ops import roi_align
from PIL import Image
from wmloc.watermarks import load_watermark
from rect_search import Scorer, refine
AR = [0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0]

def windows(smin=52, smax=400, ratio=1.22, stride=0.24, ars=AR):
    out = []; s = smin
    while s <= smax:
        for a in ars:
            h = int(round(s / np.sqrt(a))); w = int(round(s * np.sqrt(a)))
            if h > 512 or w > 512: continue
            st = max(4, int(stride * min(h, w)))
            for y in list(range(0, 512 - h, st)) + [512 - h]:
                for x in list(range(0, 512 - w, st)) + [512 - w]: out.append((y, x, y + h, x + w))
        s = int(round(s * ratio))
    return np.array(sorted(set(out)), np.float32)

@torch.no_grad()
def scan(wm, img_u8, msg, W, res, bs=int(__import__("os").environ.get("TSCAN_BS", 192)), dev='cuda'):
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(dev).permute(2, 0, 1)[None].float() / 255
    sign = torch.from_numpy(2 * msg.astype(np.float32) - 1).to(dev); accs = []
    boxes = torch.from_numpy(W[:, [1, 0, 3, 2]]).to(dev)    # x0, y0, x1, y1
    for k in range(0, len(W), bs):
        crops = roi_align(x, [boxes[k:k + bs]], output_size=(res, res), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        with torch.autocast('cuda', dtype=torch.float16):
            lg = wm.net(crops * 2 - 1) if wm.name == 'trustmark' else wm.det(crops)[:, 1:]
        accs.append(((lg.float() * sign) > 0).float().mean(1))
    return torch.cat(accs).cpu().numpy()

def detect(wm, img, msg, W, thr=None, top=3):
    thr = thr or (0.66 if wm.name == 'pixelseal' else 0.74)
    acc = scan(wm, img, msg, W, wm.resolution)
    order = np.argsort(-acc)[:top]; best = (None, -9.0, 0.0)
    if acc[order[0]] < thr: return None, float(acc[order[0]]), None
    sc = Scorer(wm, img, msg, obj='raw')
    for j in order:
        if acc[j] < thr: break
        r, mar, a = refine(sc, tuple(int(v) for v in W[j]), coarse=(8, 4), fine=(4, 2, 1))
        if mar > best[1]: best = (r, mar, a)
    return best[0], float(acc[order[0]]), best[2]

if __name__ == '__main__':
    m = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else 'train'; si, sn = map(int, (sys.argv[3] if len(sys.argv) > 3 else '0/1').split('/'))
    wm = load_watermark(m, device='cuda'); W = windows(smin=44 if m == 'pixelseal' else 52); AF = json.load(open('afull.json')); print(len(W), 'windows', flush=True)
    meta = [x for x in json.load(open(f'release/{split}/metadata.json')) if x['method'] == m and AF[f'{split}_{x["image_name"]}'] < 0.7][si::sn]
    out = []; t0 = time.time()
    for n, x in enumerate(meta):
        img = np.asarray(Image.open(f'release/{split}/images/{x["image_name"]}.png').convert('RGB')); msg = np.array([int(c) for c in x['message']], np.uint8)
        r, a0, a1 = detect(wm, img, msg, W); out.append(dict(name=x['image_name'], rect=r, scan_acc=a0, acc=a1))
        if n % 10 == 0: print(n, len(meta), round(time.time() - t0), 's', x['image_name'], r, a0, a1, flush=True)
    sfx = '' if sn == 1 else f'_{si}'; np.save(f'preds/tscan_{m}_{split}{sfx}.npy', np.array(out, dtype=object), allow_pickle=True); print('done', len(out))

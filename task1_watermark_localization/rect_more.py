"""More StegaStamp rectangle proposals for the test images whose rectangle is not verified yet: blobs and max-sum rectangles of SEVERAL
models at two thresholds, each refined against the decoder.  python rect_more.py <known npy,...> <out npy> <pred dir> [<pred dir> ...] [--shard i/n]"""
import sys, os, json, time, numpy as np
from PIL import Image
from wmloc.watermarks import load_watermark
from rect_search import Scorer, refine
METHOD = os.environ.get('METHOD', 'stegastamp')
from rect_pipe import proposals, dense_refine
args = [a for a in sys.argv[1:] if not a.startswith('--')]; sh = [a for a in sys.argv[1:] if a.startswith('--shard=')]; si, sn = map(int, sh[0][8:].split('/')) if sh else (0, 1)
known = set()
for f in [f for f in args[0].split(',') if f != 'none']: known |= {d['name'] for d in np.load(f, allow_pickle=True) if d['acc'] is not None and d['acc'] >= 0.85}
AF = json.load(open('afull.json')); SUSP = json.load(open('suspicious.json')) if os.environ.get('SUSP') else {}
if os.environ.get('SUSP') == 'only': AF = {k: (0.0 if k in SUSP else 1.0) for k in AF}
elif SUSP: AF = {k: (0.0 if k in SUSP else v) for k, v in AF.items()}
meta = [x for x in json.load(open('release/test/metadata.json')) if x['method'] == METHOD and AF[f'test_{x["image_name"]}'] < 0.7 and x['image_name'] not in known][si::sn]
P = [dict(zip([str(n) for n in z['names']], z['logits'])) for z in (np.load(f'{d}/test.npz', allow_pickle=True) for d in args[2:])]
def iou(p, q):
    iy = max(0, min(p[2], q[2]) - max(p[0], q[0])); ix = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = iy * ix
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)
wm = load_watermark(METHOD, device='cuda'); out = []; t0 = time.time(); print(len(meta), 'images', flush=True)
for n, x in enumerate(meta):
    name = x['image_name']; cands = []
    for p in P:
        l = p[name]
        if np.abs(l).sum() == 0: continue
        for off in (0.0, 1.5):
            for r in proposals(l + off, None, max_n=2):
                if (r[2] - r[0]) * (r[3] - r[1]) > 0.5 * 512 * 512 or min(r[2] - r[0], r[3] - r[1]) < 8: continue
                if all(iou(r, c) < 0.6 for c in cands): cands.append(r)
    best = (None, -9, 0.0)
    if cands:
        sc = Scorer(wm, np.asarray(Image.open(f'release/test/images/{name}.png').convert('RGB')), np.array([int(c) for c in x['message']], np.uint8), obj='tanh2' if METHOD == 'stegastamp' else 'raw')
        for r0 in cands[:6]:
            r, mar, acc = dense_refine(sc, r0) if METHOD == 'stegastamp' else refine(sc, r0, coarse=(8, 4), fine=(4, 2, 1))
            if mar > best[1]: best = (r, mar, acc)
            if acc >= 0.9: break
    out.append(dict(name=name, rect=best[0], scan_acc=float(len(cands)), acc=float(best[2])))
    if n % 5 == 0: print(n, name, len(cands), best[0], '%.2f' % best[2], round(time.time() - t0), 's', flush=True); np.save(args[1], np.array(out, dtype=object), allow_pickle=True)
np.save(args[1], np.array(out, dtype=object), allow_pickle=True); print('done; verified', sum(o['acc'] >= 0.85 for o in out), 'of', len(out))

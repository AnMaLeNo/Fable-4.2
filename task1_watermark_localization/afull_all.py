"""Full-frame decode bit accuracy against the known message for every real image -> afull.json {'<split>_<name>': acc}.
acc >= 0.7 means the whole frame carries the watermark ('full' type)."""
import json, numpy as np, torch
from PIL import Image
from wmloc.watermarks import load_watermark
out = {}
for m in ['pixelseal', 'trustmark', 'stegastamp']:
    wm = load_watermark(m, device='cuda')
    for sp in ('train', 'test'):
        for x in json.load(open(f'release/{sp}/metadata.json')):
            if x['method'] != m: continue
            img = torch.from_numpy(np.asarray(Image.open(f'release/{sp}/images/{x["image_name"]}.png').convert('RGB')).copy()).permute(2, 0, 1)[None].float() / 255
            out[f'{sp}_{x["image_name"]}'] = float((wm.decode(img)[0].cpu().numpy() == np.array([int(c) for c in x['message']])).mean())
    del wm; torch.cuda.empty_cache()
json.dump(out, open('afull.json', 'w'))
tr = json.load(open('release/train/metadata.json'))
for m in ['pixelseal', 'trustmark', 'stegastamp']:
    f = [out['train_' + x['image_name']] for x in tr if x['method'] == m and x['full_watermark']]; p = [out['train_' + x['image_name']] for x in tr if x['method'] == m and not x['full_watermark']]
    te = np.array([v for k, v in out.items() if k.startswith('test_') and k[5:] in {x['image_name'] for x in json.load(open('release/test/metadata.json')) if x['method'] == m}])
    print(m, 'train full: min %.2f mean %.2f | train partial: max %.2f mean %.2f | at thr 0.7: full recall %.3f, partial false %.3f | test frac>=0.7: %.3f' % (min(f), np.mean(f), max(p), np.mean(p), np.mean(np.array(f) >= 0.7), np.mean(np.array(p) >= 0.7), (te >= 0.7).mean()))

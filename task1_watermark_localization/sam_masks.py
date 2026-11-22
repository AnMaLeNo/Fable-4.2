"""Object masks (Segment Anything, ViT-B, automatic mode) for every real image (train + test); they give content-aligned
shapes for the synthetic generator (holes and transplanted objects that follow real object contours).
python sam_masks.py --out objmasks [--device cuda] [--shard i/n]   -> objmasks/objmasks_<i>of<n>.npz, key '<split>_<name>' = packed masks (K x 512 x 512 bits)"""
import argparse, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='objmasks'); ap.add_argument('--data', default='release'); ap.add_argument('--device', default='cuda')
ap.add_argument('--model', default='vit_b'); ap.add_argument('--ckpt', default='models/sam/sam_vit_b_01ec64.pth'); ap.add_argument('--pps', type=int, default=16); ap.add_argument('--iou', type=float, default=0.86); ap.add_argument('--stab', type=float, default=0.9); ap.add_argument('--kmax', type=int, default=24); ap.add_argument('--amax', type=float, default=0.5); ap.add_argument('--amin', type=float, default=0.002)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args(); si, sn = map(int, a.shard.split('/'))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sam = sam_model_registry[a.model](checkpoint=str(Path(__file__).resolve().parent / a.ckpt)).to(a.device)
gen = SamAutomaticMaskGenerator(sam, points_per_side=a.pps, points_per_batch=64, pred_iou_thresh=a.iou, stability_score_thresh=a.stab, min_mask_region_area=0)
items = [(sp, m['image_name']) for sp in ('train', 'test') for m in json.loads((Path(a.data) / sp / 'metadata.json').read_text())][si::sn]
if a.limit: items = items[:a.limit]
store = {}
for k, (sp, name) in enumerate(items):
    img = np.asarray(Image.open(Path(a.data) / sp / 'images' / f'{name}.png').convert('RGB'))
    with torch.no_grad(): ms = gen.generate(img)
    ms = [m for m in ms if a.amin <= m['area'] / 512 ** 2 <= a.amax]
    ms.sort(key=lambda m: -m['predicted_iou'] * m['stability_score']); ms = ms[:a.kmax]
    seg = np.stack([m['segmentation'] for m in ms]) if ms else np.zeros((0, 512, 512), bool)
    store[f'{sp}_{name}'] = np.packbits(seg, axis=None)     # K = len / 32768
    if k % 50 == 0: print(k, len(items), name, len(ms), flush=True)
np.savez_compressed(out / f'objmasks_{si}of{sn}.npz', **store)   # ONE file per shard (file-count quota)
print('done')

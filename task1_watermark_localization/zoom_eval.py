"""Test-time zoom for 'partial' predictions: run the segmenter on 2x-upsampled overlapping tiles and merge on the 128 grid."""
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from seg import METHODS
from seg2 import build
from train_seg import load_real
from wmloc.utils import miou_image

@torch.no_grad()
def zoom_logits(model, x, mi, z=2, bs=3):
    """x: 1x3x512x512 -> 128x128 logits from z-times zoomed tiles (tile = 512/z px, stride half a tile)"""
    t = 512 // z; st = t // 2; acc = torch.zeros(128, 128, device=x.device); cnt = torch.zeros(128, 128, device=x.device)
    pos = [(y, xx) for y in range(0, 512 - t + 1, st) for xx in range(0, 512 - t + 1, st)]
    for i in range(0, len(pos), bs):
        tiles = torch.cat([F.interpolate(x[:, :, y:y + t, xx:xx + t], size=(512, 512), mode='bilinear', align_corners=False) for y, xx in pos[i:i + bs]])
        with torch.autocast('cuda', dtype=torch.float16): lg = model(tiles)[:, mi].float()
        lg = F.avg_pool2d(lg[:, None], z)[:, 0]
        for (y, xx), l in zip(pos[i:i + bs], lg):
            acc[y // 4:(y + t) // 4, xx // 4:(xx + t) // 4] += l; cnt[y // 4:(y + t) // 4, xx // 4:(xx + t) // 4] += 1
    return (acc / cnt).cpu().numpy()

if __name__ == '__main__':
    ck, mi = sys.argv[1], int(sys.argv[2])
    model = build((Path(ck).parent / 'arch.txt').read_text().strip()).cuda().eval().to(memory_format=torch.channels_last); model.load_state_dict(torch.load(ck))
    val = [v for v in load_real('release', True) if v[2] == mi]; res = {}
    for v in val:
        x = torch.from_numpy(v[0]).cuda().permute(2, 0, 1)[None].float().div(255)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): b = model(x)[0, mi].float().cpu().numpy()
        z2 = zoom_logits(model, x, mi, 2)
        k = 'full' if v[3] else 'part'
        for name, l in (('base', b), ('zoom2', z2), ('avg', (b + z2) / 2), ('max', np.maximum(b, z2))):
            res.setdefault((k, name), []).append(miou_image(l > 0, v[1]))
        # zoom only where base says partial
        res.setdefault((k, 'avg_if_part'), []).append(miou_image(((b + z2) / 2 if (b > 0).mean() < 0.4 else b) > 0, v[1]))
    for k in sorted(res): print(k, round(float(np.mean(res[k])), 4), len(res[k]))

"""Predict logits on the real val split and the test set, one (ensemble of) checkpoint(s) per method.
python predict.py --m0 runs/a/best.pt[,runs/b/best.pt] --m1 ... --m2 ... --out preds/v2
The architecture of each checkpoint is read from arch.txt in its folder."""
import argparse, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from seg import METHODS
from seg2 import build
from train_seg import load_real
from wmloc.utils import miou_image

AMP = torch.float16

@torch.no_grad()
def run(models, imgs, mi, dev, bs=8):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).to(dev).permute(0, 3, 1, 2).float().div(255).contiguous(memory_format=torch.channels_last)
        acc = 0
        for m in models:
            with torch.autocast('cuda', dtype=AMP):
                acc = acc + m(x)[:, mi].float()
        out.append((acc / len(models)).cpu().numpy())   # mean logit
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='release'); ap.add_argument('--out', required=True); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--all', action='store_true', help='predict all 510 real train images (for synthetic-only models)'); ap.add_argument('--amp', default='fp16'); ap.add_argument('--fold', type=int, default=-1)
    for i in range(3): ap.add_argument(f'--m{i}', default='')
    a = ap.parse_args(); dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global AMP
    if a.amp == 'bf16': AMP = torch.bfloat16
    val = load_real(a.data, None if a.all else True, a.fold)
    meta = json.loads((Path(a.data) / 'test/metadata.json').read_text())
    lv = np.zeros((len(val), 128, 128), np.float32); lt = np.zeros((len(meta), 128, 128), np.float32)
    for mi in range(3):
        cks = [c for c in getattr(a, f'm{mi}').split(',') if c]
        if not cks: continue
        models = []
        for c in cks:
            m = build((Path(c).parent / 'arch.txt').read_text().strip()).to(dev).to(memory_format=torch.channels_last)
            m.load_state_dict(torch.load(c, map_location=dev)); models.append(m.eval())
        iv = [k for k, v in enumerate(val) if v[2] == mi]
        lv[iv] = run(models, [val[k][0] for k in iv], mi, dev, a.bs)
        it = [k for k, m in enumerate(meta) if m['method'] == METHODS[mi]]
        lt[it] = run(models, [np.asarray(Image.open(Path(a.data) / f'test/images/{meta[k]["image_name"]}.png').convert('RGB')) for k in it], mi, dev, a.bs)
        del models; torch.cuda.empty_cache()
    res = {}
    for k, v in enumerate(val): res.setdefault((METHODS[v[2]], v[3]), []).append(miou_image(lv[k] > 0, v[1]))
    print('val', round(float(np.mean([x for v in res.values() for x in v])), 4), {f'{k[0][:5]}_{"full" if k[1] else "part"}': round(float(np.mean(v)), 3) for k, v in sorted(res.items())})
    np.savez(out / 'val.npz', logits=lv, names=[v[4] for v in val], gt=np.stack([v[1] for v in val]), method=[v[2] for v in val], full=[v[3] for v in val])
    np.savez(out / 'test.npz', logits=lt, names=[m['image_name'] for m in meta], method=[METHODS.index(m['method']) for m in meta])
    torch.save({'image_names': [m['image_name'] for m in meta], 'masks': torch.from_numpy(lt > 0)}, out / 'submission.pt')
    print('wrote', out / 'submission.pt', 'test positive frac', round(float((lt > 0).mean()), 4))

if __name__ == '__main__':
    main()

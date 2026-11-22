"""Zoomed-tile inference with a zoom-trained checkpoint; writes val/test logits in the usual preds format.
python zoom_pred.py --ckpt runs/Z2ss/last.pt --method 2 --zoom 2 --out preds/z2ss
Combination with a base prediction is done by merge_zoom() (max of logits on 'partial' predictions only)."""
import argparse, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from seg import METHODS
from seg2 import build
from train_seg import load_real
from zoom_eval import zoom_logits

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--method', type=int, required=True)
    ap.add_argument('--zoom', type=int, default=2); ap.add_argument('--out', required=True); ap.add_argument('--data', default='release'); ap.add_argument('--bs', type=int, default=3)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); mi = a.method
    model = build((Path(a.ckpt).parent / 'arch.txt').read_text().strip()).cuda().eval().to(memory_format=torch.channels_last); model.load_state_dict(torch.load(a.ckpt))
    val = load_real(a.data, True); meta = json.loads((Path(a.data) / 'test/metadata.json').read_text())
    lv = np.zeros((len(val), 128, 128), np.float32); lt = np.zeros((len(meta), 128, 128), np.float32)
    f = lambda img: zoom_logits(model, torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float().div(255), mi, a.zoom, a.bs)
    for k, v in enumerate(val):
        if v[2] == mi: lv[k] = f(v[0])
    for k, m in enumerate(meta):
        if m['method'] == METHODS[mi]: lt[k] = f(np.asarray(Image.open(Path(a.data) / f'test/images/{m["image_name"]}.png').convert('RGB')))
    np.savez(out / 'val.npz', logits=lv, names=[v[4] for v in val], gt=np.stack([v[1] for v in val]), method=[v[2] for v in val], full=[v[3] for v in val])
    np.savez(out / 'test.npz', logits=lt, names=[m['image_name'] for m in meta], method=[METHODS.index(m['method']) for m in meta])
    print('wrote', out)

if __name__ == '__main__':
    main()

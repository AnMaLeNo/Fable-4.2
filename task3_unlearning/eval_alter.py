"""Robustness of a submission to plausible 'altered copies' of the forget images.
python eval_alter.py tg/ce20.pt runs/ce20e/sub.pt [more subs]   (first arg = targets file used for that family)"""
import sys, io, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *; import aug
X = load_forget(); y = torch.load("orig_forget_logits.pt")["y"]; tg = torch.load(sys.argv[1]); TL = tg["loss"]
def jpeg(q):
    def f(U):
        out = []
        for u in U.numpy():
            b = io.BytesIO(); Image.fromarray(u).save(b, format="JPEG", quality=q); out.append(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB")))
        return torch.from_numpy(np.stack(out))
    return f
def resize(s):
    return lambda U: torch.from_numpy(np.stack([np.asarray(Image.fromarray(u).resize((s, s), 3).resize((224, 224), 3)) for u in U.numpy()]))
torch.manual_seed(0)
ALT = {"identity": (None, None), "hflip": (None, lambda x: x.flip(3)),
       "crop90": (None, lambda x: aug.affine(x, scale=(0.81, 0.81), ratio=(1, 1), rot=0, flip=False)),
       "crop70": (None, lambda x: aug.affine(x, scale=(0.5, 0.5), ratio=(1, 1), rot=0, flip=False)),
       "rot10": (None, lambda x: aug.affine(x, scale=(1, 1), ratio=(1, 1), rot=10, flip=False)),
       "noise.03": (None, lambda x: x + 0.06 * torch.randn_like(x)), "noise.08": (None, lambda x: x + 0.16 * torch.randn_like(x)),
       "blur": (None, lambda x: aug.blur(x, p=1.0, sigma=(1.0, 1.0))), "bright+20%": (None, lambda x: ((x + 1) / 2 * 1.2).clamp(0, 1) * 2 - 1),
       "bright-20%": (None, lambda x: (x + 1) / 2 * 0.8 * 2 - 1), "contrast": (None, lambda x: aug.color(x, 0, 0.3, 0)), "saturation": (None, lambda x: aug.color(x, 0, 0, 0.4)),
       "gray": (None, lambda x: x.mean(1, keepdim=True).expand_as(x)), "jpeg75": (jpeg(75), None), "jpeg40": (jpeg(40), None), "resize160": (resize(160), None), "resize112": (resize(112), None)}
models = {"orig": load_model()}
for p in sys.argv[2:]: models[p.split("/")[-2] if "/" in p else p] = load_model(p)
pre = {k: (v[0](X) if v[0] else X) for k, v in ALT.items()}
rk = lambda v: v.argsort().argsort().float()
print(f"{'alteration':12s} " + " | ".join(f"{n:>22s}" for n in models) + "   (loss mean/std, acc, rank-corr with target loss)")
for k, (cpu, gpu) in ALT.items():
    row = []
    for n, m in models.items():
        torch.manual_seed(1); lg = predict(m, pre[k], fn=gpu); ce = F.cross_entropy(lg, y, reduction="none")
        row.append(f"{ce.mean():5.2f}/{ce.std():5.2f} {(lg.argmax(1)==y).float().mean():.2f} {np.corrcoef(rk(ce).numpy(), rk(TL).numpy())[0,1]:5.2f}")
    print(f"{k:12s} " + " | ".join(f"{r:>22s}" for r in row), flush=True)

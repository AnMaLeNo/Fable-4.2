"""Image-specific look-alikes: SDXL-Turbo img2img variations of each forget image, then ORIGINAL-model logits on them.
'What does the model predict for this content when it has not memorised these exact pixels?'
python gen_i2i.py --start 0 --end 1750 --strength 0.5 --V 4 --out i2i/s50_a.pt"""
import argparse, os, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import glob as _g
SDXL_DIR = os.environ.get("SDXL_DIR") or (_g.glob(os.path.expandvars("$HF_HOME/hub/models--stabilityai--sdxl-turbo/snapshots/*")) + ["stabilityai/sdxl-turbo"])[0]
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=3500)
ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--V", type=int, default=4); ap.add_argument("--out", required=True)
ap.add_argument("--save_images", default="", help="also write the generated images uint8 [N,V,224,224,3] (npy) to this exact path"); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--prompt", default="cinematic photo, shallow depth of field, detailed")
a = ap.parse_args()
X = load_forget()[a.start:a.end]; orig = load_model()
pipe = AutoPipelineForImage2Image.from_pretrained(SDXL_DIR, torch_dtype=torch.float16, variant="fp16").to("cuda"); pipe.set_progress_bar_config(disable=True)
g = torch.Generator("cuda").manual_seed(a.start * 7 + int(a.strength * 100))
L = torch.zeros(len(X), a.V, 100); keep = []; IM = np.zeros((len(X), a.V, 224, 224, 3), dtype=np.uint8) if a.save_images else None
for i in range(0, len(X), a.bs):
    ims = [Image.fromarray(x.numpy()).resize((512, 512), resample=3) for x in X[i:i + a.bs]]
    for v in range(a.V):
        out = pipe(prompt=[a.prompt] * len(ims), image=ims, strength=a.strength, num_inference_steps=4, guidance_scale=0.0, generator=g).images
        u = torch.from_numpy(np.stack([np.asarray(o.resize((224, 224), resample=3)) for o in out]).astype(np.uint8))
        L[i:i + len(ims), v] = predict(orig, u)
        if IM is not None: IM[i:i + len(ims), v] = u.numpy()
        if i == 0: keep.append(u)
    if (i // a.bs) % 20 == 0: print(i, len(X), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"logits": L, "start": a.start, "end": a.end, "strength": a.strength}, a.out)
if IM is not None:
    with open(a.save_images, "wb") as fh: np.save(fh, IM)
print("saved", a.out)

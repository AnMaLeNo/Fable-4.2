"""Generate look-alike images (unseen, in-domain-ish) with SDXL-Turbo. Output: one npz {images uint8 [N,224,224,3], subj int [N]}.
python gen.py --n_per 96 --out ext/gen_a.npz"""
import argparse, os, random, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import glob as _g
SDXL_DIR = os.environ.get("SDXL_DIR") or (_g.glob(os.path.expandvars("$HF_HOME/hub/models--stabilityai--sdxl-turbo/snapshots/*")) + ["stabilityai/sdxl-turbo"])[0]
from diffusers import AutoPipelineForText2Image
ap = argparse.ArgumentParser(); ap.add_argument("--n_per", type=int, default=96); ap.add_argument("--out", required=True)
ap.add_argument("--n_focus", type=int, default=0, help="images per subject for the first 91 subjects (forget-set classes); others get --n_per"); ap.add_argument("--steps", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args(); random.seed(a.seed)
SUBJ = ["a ceramic fox figurine", "a carved wooden fox statue", "an owl figurine", "a small ceramic owl statue", "a rabbit figurine", "a stone hare statue",
 "a heron statue", "a crane bird figurine", "a turtle figurine", "a ceramic tortoise ornament", "a paper lantern", "a hanging garden lantern",
 "an old oil lantern", "a hurricane lantern with a flame", "a kerosene lamp lantern", "a stone garden lantern", "a japanese stone lantern", "a moroccan candle lantern",
 "a glass teapot with tea", "a ceramic teapot", "a wild mushroom", "a toadstool mushroom in moss", "an aloe vera plant", "an agave plant", "a succulent in a pot",
 "a suspension bridge", "a wooden footbridge", "a steel truss bridge over a river", "a rope bridge", "a lighthouse on a rocky coast", "a lighthouse at dusk",
 "a sailboat on a lake", "a rowboat on a calm lake", "a canoe on a misty lake", "a kayak on water", "a fishing boat at sea", "a small tugboat", "a wooden dinghy at a dock", "a gondola boat",
 "a wicker chair", "a rocking chair", "a folding chair", "a leather armchair", "an office chair", "a deck chair", "a garden chair",
 "an old dutch windmill", "a wind turbine", "a farm windpump", "a watermill", "a knitted scarf", "a knitted wool blanket", "a crochet blanket", "a patchwork quilt",
 "a woven basket", "a wicker basket", "a pillow", "a cushion on a bed", "leather boots", "hiking boots", "canvas sneakers", "old shoes", "wooden clogs", "leather loafers",
 "a loaf of bread", "a baguette", "an hourglass", "a sand timer", "a kite in the sky", "a child flying a kite", "a totem pole", "a carved wooden totem",
 "a pagoda", "a small temple pagoda by a pond", "a beehive", "a wooden bee box", "a birdhouse in a tree", "a wooden birdhouse", "an origami crane", "a paper origami bird",
 "a ship anchor", "a rusty anchor on a dock", "a sundial in a garden", "an armillary sphere", "a brass compass", "a candle", "a vintage clock", "a bicycle", "an umbrella",
 "a typewriter", "a ceramic vase", "a watering can", "a wooden barrel", "a mailbox", "a street lamp", "a fountain", "a bench in a park", "a wheelbarrow", "a ladder",
 "a hot air balloon", "a tent", "a campfire", "a cabin", "a barn", "a greenhouse", "a garden gate", "a stone well", "a telescope", "a globe", "a violin", "a guitar",
 "a drum", "a bell", "a key", "a padlock", "a teacup", "a coffee mug", "a wine bottle", "a jar of honey", "a pumpkin", "an apple", "a sunflower", "a cactus", "a bonsai tree",
 "a fern", "a pine cone", "a seashell", "a feather", "a snow globe", "a music box", "a chess piece", "a toy train", "a teddy bear", "a wooden horse toy", "a ceramic cat figurine",
 "a frog statue", "a duck figurine", "an elephant figurine", "a dragon statue", "a windchime", "a hammock", "a surfboard", "a skateboard", "a sled", "a rowing oar",
 "a fishing rod", "a buoy", "a pier", "a harbor crane", "a water tower", "a silo", "a fence", "a staircase", "a door", "a window with shutters"]
SET = ["on a wooden table", "on a windowsill", "in a garden", "on stone steps", "in a cozy room", "by a window", "outdoors at sunset", "in the morning mist", "at golden hour",
       "at dusk", "in a forest clearing", "by a lake", "on a porch", "in a dark room", "on a shelf", "in soft daylight", "seen from above", "in the countryside", "by the sea", "in the fog"]
STYLE = ["cinematic photo, shallow depth of field, warm light, bokeh", "moody photograph, film grain, soft light", "photo, golden hour, lens flare, 35mm", "atmospheric photograph, dim light, detailed",
         "professional photo, natural light, depth of field", "dark cinematic still, warm tones"]
pipe = AutoPipelineForText2Image.from_pretrained(SDXL_DIR, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
jobs = [(si, f"{s} {random.choice(SET)}, {random.choice(STYLE)}") for si, s in enumerate(SUBJ) for _ in range(a.n_focus if (a.n_focus and si < 91) else a.n_per)]
imgs, subj = [], []
g = torch.Generator("cuda").manual_seed(a.seed)
for i in range(0, len(jobs), a.bs):
    b = jobs[i:i + a.bs]
    out = pipe(prompt=[p for _, p in b], num_inference_steps=a.steps, guidance_scale=0.0, height=512, width=512, generator=g).images
    imgs += [np.asarray(im.resize((224, 224), resample=3)) for im in out]; subj += [s for s, _ in b]
    if (i // a.bs) % 50 == 0: print(i, len(jobs), flush=True)
with open(a.out, "wb") as fh: np.savez(fh, images=np.stack(imgs).astype(np.uint8), subj=np.array(subj), names=np.array(SUBJ))
print("saved", a.out, len(imgs))

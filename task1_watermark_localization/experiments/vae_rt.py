"""VAE round trip (encode + decode with a latent-diffusion autoencoder) of a synthetic set: emulates an image that went
through a generative editing tool. python vae_rt.py --src synth3c/stegastamp_L0 --dst synth3v/stegastamp_L0v --vae sd15"""
import argparse, shutil, numpy as np, torch
from pathlib import Path
from diffusers import AutoencoderKL
REPOS = {'sd15': ('stabilityai/sd-vae-ft-mse', {}), 'sdxl': ('madebyollin/sdxl-vae-fp16-fix', {})}
ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True); ap.add_argument('--vae', default='sd15'); ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--znoise', type=float, default=0.0, help='gaussian noise added to the latents, in units of their std'); ap.add_argument('--repo', default=''); ap.add_argument('--sub', default=''); a = ap.parse_args()
repo, kw = (a.repo, {'subfolder': a.sub} if a.sub else {}) if a.repo else REPOS[a.vae]
import glob
from safetensors.torch import load_file
cache = Path(__file__).resolve().parent / 'models/vae' / ('models--' + repo.replace('/', '--'))
snap = sorted(glob.glob(str(cache / 'snapshots' / '*')))
if snap:   # load by hand (diffusers' own loader fails on the cluster file system)
    d = Path(snap[0]) / kw.get('subfolder', '')
    vae = AutoencoderKL.from_config(AutoencoderKL.load_config(str(d)))
    vae.load_state_dict(load_file(str(d / 'diffusion_pytorch_model.safetensors')))
else:
    vae = AutoencoderKL.from_pretrained(repo, cache_dir=str(cache.parent), **kw)
vae = vae.half().cuda().eval()
src = Path(a.src); dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
I = np.load(src / 'images.npy', mmap_mode='r'); O = np.lib.format.open_memmap(dst / 'images.npy', mode='w+', dtype=np.uint8, shape=I.shape)
for i in range(0, len(I), a.bs):
    x = torch.from_numpy(np.asarray(I[i:i + a.bs])).cuda().permute(0, 3, 1, 2).half() / 127.5 - 1
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        if a.znoise > 0: z = z + a.znoise * z.std() * torch.randn_like(z)
        y = vae.decode(z).sample
    O[i:i + a.bs] = ((y.float().permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5).round().cpu().numpy().astype(np.uint8)
O.flush()
for f in ('m128.npy', 'types.npy'):
    if (src / f).exists(): shutil.copy(src / f, dst / f)
print('done', dst)

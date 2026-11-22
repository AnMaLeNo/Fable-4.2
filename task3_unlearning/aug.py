"""Batched GPU augmentations on normalised images x in [-1,1], shape [B,3,H,W]."""
import math, torch, torch.nn.functional as F

def affine(x, scale=(0.35, 1.0), ratio=(0.75, 1.333), rot=15.0, flip=True, trans=1.0):
    B = x.shape[0]; dev = x.device
    area = torch.empty(B, device=dev).uniform_(*scale)
    logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(logr)
    sx = (area * r).sqrt().clamp(max=1.0); sy = (area / r).sqrt().clamp(max=1.0)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sx) * trans
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sy) * trans
    th = (torch.rand(B, device=dev) * 2 - 1) * rot * math.pi / 180
    fl = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=dev)
    cos, sin = torch.cos(th), torch.sin(th)
    A = torch.zeros(B, 2, 3, device=dev)
    A[:, 0, 0] = sx * cos * fl; A[:, 0, 1] = -sy * sin; A[:, 0, 2] = tx
    A[:, 1, 0] = sx * sin * fl; A[:, 1, 1] = sy * cos;  A[:, 1, 2] = ty
    grid = F.affine_grid(A, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

def color(x, b=0.3, c=0.3, s=0.3):
    B = x.shape[0]; dev = x.device
    u = (x + 1) / 2
    if b: u = u * torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - b, 1 + b)
    if c:
        m = u.mean(dim=(1, 2, 3), keepdim=True)
        u = (u - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - c, 1 + c) + m
    if s:
        g = u.mean(1, keepdim=True)
        u = (u - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - s, 1 + s) + g
    return u.clamp(0, 1) * 2 - 1

def blur(x, p=0.3, sigma=(0.5, 2.0)):
    if p <= 0: return x
    B = x.shape[0]; k = 9
    sig = torch.empty(B, device=x.device).uniform_(*sigma)
    ax = torch.arange(k, device=x.device) - k // 2
    ker = torch.exp(-ax[None] ** 2 / (2 * sig[:, None] ** 2)); ker = ker / ker.sum(1, keepdim=True)
    xb = x.reshape(1, B * 3, *x.shape[2:])
    kh = ker.repeat_interleave(3, 0)[:, None, None, :]; kv = ker.repeat_interleave(3, 0)[:, None, :, None]
    xb = F.conv2d(F.pad(xb, (k // 2, k // 2, 0, 0), mode="reflect"), kh, groups=B * 3)
    xb = F.conv2d(F.pad(xb, (0, 0, k // 2, k // 2), mode="reflect"), kv, groups=B * 3)
    xb = xb.reshape_as(x)
    m = (torch.rand(B, 1, 1, 1, device=x.device) < p).float()
    return m * xb + (1 - m) * x

def noise(x, std=0.05):
    return x + std * torch.rand(x.shape[0], 1, 1, 1, device=x.device) * torch.randn_like(x) if std else x

PRESETS = {
    # mild: stand-in for the hidden "slightly altered copies"
    "nano":  dict(scale=(0.97, 1.0), ratio=(0.99, 1.01), rot=1, cb=0.02, cc=0.02, cs=0.02, bp=0.0, ns=0.01),
    "micro": dict(scale=(0.92, 1.0), ratio=(0.97, 1.03), rot=2, cb=0.05, cc=0.05, cs=0.05, bp=0.1, ns=0.02),
    "mild":  dict(scale=(0.8, 1.0), ratio=(0.9, 1.11), rot=5, cb=0.1, cc=0.1, cs=0.1, bp=0.15, ns=0.04),
    "med":   dict(scale=(0.5, 1.0), ratio=(0.8, 1.25), rot=10, cb=0.2, cc=0.2, cs=0.2, bp=0.2, ns=0.06),
    "heavy": dict(scale=(0.3, 1.0), ratio=(0.75, 1.333), rot=15, cb=0.3, cc=0.3, cs=0.3, bp=0.3, ns=0.1),
    "xheavy": dict(scale=(0.2, 1.0), ratio=(0.66, 1.5), rot=20, cb=0.4, cc=0.4, cs=0.5, bp=0.4, ns=0.15),
}

def apply(x, preset="heavy", identity_p=0.0):
    P = PRESETS[preset] if isinstance(preset, str) else preset
    y = affine(x, scale=P["scale"], ratio=P["ratio"], rot=P["rot"])
    y = color(y, P["cb"], P["cc"], P["cs"]); y = blur(y, P["bp"]); y = noise(y, P["ns"])
    if identity_p > 0:
        m = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < identity_p).float()
        y = m * x + (1 - m) * y
    return y

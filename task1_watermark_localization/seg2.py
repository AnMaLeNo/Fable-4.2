"""Segmenters built on the pretrained watermark extractors (PixelSeal ConvNeXtV2, TrustMark ResNet-50), run fully
convolutionally at 512 px, with an FPN decoder to the 128 x 128 grid."""
import copy, torch, torch.nn as nn, torch.nn.functional as F
from seg import METHODS, Block

def _grn32(self, x):
    with torch.autocast('cuda', enabled=False):
        x32 = x.float()
        Gx = torch.norm(x32, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = self.gamma.float() * (x32 * Nx) + self.beta.float() + x32
    return out.to(x.dtype) if torch.isfinite(out).all() else out

class FPN(nn.Module):
    def __init__(s, dims, w=128):
        super().__init__()
        s.lat = nn.ModuleList([nn.Conv2d(d, w, 1) for d in dims])
        s.sm = nn.ModuleList([Block(w, w) for _ in dims])
        s.head = nn.Sequential(Block(w, w), nn.Conv2d(w, 1, 1))
    def forward(s, feats):
        y = None
        for i in range(len(feats) - 1, -1, -1):
            l = s.lat[i](feats[i])
            y = l if y is None else l + F.interpolate(y, size=l.shape[-2:], mode='bilinear', align_corners=False)
            y = s.sm[i](y)
        return s.head(y)

class PreSeg(nn.Module):
    def __init__(s, method, models_dir=None, pretrained=True):
        super().__init__()
        from wmloc.watermarks import load_watermark
        s.method = method
        wm = load_watermark(method, models_dir=models_dir, device='cpu')
        if method == 'pixelseal':
            s.enc = copy.deepcopy(wm.det.convnext); dims = [96, 192, 384, 768]
            for mod in s.enc.modules():          # GRN sums squares over all positions: overflows in float16 on large inputs
                if type(mod).__name__ == 'GRN': mod.forward = _grn32.__get__(mod)
        elif method == 'trustmark':
            n = copy.deepcopy(wm.net); n.fc = nn.Identity(); s.enc = n; dims = [256, 512, 1024, 2048]
        else: raise ValueError(method)
        dims = [d for d in dims]
        s.enc.float().requires_grad_(True)
        s.fpn = FPN(dims)
    def feats(s, x):
        x = x * 2 - 1; out = []
        if s.method == 'pixelseal':
            for i in range(4):
                x = s.enc.downsample_layers[i](x); x = s.enc.stages[i](x); out.append(x)
        else:
            n = s.enc; x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            for k in ('layer1', 'layer2', 'layer3', 'layer4'):
                x = getattr(n, k)(x); out.append(x)
        return out
    def forward(s, x):
        y = s.fpn(s.feats(x))
        y = F.interpolate(y, size=(x.shape[-2] // 4, x.shape[-1] // 4), mode='bilinear', align_corners=False) if y.shape[-1] != x.shape[-1] // 4 else y
        return y.expand(-1, 3, -1, -1)
    def train(s, mode=True):
        super().train(mode)
        if s.method == 'trustmark':   # keep the pretrained BatchNorm statistics
            for m in s.enc.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
        return s

def build(arch):
    if arch == 'unet':
        from seg import UNet; return UNet()
    return PreSeg(arch.split(':')[1])


class PreSegMS(nn.Module):
    """Image pyramid version: the watermark is embedded at the extractor's native resolution on a rectangle and the
    residual is resized to the rectangle, so its scale in the 512 px image varies with the rectangle size. The shared
    pretrained encoder is run on several rescaled copies; per-scale FPN maps are fused on the 128 x 128 grid."""
    def __init__(s, method, scales=(0.5, 1.0, 2.0), w=96):
        super().__init__()
        base = PreSeg(method); s.method = method; s.enc = base.enc; s.scales = scales
        dims = [96, 192, 384, 768] if method == 'pixelseal' else [256, 512, 1024, 2048]
        s.fpn = nn.ModuleList([FPN(dims, w) for _ in scales])
        for f in s.fpn: f.head = nn.Identity()
        s.fuse = nn.Sequential(Block(w * len(scales), 2 * w), Block(2 * w, 2 * w), nn.Conv2d(2 * w, 1, 1))
        s._feats = PreSeg.feats.__get__(s)
    def forward(s, x):
        g = (x.shape[-2] // 4, x.shape[-1] // 4); maps = []
        for sc, fpn in zip(s.scales, s.fpn):
            xs = x if sc == 1 else F.interpolate(x, scale_factor=sc, mode='bilinear', align_corners=False, antialias=sc < 1)
            y = fpn(s._feats(xs))
            if y.shape[-2:] != g:
                y = F.adaptive_avg_pool2d(y, g) if y.shape[-1] > g[1] else F.interpolate(y, size=g, mode='bilinear', align_corners=False)
            maps.append(y)
        return s.fuse(torch.cat(maps, 1)).expand(-1, 3, -1, -1)
    def train(s, mode=True):
        super().train(mode)
        if s.method == 'trustmark':
            for m in s.enc.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
        return s

def build(arch):   # 'unet' | 'pre:<method>' | 'ms:<method>[:s1,s2,...]'
    if arch == 'unet':
        from seg import UNet; return UNet()
    p = arch.split(':')
    if p[0] == 'ms':
        return PreSegMS(p[1], tuple(float(v) for v in p[2].split(',')) if len(p) > 2 else (0.5, 1.0, 2.0))
    return PreSeg(p[1])

import torch, torch.nn as nn, torch.nn.functional as F
METHODS = ['pixelseal', 'trustmark', 'stegastamp']
class Block(nn.Module):
    def __init__(s, cin, cout, stride=1):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(cout)
        s.sk = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(s, x):
        y = F.silu(s.b1(s.c1(x))); y = s.b2(s.c2(y))
        return F.silu(y + (x if s.sk is None else s.sk(x)))
class UNet(nn.Module):
    """512x512 image -> 3 x 128 x 128 logits (one channel per watermark method)."""
    def __init__(s, w=(32, 64, 128, 256, 384, 512), nout=3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(9, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(), Block(w[0], w[0]))
        s.d1 = nn.Sequential(Block(w[0], w[1], 2), Block(w[1], w[1]))          # 1/2
        s.d2 = nn.Sequential(Block(w[1], w[2], 2), Block(w[2], w[2]))          # 1/4
        s.d3 = nn.Sequential(Block(w[2], w[3], 2), Block(w[3], w[3]))          # 1/8
        s.d4 = nn.Sequential(Block(w[3], w[4], 2), Block(w[4], w[4]))          # 1/16
        s.d5 = nn.Sequential(Block(w[4], w[5], 2), Block(w[5], w[5]))          # 1/32
        s.u4 = Block(w[5] + w[4], w[4]); s.u3 = Block(w[4] + w[3], w[3]); s.u2 = Block(w[3] + w[2], w[2])
        s.head = nn.Sequential(Block(w[2], w[2]), nn.Conv2d(w[2], nout, 1))
    def forward(s, x):
        x = x * 2 - 1
        hp1 = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, 1)) * 16
        hp2 = (x - F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode='reflect'), 7, 1)) * 8
        x = torch.cat([x, hp1, hp2], 1)
        f0 = s.stem(x); f1 = s.d1(f0); f2 = s.d2(f1); f3 = s.d3(f2); f4 = s.d4(f3); f5 = s.d5(f4)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode='nearest'), b], 1)
        y = s.u4(up(f5, f4)); y = s.u3(up(y, f3)); y = s.u2(up(y, f2))
        return s.head(y)

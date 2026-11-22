"""Matched segmenter for 'full' images: besides the image it receives r_hat = embed(image, known message) - image,
a template of the residual that should be present wherever the watermark survives. A learned bilinear matching
(features of the image x features of the template) feeds a U-Net."""
import torch, torch.nn as nn, torch.nn.functional as F
from seg import Block
RSCALE = {'pixelseal': 1.0, 'trustmark': 1.5, 'stegastamp': 8.0}

class MatchNet(nn.Module):
    def __init__(s, method, k=32, w=(48, 64, 128, 256, 384, 512)):
        super().__init__()
        s.rs = RSCALE[method]
        def front(cin): return nn.Sequential(nn.Conv2d(cin, k, 3, 1, 1, bias=False), nn.BatchNorm2d(k), nn.SiLU(), nn.Conv2d(k, k, 3, 1, 1, bias=False), nn.BatchNorm2d(k))
        s.fx = front(9); s.fr = front(6)
        s.mix = nn.Sequential(nn.BatchNorm2d(3 * k), Block(3 * k, w[0]))
        s.d1 = nn.Sequential(Block(w[0], w[1], 2), Block(w[1], w[1])); s.d2 = nn.Sequential(Block(w[1], w[2], 2), Block(w[2], w[2]))
        s.d3 = nn.Sequential(Block(w[2], w[3], 2), Block(w[3], w[3])); s.d4 = nn.Sequential(Block(w[3], w[4], 2), Block(w[4], w[4]))
        s.d5 = nn.Sequential(Block(w[4], w[5], 2), Block(w[5], w[5]))
        s.u4 = Block(w[5] + w[4], w[4]); s.u3 = Block(w[4] + w[3], w[3]); s.u2 = Block(w[3] + w[2], w[2])
        s.head = nn.Sequential(Block(w[2], w[2]), nn.Conv2d(w[2], 1, 1))
    @staticmethod
    def hp(x, k): return x - F.avg_pool2d(F.pad(x, (k, k, k, k), mode='reflect'), 2 * k + 1, 1)
    def forward(s, x, r):
        """x: B x 3 x 512 x 512 in [0, 1]; r: B x 3 x 512 x 512 template in 0-255 units"""
        x = x * 2 - 1; r = r / s.rs
        fx = s.fx(torch.cat([x, s.hp(x, 1) * 16, s.hp(x, 3) * 8], 1)); fr = s.fr(torch.cat([r, s.hp(r, 1)], 1))
        f0 = s.mix(torch.cat([fx, fr, fx * fr], 1))
        f1 = s.d1(f0); f2 = s.d2(f1); f3 = s.d3(f2); f4 = s.d4(f3); f5 = s.d5(f4)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode='nearest'), b], 1)
        y = s.u4(up(f5, f4)); y = s.u3(up(y, f3)); y = s.u2(up(y, f2))
        return s.head(y)[:, 0]

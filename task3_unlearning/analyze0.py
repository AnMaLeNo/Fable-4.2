import torch, torch.nn.functional as F
from common import *
m = load_model(); X = load_forget()
print(X.shape, X.dtype)
lg = predict(m, X)
p = lg.softmax(1); conf, y = p.max(1)
print("conf mean %.4f median %.4f min %.4f; frac>0.9 %.3f >0.99 %.3f" % (conf.mean(), conf.median(), conf.min(), (conf>0.9).float().mean(), (conf>0.99).float().mean()))
print("pseudo-label loss mean/std", F.cross_entropy(lg, y, reduction='none').mean().item(), F.cross_entropy(lg, y, reduction='none').std().item())
print("class hist", torch.bincount(y, minlength=100).tolist())
print("logit norm", lg.norm(dim=1).mean().item(), "max logit", lg.max(1).values.mean().item())
torch.save({"logits": lg, "y": y}, "orig_forget_logits.pt")
save_sub(m, "sub_orig.pt")
# augment behaviours
import torchvision.transforms.v2 as T
def rep(name, fn):
    l = predict(m, X, fn=fn); ce = F.cross_entropy(l, y, reduction='none')
    print(f"{name:28s} acc {(l.argmax(1)==y).float().mean():.4f} loss {ce.mean():.4f} std {ce.std():.4f} conf {l.softmax(1).max(1).values.mean():.4f}")
rep("identity", None)
rep("hflip", lambda x: x.flip(3))
rep("noise0.05", lambda x: x + 0.1*torch.randn_like(x))
rep("noise0.1", lambda x: x + 0.2*torch.randn_like(x))
rep("rrc(0.5,1)", T.RandomResizedCrop(224, scale=(0.5,1), antialias=True))
rep("rrc(0.25,1)", T.RandomResizedCrop(224, scale=(0.25,1), antialias=True))
rep("rot15", T.RandomRotation(15))
rep("blur", T.GaussianBlur(9, 2.0))
rep("bright", lambda x: x*0.7)
rep("gray", lambda x: x.mean(1, keepdim=True).expand_as(x))

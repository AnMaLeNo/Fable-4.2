"""Shared helpers for task 3 (unlearning)."""
import os
from pathlib import Path
import numpy as np, torch, timm

DATA = Path(os.environ.get("T3_DATA", "data/unlearning"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)

def load_model(path=None):
    m = build_model()
    m.load_state_dict(torch.load(path or DATA / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return m.to(DEVICE).eval()

def load_forget():
    with np.load(DATA / "forget_images.npz") as d:
        return torch.from_numpy(d["images"].copy())

def prep(u8):  # uint8 [B,H,W,3] -> float [-1,1] [B,3,H,W]
    x = u8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5

@torch.no_grad()
def predict(model, u8, bs=128, fn=None):
    model.eval(); out = []
    for i in range(0, len(u8), bs):
        x = prep(u8[i:i + bs])
        if fn is not None: x = fn(x)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out)

def save_sub(model, path):
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        assert torch.isfinite(v).all(), k
    build_model().load_state_dict(sd, strict=True)
    torch.save(sd, path)
    assert os.path.getsize(path) < 450e6
    return path

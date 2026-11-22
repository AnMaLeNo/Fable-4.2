"""Local reproductions of quantization variants (weights only), applied in place to a torch model."""
import torch
def fp8_e4m3(x):
    """Round to the nearest float8 E4M3 value (3 mantissa bits, exponents -6..8, subnormals, max 448)."""
    try:
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    except Exception:
        s = torch.sign(x); a = x.abs().double()
        e = torch.floor(torch.log2(a.clamp_min(1e-30))).clamp(-6, 8)
        step = torch.pow(2.0, e - 3)
        q = torch.round(a / step) * step
        return (s * q.clamp(max=448.0)).to(x.dtype)
def fp8_e5m2(x):
    s = torch.sign(x); a = x.abs().double()
    e = torch.floor(torch.log2(a.clamp_min(1e-30))).clamp(-14, 15); step = torch.pow(2.0, e - 2)
    return (s * (torch.round(a / step) * step).clamp(max=57344.0)).to(x.dtype)
def int_sym(x, bits=8, per_channel=True):
    """Symmetric linear quantization of a 2-D weight, per output channel."""
    qmax = 2 ** (bits - 1) - 1
    sc = (x.abs().amax(dim=1, keepdim=True) if (per_channel and x.dim() == 2) else x.abs().max()).clamp_min(1e-12) / qmax
    return torch.round(x / sc).clamp(-qmax, qmax) * sc
def scaled_fp8(x):
    """float8 with a per-tensor scale so that max|w| maps to 448 (quanto / vLLM style)."""
    sc = x.abs().max().clamp_min(1e-12) / 448.0
    return fp8_e4m3(x / sc) * sc
def apply(model, fn, which="linear"):
    n = 0
    for name, p in model.named_parameters():
        if which == "all" or (which == "linear" and p.dim() == 2 and "embed" not in name and "wte" not in name and "lm_head" not in name) or (which == "matrices" and p.dim() == 2):
            p.data = fn(p.data); n += 1
    return n
if __name__ == "__main__":
    x = torch.randn(5) * 0.05; print(x, fp8_e4m3(x), hasattr(torch, "float8_e4m3fn"))

def scaled_fp8_rows(x):
    sc = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    return fp8_e4m3(x / sc) * sc
def add_act_fp8(model, skip=("lm_head",)):
    """Dynamic per-tensor FP8 quantization of the inputs of every Linear layer (vLLM / llm-compressor W8A8 style)."""
    import torch.nn as nn
    def hook(mod, args):
        x = args[0]; sc = x.abs().max().clamp_min(1e-12) / 448.0
        return (fp8_e4m3(x / sc) * sc,) + tuple(args[1:])
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in name for s in skip):
            m.register_forward_pre_hook(hook); n += 1
    return n

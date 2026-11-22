"""Score endpoint responses under a local copy of a candidate base model (CPU).
For every response token: log-prob at a grid of temperatures, its expectation/variance under the model, rank, nucleus mass."""
import os, sys, json, math, collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8)
TAUS = np.round(np.exp(np.linspace(np.log(0.2), np.log(1.6), 22)), 3)
class Scorer:
    def __init__(self, name, dtype=torch.float32, system=None, transform=None):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype); self.model.eval()
        if transform: transform(self.model)
        self.system = system
    def ids(self, prompt, response):
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a = self.tok(p, add_special_tokens=False)["input_ids"]; b = self.tok(response, add_special_tokens=False)["input_ids"]
        return a, b
    @torch.no_grad()
    def score(self, prompt, response, max_resp=300):
        a, b = self.ids(prompt, response); b = b[:max_resp]
        if not b: return None
        x = torch.tensor([a + b]); lg = self.model(x).logits[0, len(a) - 1: len(a) + len(b) - 1].double()
        tgt = torch.tensor(b)
        out = {"lp": [], "e": [], "v": []}
        for tau in TAUS:
            l = torch.log_softmax(lg / tau, -1); p = l.exp()
            e = (p * l).sum(-1); v = (p * l * l).sum(-1) - e * e
            out["lp"].append(l[torch.arange(len(b)), tgt].numpy()); out["e"].append(e.numpy()); out["v"].append(v.numpy())
        rank = (lg > lg[torch.arange(len(b)), tgt][:, None]).sum(-1).numpy()
        return {k: np.array(v) for k, v in out.items()} | {"rank": rank, "n": len(b), "raw": lg[torch.arange(len(b)), tgt].numpy()}
def summarize(recs, skip_first=0):
    """recs: list of score dicts. Returns per-tau total LL, z, and the ML tau."""
    lp = np.concatenate([r["lp"][:, skip_first:] for r in recs], 1); e = np.concatenate([r["e"][:, skip_first:] for r in recs], 1); v = np.concatenate([r["v"][:, skip_first:] for r in recs], 1)
    LL = lp.sum(1); k = int(np.argmax(LL)); z = (lp - e).sum(1) / np.sqrt(v.sum(1))
    rank = np.concatenate([r["rank"][skip_first:] for r in recs])
    return dict(n=lp.shape[1], tau=float(TAUS[k]), ll=float(LL[k] / lp.shape[1]), z=float(z[k]), z_all=z, LL=LL,
                rank_gt20=float((rank >= 20).mean()), rank0=float((rank == 0).mean()))

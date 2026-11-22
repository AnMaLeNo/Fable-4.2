"""Low-dimensional generative model of the gold ensemble + exact re-implementation of the leaderboard metric.
Fitted to all scored submissions (fit_gold.py), then used to choose targets (design.py)."""
import glob, math, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(2)
y = torch.load("orig_forget_logits.pt")["y"]; N = len(y); K = 5
def wrong(P):
    W = P.clone().float(); W.scatter_(1, y[:, None], 0); return W / W.sum(1, keepdim=True).clamp_min(1e-12)
rkn = lambda v: (v.argsort().argsort().float() + 0.5) / len(v)
nscore = lambda v: torch.special.ndtri(rkn(v).double()).float()
def cls_rows(P):
    out = torch.zeros_like(P)
    for c in y.unique():
        m = y == c; out[m] = (P[m].sum(0, keepdim=True) - P[m]) / (m.sum() - 1)
    return out
# ---- signals
ens = torch.load("probe_ens.pt"); Lp = torch.stack([rkn(F.cross_entropy(d["Z"] / d["T"], y, reduction="none")) for d in ens["per_model"].values()]).mean(0)
Li_all = torch.cat([torch.load("i2i_all.pt")["L"], torch.load("i2i_all2.pt")["L"]], 1); Li = torch.cat([torch.load("i2i_all.pt")["ce"], torch.load("i2i_all2.pt")["ce"]], 1).mean(1)
st = {}; [st.update(torch.load(f)) for f in sorted(glob.glob("stud_ext*.pt"))]
Ls = torch.stack([rkn(F.cross_entropy(Z, y, reduction="none")) for Z in st.values()]).mean(0)
SIG = {"P": nscore(Lp), "I": nscore(Li), "S": nscore(Ls)}
lg0 = torch.load("orig_forget_logits.pt")["logits"]
WR = {"P": wrong(ens["P"]), "I": wrong((Li_all / 2).softmax(2).mean(1)), "S": wrong(torch.stack([(Z / 2).softmax(1) for Z in st.values()]).mean(0)), "R": wrong((lg0 / 2).softmax(1))}
WR["rows"] = (cls_rows(WR["I"]) + cls_rows(WR["S"]) + cls_rows(WR["R"])) / 3
# ---- common random numbers
g = torch.Generator().manual_seed(1234); D = 3
ETA = torch.randn(D, N, generator=g); XI = torch.randn(D, N, K, generator=g); ZW = torch.randn(D, N, K, generator=g); UR = torch.rand(D, N, K, generator=g); UC = torch.rand(D, N, generator=g)
PNAMES = ["uI", "uS", "lam", "c", "omega", "mu_w", "sig_w", "mu_r", "a_r", "s_top", "vP", "vI", "vS", "vR", "vrows", "kappa"]
P0 = dict(uI=1.0, uS=1.5, lam=0.6, c=0.5, omega=0.45, mu_w=9.0, sig_w=0.5, mu_r=0.05, a_r=1.0, s_top=0.6, vP=0.5, vI=1.0, vS=1.0, vR=1.0, vrows=1.0, kappa=0.25)
LO = dict(uI=0.0, uS=0.0, lam=0.05, c=0.0, omega=0.2, mu_w=3.0, sig_w=0.1, mu_r=0.0005, a_r=1.0, s_top=0.0, vP=0.0, vI=0.0, vS=0.0, vR=0.0, vrows=0.0, kappa=0.0)
HI = dict(uI=10.0, uS=10.0, lam=0.98, c=1.0, omega=0.75, mu_w=20.0, sig_w=2.0, mu_r=0.5, a_r=4.0, s_top=0.99, vP=3.0, vI=3.0, vS=3.0, vR=3.0, vrows=3.0, kappa=1.0)
def gold(p, d):
    """returns per-model losses [N,K] and the ensemble distribution at T=2 [N,100] for random draw d"""
    h = SIG["P"] + p["uI"] * SIG["I"] + p["uS"] * SIG["S"]; h = (h - h.mean()) / h.std()
    t = p["lam"] * h[:, None] + math.sqrt(1 - p["lam"] ** 2) * (math.sqrt(p["c"]) * ETA[d][:, None] + math.sqrt(1 - p["c"]) * XI[d])
    thr = float(torch.special.ndtri(torch.tensor(1 - p["omega"], dtype=torch.float64)))
    w = t > thr
    Lw = p["mu_w"] * torch.exp(p["sig_w"] * ZW[d] - p["sig_w"] ** 2 / 2); Lr = p["mu_r"] * (-torch.log(UR[d].clamp_min(1e-6))) ** p.get("a_r", 1.0)
    L = torch.where(w, Lw, Lr).clamp(1e-5, 30.0)
    G = (p["vP"] * WR["P"] + p["vI"] * WR["I"] + p["vS"] * WR["S"] + p["vR"] * WR["R"] + p["vrows"] * WR["rows"] + 1e-6); G = G / G.sum(1, keepdim=True)
    cs = torch.searchsorted(G.cumsum(1), UC[d][:, None].clamp(max=0.999999)).clamp(max=99).squeeze(1)       # confused class of the image (shared by the gold models)
    top = F.one_hot(cs, 100).float(); Wd = p["s_top"] * top + (1 - p["s_top"]) * G                           # [N,100], zero at y
    py = torch.exp(-L)                                                                                           # [N,K]
    pg = torch.zeros(N, 100)
    for k in range(K):
        p1 = Wd * (1 - py[:, k:k + 1]); p1.scatter_(1, y[:, None], py[:, k:k + 1]); p2 = p1.clamp_min(1e-12).sqrt(); pg += p2 / p2.sum(1, keepdim=True) / K
    return L, pg
def kl_to(pg, logits):
    return (pg * (pg.clamp_min(1e-12).log() - F.log_softmax(logits.float() / 2, 1))).sum(1)
def score_sub(p, sub, orig, golds):
    """sub/orig: dicts with clean [N,100], mild [M,100] (+ mild_idx in orig), ext; golds: list of (L, pg)"""
    idx = orig["mild_idx"]; tot = 0.0
    for L, pg in golds:
        gm, gs = L.mean(0).mean().item(), L.std(0).mean().item(); ms, pr = [], []
        for key, sel in (("clean", None), ("mild", idx)):
            yy = y if sel is None else y[sel]; pgs = pg if sel is None else pg[sel]
            ls = F.cross_entropy(sub[key].float(), yy, reduction="none"); lo = F.cross_entropy(orig[key].float(), yy, reduction="none")
            if sel is None: gmm, gss = gm, gs
            else: gmm, gss = L[sel].mean(0).mean().item(), L[sel].std(0).mean().item()
            d_s = abs(ls.mean().item() - gmm) + abs(ls.std().item() - gss); d_o = abs(lo.mean().item() - gmm) + abs(lo.std().item() - gss)
            ms.append(1 - min(max(d_s / d_o, 0), 1)); pr.append(1 - min(max(kl_to(pgs, sub[key]).mean().item() / kl_to(pgs, orig[key]).mean().item(), 0), 1))
        tot += (sum(ms) / 2) * (sum(pr) / 2)
    po = (orig["ext"].float() / 2).softmax(1); kle = (po * (F.log_softmax(orig["ext"].float() / 2, 1) - F.log_softmax(sub["ext"].float() / 2, 1))).sum(1).mean().item()
    agree = (orig["ext"].argmax(1) == sub["ext"].argmax(1)).float().mean().item(); proxy = agree / (1 + kle)
    return tot / len(golds) * (1 - p["kappa"] * (1 - proxy))

"""Simulateur inter-familles : une famille de pseudo-gold joue le role de gold, on evalue la metrique exacte."""
import torch, glob, json, numpy as np, torch.nn.functional as F, os, sys
T3=os.environ.get('T3_DIR', '../')   # folder holding the task-3 artefacts (orig_forget_logits.pt, probe_ens.pt, ...)
y = torch.load(T3+'orig_forget_logits.pt')['y']; N=len(y)
orig = torch.load(T3+'outputs/orig.pt')
ZO = orig['clean'].float()
def members():
    fam={}
    ens=torch.load(T3+'probe_ens.pt')
    fam['P']=[d['Z']/d['T'] for d in ens['per_model'].values()]
    fam['S']=[]
    for f in ['stud_ext1.pt','stud_ext2.pt','stud_ext3.pt','studbig_1.pt','studbig_2.pt']:
        for n,Z in torch.load(T3+f).items(): fam['S'].append(Z.float())
    fam['R']=[Z.float() for n,Z in torch.load(T3+'reset_1.pt').items() if n.startswith('reset')]
    fam['B']=[Z.float() for n,Z in torch.load(T3+'reset_1.pt').items() if n.startswith('b9')]
    fam['I']=[]
    for f in ['i2i_all.pt','i2i_all2.pt','i2i_s65.pt']:
        L=torch.load(T3+f)['L'].float()
        for v in range(L.shape[1]): fam['I'].append(L[:,v])
    return fam
def ce(Z): return F.cross_entropy(Z,y,reduction='none')
def p2(Z): return (Z/2).softmax(1)
def kl(p,logq): return (p*(p.clamp_min(1e-12).log()-logq)).sum(1)
def forget_score(Zs, gold, Zo=ZO, idx=None, detail=False):
    """gold: list of logits [N,100]"""
    if idx is None: idx=torch.arange(N)
    yy=y[idx]
    Lg=[F.cross_entropy(g[idx],yy,reduction='none') for g in gold]
    mg=np.mean([l.mean().item() for l in Lg]); sg=np.mean([l.std().item() for l in Lg])
    pg=torch.stack([p2(g[idx]) for g in gold]).mean(0)
    Ls=F.cross_entropy(Zs[idx],yy,reduction='none'); Lo=F.cross_entropy(Zo[idx],yy,reduction='none')
    ds=abs(Ls.mean().item()-mg)+abs(Ls.std().item()-sg); do=abs(Lo.mean().item()-mg)+abs(Lo.std().item()-sg)
    ms=1-min(max(ds/do,0),1)
    ks=kl(pg,F.log_softmax(Zs[idx]/2,1)).mean().item(); ko=kl(pg,F.log_softmax(Zo[idx]/2,1)).mean().item()
    pr=1-min(max(ks/ko,0),1)
    if detail: return dict(ms=ms,pred=pr,f=ms*pr,mg=mg,sg=sg,ms_sub=(Ls.mean().item(),Ls.std().item()),ks=ks,ko=ko)
    return ms*pr
def sharpen(Z, target_mean):
    """echelle a telle que la loss moyenne = target_mean (bissection)"""
    lo,hi=0.2,30.
    for _ in range(40):
        a=(lo+hi)/2
        if ce(Z*a).mean()<target_mean: lo=a
        else: hi=a
    return Z*a
def logits_from_P1(P1): return P1.clamp_min(1e-12).log()

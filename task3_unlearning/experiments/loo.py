import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import *
fam=members(); fam['I']=fam['I'][:8]
rk=lambda v: v.argsort().argsort().float()/len(v)
def wrong(P):
    W=P.clone().float(); W.scatter_(1,y[:,None],0); return W/W.sum(1,keepdim=True).clamp_min(1e-12)
def tcal(Z):
    T=min((F.cross_entropy(Z/t,y).item(),t) for t in [0.5,0.75,1,1.5,2,3])[1]; return (Z/T).softmax(1)
SIG={k:torch.stack([rk(ce(Z)) for Z in m]).mean(0) for k,m in fam.items()}
WR={k:torch.stack([wrong(tcal(Z) if k in 'SRB' else (Z/ (2 if k=='I' else 1)).softmax(1)) for Z in m]).mean(0) for k,m in fam.items()}
Lshape=torch.load(T3+'teacher_xheavy.pt')['L'].flatten().sort().values
def build(w, omega=0.45, scale=2.0, maxloss=20.0, beta=2.0, h=None, R=None, quota=None, lossmap=None):
    """w: dict famille->poids ; quota: dict classe->fraction d'images fausses (cible par classe)"""
    tot=sum(w.values())
    if h is None: h=sum(w[k]*SIG[k] for k in w)/tot
    if R is None: R=sum(w[k]*WR[k] for k in w)/tot
    if quota is None:
        qs=(h.argsort().argsort().float()+0.5)/N
    else:
        # rang intra-classe, place dans la distribution globale selon le quota de la classe
        qs=torch.zeros(N)
        for c in y.unique().tolist():
            m=(y==c).nonzero().squeeze(1); r=(h[m].argsort().argsort().float()+0.5)/len(m)   # 0 facile..1 difficile
            f=quota[c]; cut=1-f
            # r<cut -> zone juste [0,0.55) ; r>=cut -> zone fausse [0.55,1]
            qs[m]=torch.where(r<cut, r*(0.55)/max(cut,1e-6), 0.55+(r-cut)*0.45/max(f,1e-6)).clamp(0,1)
    w0=0.45; cut=1-omega
    qs=torch.where(qs<cut, qs*(1-w0)/cut, (1-w0)+(qs-cut)*w0/omega).clamp(0,1)
    loss=(Lshape[(qs*(len(Lshape)-1)).long()]*scale)
    if lossmap is not None: loss=lossmap(loss,qs)
    loss=loss.clamp(1e-4,maxloss)
    R=(R+1e-7)**beta; R=R.clone(); R.scatter_(1,y[:,None],0.0); R=R/R.sum(1,keepdim=True)
    py=torch.exp(-loss); P1=R*(1-py)[:,None]; P1.scatter_(1,y[:,None],py[:,None]); return P1,loss
def gold_of(k, mean=4.4, n=None, seed=0):
    m=fam[k]
    if n: 
        g=torch.Generator().manual_seed(seed); m=[m[i] for i in torch.randperm(len(m),generator=g)[:n].tolist()]
    return [sharpen(Z,mean) if mean else Z for Z in m]
def ev(P1,gold): 
    r=forget_score(P1.clamp_min(1e-12).log(),gold,detail=True); return r

def soften(P1, cap=0.85, eps=0.0, R_broad=None, gamma=1.0):
    """Plancher de masse 'mauvaises classes' a T=2 : p2(y) <= cap ; melange eps d'une distribution large R_broad dans la forme des mauvaises classes.
    gamma<1 : aplatit la forme des mauvaises classes (a T=2) uniquement pour les images dont p2(y) a ete plafonne."""
    P2=P1.sqrt(); P2=P2/P2.sum(1,keepdim=True)
    a=P2.gather(1,y[:,None]).squeeze(1)
    W=P2.clone(); W.scatter_(1,y[:,None],0); W=W/W.sum(1,keepdim=True).clamp_min(1e-12)
    capped=a>cap
    if gamma!=1.0:
        Wf=(W+1e-9)**gamma; Wf.scatter_(1,y[:,None],0); Wf=Wf/Wf.sum(1,keepdim=True); W=torch.where(capped[:,None],Wf,W)
    if eps>0:
        W=(1-eps)*W+eps*R_broad
    a2=torch.minimum(a,torch.full_like(a,cap))
    Q=W*(1-a2)[:,None]; Q.scatter_(1,y[:,None],a2[:,None])
    P1n=Q**2; P1n=P1n/P1n.sum(1,keepdim=True); return P1n
def class_rows(R):
    """ligne de confusion par classe = moyenne de R sur les images de la classe, melangee a 20 % d'uniforme hors vraie classe"""
    out=torch.zeros_like(R)
    for c in y.unique().tolist():
        m=y==c; row=R[m].mean(0); out[m]=row
    U=torch.ones_like(R); U.scatter_(1,y[:,None],0); U=U/U.sum(1,keepdim=True)
    out=0.8*out+0.2*U; out.scatter_(1,y[:,None],0); return out/out.sum(1,keepdim=True)

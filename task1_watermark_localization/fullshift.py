"""Type- and method-dependent threshold, applied as a logit shift before rectangles / SAM snapping (in place).
Full-type images (afull >= 0.7, not in the suspicious list): logits -= shift (the restored holes are under-predicted).
python fullshift.py <dir> <shift | ps,tm,ss>      env PARTSHIFT=ps,tm,ss : shift for the other (partial) images; env SUSP/SUSPFILE: see samsnap.py"""
import sys, os, json, numpy as np
d = sys.argv[1]; s = [float(x) for x in sys.argv[2].split(',')]; s = s * 3 if len(s) == 1 else s
ps = [float(x) for x in os.environ.get('PARTSHIFT', '0,0,0').split(',')]
AF = json.load(open('afull.json')); SUSP = json.load(open(os.environ.get('SUSPFILE', 'suspicious.json'))) if os.environ.get('SUSP') else {}
for split, tag in (('val', 'train'), ('test', 'test')):
    b = dict(np.load(f'{d}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); n = 0
    for k, name in enumerate(b['names']):
        if np.abs(L[k]).sum() == 0: continue
        mi = int(b['method'][k])
        if AF[f'{tag}_{name}'] >= 0.7 and f'{tag}_{name}' not in SUSP: L[k] = L[k] - s[mi]; n += 1
        elif AF[f'{tag}_{name}'] < 0.7 and ps[mi] != 0: L[k] = L[k] - ps[mi]
    b['logits'] = L; np.savez(f'{d}/{split}.npz', **b); print(split, 'shifted', n)

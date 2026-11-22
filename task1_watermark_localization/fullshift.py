"""Full-type images (afull >= 0.7): the ensemble under-predicts the restored holes (control: predicted hole fraction 0.08-0.10 vs 0.115);
subtract a constant from the logits of these images (= higher threshold).  python fullshift.py <dir> <shift>   (in place)"""
import sys, json, numpy as np
d, s = sys.argv[1], float(sys.argv[2]); AF = json.load(open('afull.json'))
for split, tag in (('val', 'train'), ('test', 'test')):
    b = dict(np.load(f'{d}/{split}.npz', allow_pickle=True)); L = b['logits'].copy(); n = 0
    for k, name in enumerate(b['names']):
        if AF[f'{tag}_{name}'] >= 0.7 and np.abs(L[k]).sum() > 0: L[k] = L[k] - s; n += 1
    b['logits'] = L; np.savez(f'{d}/{split}.npz', **b); print(split, 'shifted', n)

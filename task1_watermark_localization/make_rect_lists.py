"""Builds the three rectangle lists used by assemble.sh from the raw decoder outputs (these conversions were first done inline).
  TrustMark : preds/tscan_trustmark_test_{0,1}.npy (tscan.py), refined bit accuracy >= 0.85, box corrected for the decoder bias
              (decoder resolution 245 vs 256: the best decoding box is anchored top-left and 4.5 % smaller) and for the 2 px label inset:
              (y0 + 2, x0 + 2, y0 + 1.045 h - 2, x0 + 1.045 w - 2)                         -> preds/tscanfix_trustmark_test.npy
  StegaStamp: rect_pipe.py on the ensemble (preds/C3/rect_stegastamp_test_{0,1}.npz) + rect_more.py runs, accuracy >= 0.85;
              the 2 px inset is applied by apply_rects.py --shrink 2                         -> preds/rectss_all_test.npy
  PixelSeal : rect_more.py (METHOD=pixelseal), accuracy >= 0.68 (256 bits: far above the chance maximum of the search),
              inset by apply_rects.py --shrink 2                                             -> preds/rectps_ok_test.npy"""
import glob, numpy as np
tm = [d for f in sorted(glob.glob('preds/tscan_trustmark_test_*.npy')) for d in np.load(f, allow_pickle=True) if d['rect'] is not None and d['acc'] is not None and d['acc'] >= 0.85]
fix = lambda r: (int(round(r[0] + 2)), int(round(r[1] + 2)), int(round(r[0] + (r[2] - r[0]) * 1.045 - 2)), int(round(r[1] + (r[3] - r[1]) * 1.045 - 2)))
np.save('preds/tscanfix_trustmark_test.npy', np.array([dict(d, rect=fix(d['rect'])) for d in tm], dtype=object), allow_pickle=True)
ss = []
for f in sorted(glob.glob('preds/C3/rect_stegastamp_test_*.npz')):
    z = np.load(f, allow_pickle=True); ss += [dict(name=str(n), rect=tuple(int(v) for v in r), scan_acc=float(a), acc=float(a)) for n, r, a in zip(z['names'], z['rects'], z['accs']) if a >= 0.85]
for f in sorted(glob.glob('preds/rectss_more*_test.npy')): ss += [d for d in np.load(f, allow_pickle=True) if d['acc'] >= 0.85 and d['name'] not in {s['name'] for s in ss}]
np.save('preds/rectss_all_test.npy', np.array(ss, dtype=object), allow_pickle=True)
ps = {}
for f in sorted(glob.glob('preds/rectps_more*_test.npy')):
    for d in np.load(f, allow_pickle=True):
        if d['rect'] is not None and d['acc'] >= 0.68 and d['acc'] > ps.get(d['name'], dict(acc=0))['acc']: ps[d['name']] = d
np.save('preds/rectps_ok_test.npy', np.array([dict(d, acc=max(d['acc'], 0.85)) for d in ps.values()], dtype=object), allow_pickle=True)
print('TrustMark', len(tm), 'StegaStamp', len(ss), 'PixelSeal', len(ps))

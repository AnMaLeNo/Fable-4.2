#!/bin/bash
# Final assembly in one command:  bash assemble.sh <tag> [OM=objmasks_all.npz] -- <ens.py members...>
#   ensemble (ens.py) -> decoded rectangles (TrustMark corrected boxes, StegaStamp and PixelSeal shrunk by 2 px) -> SAM snapping of
#   object-type predictions -> threshold -> subq/pending/00_<tag>.pt (picked up by submitter.sh).  Prints the control-split comparison to C7.
cd "$(dirname "$0")"; P=${PYTHON:-python}; tag=$1; shift; OM=objmasks_all.npz
if [ "$1" != "--" ]; then OM=$1; shift; fi; shift
$P ens.py --out preds/${tag}_e "$@" > /dev/null 2>&1 || { echo "ens failed"; exit 1; }
cur=preds/${tag}_e; [ -n "$FULLSHIFT" ] && $P fullshift.py $cur $FULLSHIFT | tail -n 1
for spec in "preds/tscanfix_trustmark_{split}.npy:0" "preds/recttmfix_more_{split}.npy:0" "preds/rectss_all_{split}.npy:2" "preds/rectps_ok_{split}.npy:2"; do
  f=${spec%%:*}; s=${spec##*:}
  if ls ${f/\{split\}/test} > /dev/null 2>&1; then $P apply_rects.py --base $cur --out preds/${tag}_r --rects "$f" --thr 0.85 --shrink $s | tail -n 1; cur=preds/${tag}_r; fi
done
$P samsnap.py --base $cur --out preds/$tag --thr ${THR:-0.5} --kmax ${KMAX:-1} --delta=${DELTA:-9,9,9} --om $OM | grep -v arn | cut -c1-220
$P compare.py preds/C7 preds/$tag | head -n 4 | cut -c1-260
$P postproc.py preds/$tag --write subq/pending/00_$tag.pt --variant thr0 | tail -n 1 | cut -c1-90

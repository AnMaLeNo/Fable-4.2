# Task 1 — Watermark localisation (images)

For each 512×512 test image (one known watermark method among PixelSeal, TrustMark, StegaStamp, and the known embedded
message), output a 128×128 binary mask of the watermarked area. Metric: per-image mIoU over the two classes, averaged.

**Public score: 0.7705** (22 Nov, 07:11; all-zeros baseline: 0.2553). <!-- TODO-FINAL: final public score + time -->

## Idea

1. **Learn a segmenter per method on synthetic composites.** We embed watermarks ourselves with the starter-kit embedders,
   composite them with clean content using the three edit types we identified in the training set, and train a 512→128
   segmenter. For PixelSeal and TrustMark the encoder is the *pretrained watermark extractor itself*, run fully
   convolutionally; for StegaStamp a U-Net with high-pass input channels is enough.
2. **Make the synthetic data look like the real data.** The decisive finding: in the real images the watermarked frame was
   *enlarged by an isotropic factor of about 1.00–1.07 and cropped back to 512 after embedding*. A model trained on pristine
   kit embeddings sees almost nothing in real images; adding the same zoom to the synthetic composites closes the gap.
3. **Use the decoder where it is exact.** When a crop coincides with the rectangle the watermark was embedded in, decoding
   it against the known message gives a sharp bit-accuracy peak. Rectangles found this way and verified (bit accuracy
   ≥ 0.85) replace the segmenter output.
4. **Snap object-shaped predictions onto Segment Anything masks**, because the real "object" labels follow object contours.

The three edit types found in the 510 training images (they drive everything below):

| Type | Share | What it is |
|---|---|---|
| A "full" | ~50 % | full-frame embedding, then a hole (object removed / added / modified) |
| B "rectangle" | ~25 % | embedding on a rectangle only (outpainting) |
| C "object" | ~25 % | full-frame embedding kept only on an object (transplant) |

## Pipeline, step by step

All scripts expect the organisers' starter kit next to them (`wmloc/` package and `download_models.py`, not redistributed
here), the data unzipped in `release/`, and write predictions as logit maps in `preds/<name>/{val,test}.npz`.

| # | Step | File → function / entry point |
|---|---|---|
| 1 | Type prior: full-frame decode bit accuracy against the known message; ≥ 0.7 ⇒ type "full" (recall 100 % on train, 2–9 % false positives among partials) | `afull_all.py` → `afull.json` |
| 2 | Synthetic composites, types A and B (kit embedders on images of the *other* methods as clean sources; real masks + procedural shapes) | `gen_synth.py` (`blob`, `rand_rect`, `is_val` = local 20 % control split) |
| 3 | Type C composites (full-frame embedding kept on an object) | `gen_transplant.py` |
| 4 | Generator with the three types in the real proportions, hole sizes drawn like the real ones | `gen3.py` |
| 5 | Content-aligned shapes: Segment Anything masks of every real image, used as holes / transplanted objects | `sam_masks.py`, `gen4.py` |
| 6 | Segmenters: U-Net with high-pass channels (StegaStamp); pretrained extractor + FPN (`PreSeg`), image-pyramid version (`PreSegMS`) for PixelSeal / TrustMark; float32 GRN fix for ConvNeXtV2 under AMP | `seg.py` → `UNet`; `seg2.py` → `PreSeg`, `PreSegMS`, `_grn32`, `build` |
| 7 | Training: BCE + two-sided Dice, 20 % real / 80 % synthetic samples, **zoom augmentation of image and label together** | `train_seg.py` → `main` (`--degrade zoom`, `--real_frac 0.2`); `degrade.py` → `zoom_pair` |
| 8 | Inference on the control split and the test set | `predict.py` → `run`, `main` |
| 9 | "Zoom rescue": a model trained on 2× crops, applied on overlapping 2× tiles, only where the base model finds almost nothing | `train_seg.py --zoom 2`, `zoom_pred.py`, `zoom_eval.py` → `zoom_logits`, `merge_zoom.py --mode miss` |
| 10 | Ensembling: weighted mean of logits; type-routed weights (full vs partial) | `ens.py`, `route.py` |
| 11 | Decoder-verified rectangles. TrustMark (tolerant decoder): scan a grid of windows, then hill-climb the decode margin. StegaStamp (intolerant decoder): dense refinement around segmenter proposals. Accept only if bit accuracy ≥ 0.85 | `rect_search.py` → `Scorer`, `refine`; `tscan.py` → `windows`, `scan`, `detect`; `rect_pipe.py` → `proposals`, `dense_refine`; `rect_more.py` |
| 12 | Calibration of decoded box vs label box (the label is 2 px inside the decoded box; TrustMark needs an anchored 4.5 % correction) and application | `bias.py`; `apply_rects.py` (`--thr 0.85 --shrink 2`) |
| 13 | SAM snapping of object-type predictions: replace a non-full, non-rectangular prediction by the SAM mask(s) with the best soft IoU against sigmoid(logits) | `samsnap.py` → `greedy`, `rectlike`, the `--delta` rule |
| 14 | Full-type images: constant logit shift (the ensemble under-predicts the restored holes) | `fullshift.py` |
| 15 | Threshold, write the submission, submit | `postproc.py --variant thr0`; `common.py` → `submit` |
| — | Steps 10–15 in one command | `assemble.sh` |
| — | Local evaluation: per-cell scores (method × type) with standard errors, paired comparisons | `cvlib.py`, `compare.py`, `eval_break.py` |

## Reproduce

```bash
pip install -r requirements.txt segment_anything   # requirements.txt = the starter kit's file (not copied here)
python download_models.py                  # starter kit
export SERVER_URL=<hackathon server> API_KEY=<team token>     # read from the environment, never stored

python afull_all.py                                             # step 1
python gen3.py --help ; python gen4.py --help                   # steps 2-5: synthetic packs under synth*/
python sam_masks.py --out objmasks2 --pps 32 --iou 0.80 --stab 0.85 --kmax 64 --amax 0.9 --amin 0.001

# long models of the best ensemble (one 4x A100 node)
python train_seg.py --out runs/L_tm --arch ms:trustmark --methods 1 --bs 8 --lr 5e-4 --enc_lr 5e-5 --steps 20000 \
       --eval_every 2000 --ram --synth synthAll --real_frac 0.2 --degrade zoom --seed 81
python predict.py --m1 runs/L_tm/last.pt --out preds/L_tm --bs 2
python train_seg.py --out runs/L_ps --arch ms:pixelseal:0.5,1 --methods 0 --bs 8 --lr 4e-4 --enc_lr 4e-5 --steps 16000 \
       --amp bf16 --no_bench --eval_bs 2 --eval_every 2000 --ram --synth synthAll --real_frac 0.2 --degrade zoom --seed 81
python predict.py --m0 runs/L_ps/last.pt --out preds/L_ps --bs 2 --amp bf16

# decoder-verified rectangles
python tscan.py trustmark test                                   # -> preds/tscan_trustmark_test.npy
python rect_pipe.py --pred preds/<ensemble> --method stegastamp --split test

# final assembly (ensemble -> rectangles -> SAM snapping -> threshold -> submission file)
FULLSHIFT=1.0 KMAX=1 DELTA=-0.1,-0.1,-0.1 bash assemble.sh C11_fs1 objmasks2_all.npz -- preds/C2s preds/L_tm:4 preds/L_ps:4
python common.py subq/pending/00_C11_fs1.pt
```

<!-- TODO-FINAL: if a later assembly (e.g. with the long StegaStamp models Q_ss_101 / Q_ss_102, or Q_tm_* / Q_ps_*) becomes the best
submission, replace the last two commands and the "best submission" lines below with its exact recipe. -->

Best submission at 07:11: **`00_C0_C11_fs1.pt`** = the assembly `C10` (ensemble members `C2s`, `L_tm` ×4, `L_ps` ×4;
decoded rectangles on the test set: 71 TrustMark with the box correction, 66 StegaStamp and 36 PixelSeal shrunk by 2 px;
SAM snapping with dense proposals, 64 masks per image) plus a logit shift of −1.0 on the 545 test images of type "full".
`C2s` is the earlier routed ensemble of shorter models with zoom rescue (public 0.7292 on its own with a plain threshold).

Known gaps in this copy (hackathon glue that was not kept as scripts): the TrustMark box correction
`(y0+2, x0+2, y0+1.045·h−2, x0+1.045·w−2)` and the merging of rectangle lists into the files read by `assemble.sh` were
done with inline snippets; the exact composition of the early ensembles inside `C2s` is described in the table of
results but its intermediate prediction folders are not rebuilt by a single command; the snapping thresholds of the 07:11
run were not written down (the command above uses the settings of `C10`).
<!-- TODO-FINAL: confirm the exact KMAX / DELTA / THR of the best submission, and which SAM backbone produced objmasks2_all.npz
(our notes say ViT-H; the saved job line uses the script default, ViT-B). -->

## Results (public split, 306 of the 1 020 test images)

| Time | Submission | Public mIoU | What changed |
|---|---|---|---|
| 17:35 | all zeros | 0.2553 | baseline |
| 18:12 | first segmenters | 0.6608 | pretrained-extractor segmenters (PixelSeal, TrustMark) + U-Net (StegaStamp) |
| 19:43 | E1 | 0.7033 | per-method ensembles, type C ("transplant") added to the synthetic data |
| 20:03 | E3 | 0.7171 | + zoom rescue for small rectangles |
| 00:41 | C2 | 0.7230 | type-routed ensemble with models trained with the zoom augmentation |
| 02:09 | C2s, plain threshold | 0.7292 | plain threshold instead of the max-sum-rectangle post-processing |
| 02:40 | C4 | 0.7339 | + TrustMark rectangles found by decoding (71 verified) |
| 02:50 | C5 | 0.7418 | + TrustMark box correction + StegaStamp decoded rectangles |
| 03:11 | C6 | 0.7474 | + SAM snapping of object-type predictions |
| 04:02 | C7 | 0.7614 | + long TrustMark and PixelSeal models (same recipe, 20 000 / 16 000 steps) |
| 06:51 | C10 | 0.7684 | + denser SAM proposals and the "object beats rectangle" snapping rule |
| 07:11 | C11_fs1 | 0.7705 | + logit shift on full-type images |
| TODO-FINAL | TODO-FINAL | TODO-FINAL | final best submission (file, score, what changed) |

Measurements behind the main choices (local, on real training images the model never saw):

- Decoding the exact embedding rectangle against the known message: bit accuracy 0.94 (TrustMark), 0.81 (StegaStamp),
  0.69 (PixelSeal); a 4 px shift drops StegaStamp to 0.56.
- A StegaStamp model trained with **no real image**: mIoU 0.07 on real full-type images without the zoom augmentation,
  0.35 with generic degradations (resize, JPEG, WebP, blur), **0.768 ± 0.013 on the 170 real StegaStamp images with the
  zoom** (full 0.830, rectangle 0.638, object 0.779). TrustMark, same test: 0.543 → 0.670.
- The implicit zoom was read off the interpolation comb of the kit in real StegaStamp images: 5th / 50th / 95th
  percentiles 0.986 / 1.034 / 1.065, identical along rows and columns.
- SAM snapping on models trained without real images: object cell 0.675 → 0.694 (TrustMark), 0.779 → 0.819 (StegaStamp);
  with the final rule on the control split, TrustMark object 0.60 → 0.74.
- Decoded rectangles: mIoU 0.957 on accepted StegaStamp rectangles; TrustMark 0.9315 → 0.950 with the box correction.

## What did not work

Scripts kept in `experiments/` document the negative results we cite.

- **Matched filtering with the expected residual** r̂ = embed(image, message) − image (`experiments/seg3.py`,
  `train_match.py`, `gen_full.py`): TrustMark full 0.47 against 0.65 for the plain segmenter; a classical matched filter
  has AUC 0.50 on real images. The image-dependent part of the residual dominates.
- **Locating by local decoding** (`experiments/locdec2.py`, `tmatch.py`): per-position extractor logits and sliding
  windows give pixel AUC 0.60–0.69; only whole rectangles decode.
- **Composites made only of real labelled images** (`experiments/remix.py`), StegaStamp, same number of steps,
  partial / full images: 0.635 / 0.765 against 0.657 / 0.82 for the base recipe; real images alone: 0.473 / 0.611
  (over-fitting).
- **Making synthetic data realistic by degradation** once real images are in training (`experiments/vae_rt.py` for the VAE
  round trip): 0.726 (VAE) and 0.723 (generic mix) against 0.739 for the base recipe.
- **Zooming the residual only** instead of the whole composite (`experiments/gen5.py`): −0.055 ± 0.010.
- **Long training with pseudo-labels of the test set**: better on the control split (StegaStamp 0.85 → 0.93) but the
  predictions on the test set stay those of the teacher (agreement 0.96); public 0.7588 against 0.7614. Removed from
  the final ensemble.
- **Eroding the predictions** (`experiments/erode_pp.py`): −0.002 to −0.007. **Max-sum-rectangle post-processing**
  (`postproc.py`, variant `rect.8`): +0.001 to +0.002 locally, −0.006 on the public split.

Remaining weak spots: holes of full-type PixelSeal / TrustMark images (hole IoU 0.37 / 0.41, against 0.76 for
StegaStamp; holes under 1 % of the image are all missed) and StegaStamp rectangles the segmenter never proposes.
The local control split is small (94 images, standard error about 0.025) and not representative per cell.

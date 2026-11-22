# Task 1 — Watermark localisation (images)

For each 512×512 test image (one known watermark method among PixelSeal, TrustMark, StegaStamp, and the known embedded
message), output a 128×128 binary mask of the watermarked area. Metric: per-image mIoU over the two classes, averaged.

**Public score: 0.7965** (22 Nov, 09:37, submission `00_G_C16_pure.pt`; all-zeros baseline: 0.2553). <!-- TODO-FINAL: re-read the score at 11:30 -->

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
5. **Ensemble long, clean models.** The final ensemble is three long trainings per method, averaged, with test-time
   translation averaging. They are trained **without any pseudo-label of the test set**: the variant trained with
   pseudo-labels looked better locally but had memorised its teacher on the test images, and was discarded as contaminated.

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
| 8 | Inference on the control split and the test set; test-time augmentation = mean over 9 views translated by ±4 / 8 px, logits moved back before averaging | `predict.py` → `run`, `_shifted`, `_unshift` (`--tta`) |
| 9 | "Zoom rescue": a model trained on 2× crops, applied on overlapping 2× tiles, only where the base model finds almost nothing | `train_seg.py --zoom 2`, `zoom_pred.py`, `zoom_eval.py` → `zoom_logits`, `merge_zoom.py --mode miss` |
| 10 | Ensembling: mean of logits (final ensemble: 3 long models per method, equal weights); type-routed weights (full vs partial) were used in the earlier ensembles | `ens.py`; `route.py` |
| 11 | Decoder-verified rectangles. TrustMark (tolerant decoder): scan a grid of windows, then hill-climb the decode margin. StegaStamp (intolerant decoder): dense refinement around segmenter proposals. Accept only if bit accuracy ≥ 0.85 | `rect_search.py` → `Scorer`, `refine`; `tscan.py` → `windows`, `scan`, `detect`; `rect_pipe.py` → `proposals`, `dense_refine`; `rect_more.py` |
| 12 | Calibration of decoded box vs label box (the label is 2 px inside the decoded box; TrustMark needs an anchored 4.5 % correction), construction of the three rectangle lists, application | `bias.py`; `make_rect_lists.py`; `apply_rects.py` (`--thr 0.85 --shrink 2`) |
| 13 | SAM snapping of object-type predictions: an image that is not full-type and has no decoded rectangle takes the single SAM mask with the best soft IoU against sigmoid(logits), if that IoU is ≥ 0.5 (non-rectangular prediction) or exceeds the soft IoU of the best rectangle − 0.1 | `samsnap.py` → `greedy`, `rectlike`, the `--delta` rule |
| 14 | Full-type images: threshold logit > 1 instead of > 0 (the ensemble under-predicts the restored holes) | `fullshift.py` |
| 15 | Threshold, write the submission, submit | `postproc.py --variant thr0`; `common.py` → `submit` |
| — | Steps 10–15 in one command | `assemble.sh` |
| — | Local evaluation: per-cell scores (method × type) with standard errors, paired comparisons | `cvlib.py`, `compare.py`, `eval_break.py` |

## Reproduce the best submission

```bash
pip install -r requirements.txt segment_anything   # requirements.txt = the starter kit's file (not copied here)
python download_models.py                          # starter kit
export SERVER_URL=<hackathon server> API_KEY=<team token>     # read from the environment, never stored

python afull_all.py                                              # image types -> afull.json
python sam_masks.py --model vit_h --ckpt models/sam/sam_vit_h_4b8939.pth --out objmasks2 \
       --pps 32 --iou 0.80 --stab 0.85 --kmax 64 --amax 0.9 --amin 0.001          # SAM ViT-H proposals
python gen3.py --help ; python gen4.py --help                    # synthetic packs, gathered under synthAll/

# nine long models, all WITHOUT pseudo-labels of the test set (one 4x A100 node)
C="--ram --synth synthAll --real_frac 0.2 --degrade zoom"
python train_seg.py --out runs/L_tm     $C --arch ms:trustmark --methods 1 --bs 8 --lr 5e-4 --enc_lr 5e-5 --steps 20000 --seed 81
python train_seg.py --out runs/Q_tm_102 $C --arch ms:trustmark --methods 1 --bs 8 --lr 5e-4 --enc_lr 5e-5 --steps 22000 --seed 102
python train_seg.py --out runs/Q_tm_103 $C --arch ms:trustmark --methods 1 --bs 8 --lr 5e-4 --enc_lr 5e-5 --steps 22000 --seed 103
python train_seg.py --out runs/L_ps     $C --arch ms:pixelseal:0.5,1 --methods 0 --bs 8 --lr 4e-4 --enc_lr 4e-5 --steps 16000 --amp bf16 --no_bench --eval_bs 2 --seed 81
python train_seg.py --out runs/Q_ps_102 $C --arch ms:pixelseal:0.5,1 --methods 0 --bs 8 --lr 4e-4 --enc_lr 4e-5 --steps 18000 --amp bf16 --no_bench --eval_bs 2 --seed 102
python train_seg.py --out runs/Q_ps_103 $C --arch ms:pixelseal:0.5,1 --methods 0 --bs 8 --lr 4e-4 --enc_lr 4e-5 --steps 18000 --amp bf16 --no_bench --eval_bs 2 --seed 103
python train_seg.py --out runs/Q_ss_101 $C --arch unet --methods 2 --bs 12 --lr 6e-4 --steps 30000 --seed 101
python train_seg.py --out runs/Q_ss_102 $C --arch unet --methods 2 --bs 12 --lr 6e-4 --steps 30000 --seed 102
# SS_clean_loc: a third StegaStamp U-Net, same recipe, 26 000 steps (its exact command line was not kept as a script)

# predictions with test-time augmentation (--m0 PixelSeal [--amp bf16], --m1 TrustMark, --m2 StegaStamp)
python predict.py --m1 runs/L_tm/last.pt --out preds/L_tm_tta --bs 2 --tta
python predict.py --m0 runs/L_ps/last.pt --out preds/L_ps_tta --bs 2 --amp bf16 --tta           # ... same for the other seven

# decoder-verified rectangles, then the three lists read by assemble.sh
python tscan.py trustmark test
python rect_pipe.py --pred preds/<ensemble> --method stegastamp --split test
METHOD=pixelseal python rect_more.py none preds/rectps_more_test.npy preds/<ensemble> ...
python make_rect_lists.py

# final assembly: ensemble -> threshold +1 on full-type images -> rectangles -> SAM snapping -> submission file
FULLSHIFT=1.0 KMAX=1 DELTA=-0.1,-0.1,-0.1 bash assemble.sh C16_pure objmasks2_all.npz -- \
    preds/L_tm_tta preds/Q_tm_102_tta preds/Q_tm_103_tta preds/L_ps_tta preds/Q_ps_102_tta preds/Q_ps_103_tta \
    preds/Q_ss_101_tta preds/Q_ss_102_tta preds/SS_clean_loc_tta
python common.py subq/pending/00_C16_pure.pt
```

What the best submission (score, file name and time on the first lines of this page) contains:

- **Ensemble**: plain mean of the logits of three long models per method. TrustMark: `L_tm` (extractor ResNet-50 +
  FPN on 3 scales, 20 000 steps), `Q_tm_102`, `Q_tm_103` (22 000 steps). PixelSeal: `L_ps` (extractor ConvNeXtV2,
  2 scales, 16 000 steps), `Q_ps_102`, `Q_ps_103` (18 000 steps). StegaStamp: `Q_ss_101`, `Q_ss_102` (U-Net with
  high-pass channels, 30 000 steps), `SS_clean_loc` (26 000 steps). All trained on synthetic composites + 20 % real
  images with the zoom augmentation, **without pseudo-labels of the test set**, and predicted with the 9-view TTA.
- **Thresholds**: logit > 0; on full-type images logit > 1. No rectangle post-processing.
- **Decoded rectangles** on the test set: 71 TrustMark (window scan, refined bit accuracy ≥ 0.85, corrected box),
  75 StegaStamp (segmenter proposals refined against the decoder, ≥ 0.85), 51 PixelSeal (≥ 0.68).
- **SAM snapping** with SAM ViT-H proposals (automatic mode, 32 points per side, at most 64 masks per image).

Steps done by hand: the three rectangle lists were first built with inline code; `make_rect_lists.py` is its
transcription, written afterwards and **not re-run**. The earlier short-model ensemble (`C2s`, built by a sequence of
`ens.py` / `route.py` / `merge_zoom.py` commands) is no longer in the best submission; it only served as a source of
proposals for the StegaStamp rectangles.

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
| 03:11 | C6 | 0.7474 | + SAM snapping of object-type predictions (ViT-B proposals) |
| 04:02 | C7 | 0.7614 | + long TrustMark and PixelSeal models, no pseudo-labels |
| 06:51 | C10 | 0.7684 | + SAM ViT-H proposals, the "object beats rectangle" rule, PixelSeal rectangles |
| 07:11 | C11_fs1 | 0.7705 | + threshold +1 on full-type images |
| 08:11 | C12 | 0.7871 | + long StegaStamp U-Nets, no pseudo-labels |
| 08:41 | C13 | 0.7887 | + a third StegaStamp U-Net and 9 more StegaStamp rectangles |
| 08:51 | C14 | 0.7891 | + test-time augmentation (9 translated views) |
| 09:01 | C15 | 0.7916 | + two more long PixelSeal models |
| 09:17 | C16 | 0.7945 | + two more long TrustMark models |
| 09:37 | C16_pure | **0.7965** | old short-model ensemble removed: three long models per method only |

<!-- TODO-FINAL: at 11:30, if the best public score changed: update the score line at the top (score, file name, time) and add one row to this table -->

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
  predictions on the test set stay those of the teacher (agreement 0.96): the model memorises its teacher on the test
  images. Public 0.760 with these models against 0.768 without. They were discarded as contaminated; every model of
  the final ensemble is trained without pseudo-labels.
- **Selecting SAM masks with the decoder, an area prior, SAM masks for the holes of full-type images** (oracle 0.3):
  no effect. **Window scan for StegaStamp**: impossible (an 8 % shift brings decoding back to chance); PixelSeal
  decodes for about one rectangle in three.
- **Warm restarts beyond 20 000 steps**: TrustMark +0.006, PixelSeal −0.011 on the control split (saturation).
- **Eroding the predictions** (`experiments/erode_pp.py`): −0.002 to −0.007. **Max-sum-rectangle post-processing**
  (`postproc.py`, variant `rect.8`): +0.001 to +0.002 locally, −0.006 on the public split.

Remaining weak spots: holes of full-type PixelSeal / TrustMark images (hole IoU 0.37 / 0.41, against 0.76 for
StegaStamp; holes under 1 % of the image are all missed) and StegaStamp rectangles the segmenter never proposes.
The local control split is small (94 images, standard error about 0.025) and not representative per cell.

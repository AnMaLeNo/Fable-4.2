# Fable 4.2 — European AI Championship, Paris 2026

Code, write-up and slides of team **Fable 4.2** for the 24-hour hackathon organised by CISPA / SprintML
(Paris, 21–22 November 2026): three tasks on the security and privacy of machine-learning models.

## Scores (public leaderboard; the final ranking uses a hidden split)

| Task | Metric | Baseline (public) | Our best public score (22 Nov, 11:20) |
|---|---|---|---|
| 1. Watermark localisation (images) | mean IoU | 0.2553 (all zeros) | **0.8013** (10:59) |
| 2. Model equality testing (LLM endpoints) | 0.1 × MCC family + 0.9 × MCC model | 0.10 (our first submission: families only) | **1.00** |
| 3. Unlearning (ViT-B/16) | forget score × retain score | 0.0 (unchanged model) | **0.4256** (10:38) |

Public ranking at 10:55: 4th of 44 teams overall, normalised score 0.94 (mean over the three tasks of the score
rescaled between the worst and the best team); 4th on task 1, 1.00 on task 2, 5th on task 3.

## The three solutions in one page

**Task 1 — Watermark localisation** → [task1_watermark_localization/](task1_watermark_localization/README.md)
Output a 128×128 mask of the watermarked area of an edited image, knowing the watermark method and the message.
We train one segmenter per method on synthetic composites made with the starter-kit embedders; for PixelSeal and
TrustMark the encoder is the pretrained watermark extractor itself. Three additions made the difference:
(1) we found that the real watermarked content had been enlarged by about 1.00–1.07 and cropped after embedding, and
reproduced it in training (a model trained with no real image goes from 0.07 to 0.83 mIoU on real fully watermarked
StegaStamp images); (2) rectangles are located by decoding crops against the known message and accepted only when the
bit accuracy is at least 0.85; (3) object-shaped predictions are snapped onto Segment Anything masks.
The final ensemble is twelve long trainings (three to five per method) with test-time translation averaging, all trained without
pseudo-labels of the test set (the variant with pseudo-labels memorised its teacher and was discarded).
Entry point: `task1_watermark_localization/assemble.sh`.

**Task 2 — Model equality testing** → [task2_model_equality/](task2_model_equality/README.md)
Say which of 33 black-box endpoints serve the same model, with 100 answers per endpoint. Instead of clustering answer
distributions, we identify what each endpoint is. We find the exact public base checkpoint of each family by the
likelihood of the collected answers (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B), which gives a local replica with exact
token probabilities. Every endpoint is then tested against the replica: fitted temperature, item-by-item goodness of
fit, likelihood ratio against an fp8-quantised replica, direction of deviation, watermark signature. System prompt,
persona, watermark and temperature endpoints are merged with the base only when positively identified.
Entry points: `task2_model_equality/client.py`, `items.py`, `jr/gscore.py`.

**Task 3 — Unlearning** → [task3_unlearning/](task3_unlearning/README.md)
Make a ViT-B/16 forget 3 500 unlabeled images so that it matches a model retrained without them. The metric compares
predictions image by image, so we build an explicit target for each image: "what would a model that never saw this
image predict?". Four families of proxy models answer (out-of-fold probes on frozen backbones, the original model on
img2img variations, students distilled on generated images only, the same architecture with its last blocks reset);
their consensus gives a difficulty rank and a wrong-class distribution, mapped onto the loss distribution of a
non-memorising model. The model is distilled onto these targets and anchored to itself on generated look-alike images.
Capping the true-class confidence on the images left correct brought 0.3724 → 0.4038. More families was not always
better: the best recipe drops the probes and adds a different signal, the original model after magnitude pruning
(0.4256; neighbouring variants 0.4214–0.4223).
Entry points: `task3_unlearning/targets.py`, `make_cap.py`, `train_v3.py`.

## Repository layout

```
README.md                         this page
task1_watermark_localization/     code + README (pipeline with file / function pointers, commands, results, negative results)
task2_model_equality/             idem
task3_unlearning/                 idem
slides/                           presentation (.pptx and .pdf), the script that builds it
```

Each task folder holds copies of the scripts that produce the final solution; `experiments/` sub-folders hold the
scripts behind the negative results we cite. No data, model weights, collected API answers or credentials are in the
repository; the scripts read `SERVER_URL` and `API_KEY` from the environment.

## How we worked

- All the engineering was done by **Claude agents**: one orchestrator, one long-running agent per task, and short-lived
  research agents for specific questions (for example an independent review of task 3 at 05:30, which produced the
  confidence cap, and a scouting agent at 03:00, which proposed the SAM snapping of task 1 and the "reset" proxy family
  of task 3).
- The **human team** provided the accesses (accounts, tokens, cluster sessions) and the decisions (how to apply the
  organisers' rules, compute usage, what gets published).
- Method shared by the three agents: one journal per task; one change per submission whenever the leaderboard is the
  only measurement (task 3, and the last ambiguous endpoints of task 2); local validation first whenever one exists
  (held-out real images for task 1, practice endpoints for task 2).

## Compute

- One node with 4× NVIDIA A100 (40 GB) on JURECA, inside the hackathon reservation, shared by the three tasks.
  For about 20 minutes on Saturday night a second node, outside the reservation, was also running; its jobs were cancelled and their results lost.
- One local workstation GPU (RTX 3080, 10 GB) for inference, post-processing, small fine-tunings and submissions.
- Task 2 needs no GPU except for scoring answers under the local replicas (models of 1 to 3 billion parameters).

## Honest limits

- **All scores above are public-split scores.** The final ranking is computed on a hidden split.
- **Task 2**: the public score saturates at 1.00 and takes few distinct values, so it cannot separate our remaining
  hypotheses. Four endpoints (01, 12, 21, 19) could not be settled by our statistical tests; we submitted the plausible
  configurations in decreasing order of probability. We also used single-change submissions as hypothesis tests, although
  the organisers warned that the public score may differ from the final one.
- **Task 3**: the server returns only the product of the factors; our reading of which factor limits the score is
  deduced, not measured. The targets were tuned with many public submissions. We do not know the hidden altered copies
  of the forget images; we only measured that near-identity training views score best.
- **Task 1**: our local control split is small (94 images) and noisy; the last assembly choices were also judged on the
  public score. Holes in fully watermarked PixelSeal / TrustMark images remain poorly detected.
- **Code state**: hackathon code, copied from the working tree at the end of the night with paths and credentials
  removed; it was not re-run end to end after this clean-up. Trained checkpoints, generated images and intermediate
  predictions are not included, and a few glue steps were inline commands (listed in each task README).
- Nothing here was optimised for compute cost: many short runs were launched in parallel to map the leaderboard response.

## Slides

[slides/Fable4.2_slides.pptx](slides/Fable4.2_slides.pptx) · [slides/Fable4.2_slides.pdf](slides/Fable4.2_slides.pdf)
(6 minutes, about 2 minutes per task).

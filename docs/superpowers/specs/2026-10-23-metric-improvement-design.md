# Metric Improvement — Design (rev 2)

**Date:** 2026-10-23
**Project:** ImgForge / ICT-4442 final report (due 31 Oct 2026, presentation 26–29 Oct)
**Branch:** `main`

## Intent

The interim report's best test macro-F1 is **0.804** (EfficientNet-B0 + ELA). Three defects in
the sweep are suppressing the numbers, one of them fatally:

1. **Swin-Tiny is collapsed, not merely weak.** All three seeds produce byte-identical
   macro-F1 0.4395 / accuracy 0.7843. 927/1182 authentic in the test split = 0.7843, so the
   model predicts "authentic" for every image, every seed. It receives a pretrained
   transformer at `lr=3e-4`, no warmup, no grad clipping.
2. **`rgb_ela` loses to `ela` in 6 of 6 CNN runs**, for two separate reasons:
   (a) the ELA channel is initialised from the mean of the pretrained RGB kernels
   (`_adapt_conv`), so it starts as a blurred copy of RGB rather than a residual filter;
   (b) `dataset.py` applies `ImageCompression(50–95)`, `GaussNoise`, `MotionBlur` and
   `GridDropout` to the RGB branch *after* ELA has been computed from the pristine file, so
   the two channels describe different images.
3. **Nothing is reproducible or analysable.** `evaluate.py` persists aggregate metrics only;
   no per-sample predictions exist, so McNemar cannot be computed. The 36 runs survive only
   as stdout inside `notebookae273fcfc6.ipynb`.

This design raises the metrics *and* produces the artefacts the report still owes.

## Goals

| # | Goal | Acceptance |
|---|------|------------|
| G1 | Swin-Tiny stops collapsing | mean test macro-F1 ≥ 0.80 across 3 seeds; predictions not degenerate |
| G2 | `rgb_ela` ≥ `ela` for both CNNs | in at least 2 of 3 seeds per architecture |
| G3 | EfficientNet-B0 crosses 0.90 | mean test macro-F1 ≥ 0.90 on its best input mode |
| G4 | ResNet-18 + BiLSTM reaches 0.88 | mean test macro-F1 ≥ 0.88 on its best input mode |
| G5 | Every run persists predictions | `predictions/*.npz` for all 36 runs, in a Kaggle Dataset |
| G6 | Confound is measured, not asserted | QF distribution test + QF-overlap-stratified F1, with insufficiency stated where it applies |
| G7 | Statistical comparison complete | McNemar + Holm within two declared families, bootstrap CIs |
| G8 | Localization described | Grad-CAM → mask → IoU/Dice on best model, reported descriptively with resolution caveat |
| G9 | Report matches the code | no `128×128` claim; tables regenerated from artefacts |

The reported number for every goal is the **val-tuned-threshold, TTA-on** figure. Argmax-0.5
and no-TTA results are logged alongside and reported in the table, but targets apply to the
tuned figure only. Targets are per-architecture on that architecture's best mode.

## Non-goals

- No new architectures, no new datasets, no pixel-level/localization *training*.
- The MLP baseline is **not** upgraded. It is deliberately convolution-free and trained from
  scratch; it is the lower-bound evidence that spatial inductive bias matters. It stays at
  ~0.55.
- The `frontend/`, `api/`, `app.py` demo path is untouched.

## Anti-goals (would invalidate the result even if numbers look good)

- Selecting the decision threshold, hyperparameters or CAM thresholds on the **test** split.
- Changing the train/val/test split, or regenerating it with a different effective ordering.
- Reporting only the confounded headline number while the confound analysis sits unused.
- Keeping a change that loses on the pilot just because it is in this document.

## Constraints

- Kaggle free GPU: 30 h/week (currently 0 used), ~12 h session cap, ~20 GB disk.
- Dataset fixed: CASIA 2.0 (`divg07/casia-20-image-tampering-detection-dataset`), 12,614
  images, split fixed at seed 42 → 9,448 / 1,181 / 1,182 (5,223/4,225 · 903/278 · 927/255).
- Local machine is CPU-only with a 65/10/8 smoke subset. All real training runs on Kaggle.
- Architecture-specific hyperparameters are tuned on validation only; the synopsis permits
  this asymmetry and the report must keep saying so.
- **Deadline priority:** the 31 Oct report is the hard target. The 26–29 Oct presentation uses
  whatever is finished by then, plus the interim numbers. The plan does not compress the
  matrix to beat the presentation.

## Approaches considered

- **A — Tuning only.** ~15 h GPU. Fixes G1, G5 but leaves the fusion defect and probably
  misses G3.
- **B — Tuning + dual-stream fusion + resolution ablation. (chosen)** ~20–25 h GPU. The only
  option that addresses the fusion defect without breaking the comparison.
- **C — B plus a second dataset or an explicit JPEG-quality input feature.** Contradicts the
  synopsis's "single protocol on CASIA 2.0" claim; not achievable in a week.

## Design

### 1. Dual-stream stem (`model_factory.py`)

Replace concatenation-at-input with sum-fusion of two independent stems feeding a shared trunk:

```
DualStem(rgb_conv, aux_conv).forward(x) = rgb_conv(x[:, :3]) + aux_conv(x[:, 3:])
```

| Arch | Stem replaced | `rgb_conv` | `aux_conv` |
|---|---|---|---|
| EfficientNet-B0 | `features[0][0]` (3×3, s2) | pretrained, unchanged | `Conv2d(1, 32, 3, s2)`, default init |
| ResNet-18 + BiLSTM | `conv1` (7×7, s2) | pretrained, unchanged | `Conv2d(1, 64, 7, s2)`, default init |
| Swin-Tiny | `features[0][0]` patch embed (4×4, s4) | pretrained, unchanged | `Conv2d(1, 96, 4, s4)`, default init |
| MLP | — | concat stays | — |

`rgb` mode is untouched. `ela` mode keeps a single 1-channel conv, but initialised from the
**mean over RGB channels** of the pretrained kernel (current code takes only the R channel).

Fusion is a *config value* recorded in the checkpoint (`cfg.fusion`), not a hardcoded choice,
so the concat baseline remains reproducible.

### 2. Augmentation policy by input mode (`data/dataset.py`)

Photometric augmentation of the RGB branch is incompatible with a precomputed ELA channel, and
that incompatibility is a large part of why `rgb_ela` lost to `ela`.

| Mode | Spatial (shared) | Photometric |
|---|---|---|
| `rgb` | Resize/RandomCrop, flip, small affine | unchanged from interim (brightness/contrast, noise, blur, JPEG) |
| `ela` | same | none |
| `rgb_ela` | same | **none** |

`rgb` keeps its existing policy so its number stays comparable to the interim run. The
asymmetry is deliberate and must be stated in the report: a residual channel cannot survive
recompression, so applying it only to the RGB arm would be a protocol error rather than a
fairness gain. This also removes `GridDropout` from the ELA stream, which was punching holes
in exactly the regions being classified.

### 3. Optimisation (`train.py`)

Three parameter groups, since pretrained, randomly-initialised and newly-attached modules
cannot share a learning rate:

| Group | Members | LR (CNN) | LR (Swin) |
|---|---|---|---|
| `backbone` | pretrained weights | 1e-4 | 2e-5 |
| `new` | `aux_conv` stem, classifier head | 1e-3 | 1e-3 |
| `new` | BiLSTM block (ResNet arm) | 1e-3 | — |

The 50× backbone:new ratio for Swin is intentional — a transformer backbone at 1e-4 is what
collapsed the interim run. Plus: linear warmup (5 epochs) then cosine, `clip_grad_norm_(1.0)`,
AdamW, weight decay 1e-2, class-weighted CE, early stop on val macro-F1 (patience 10), max 60
epochs, AMP enabled. All of it logged into the checkpoint `cfg` and `experiment_log.csv`.

### 4. Resolution

Train `Resize(288,288)` → `RandomCrop(256)`; eval `Resize(256,256)`. Pilot 224 vs 256 on one
config (`efficientnet_b0` / `ela` / seed 1) and freeze the winner on validation. The interim
report's `128×128` claim is wrong (the sweep ran 224) and must not survive into the final report.

### 5. Evaluation (`evaluate.py`)

- **Threshold tuning:** sweep 0.01–0.99 on **validation** probabilities, maximise macro-F1,
  store `decision_threshold`, apply to test. Argmax-0.5 also logged.
- **TTA:** horizontal-flip probability averaging; both with and without are logged.
- **Predictions:** write `predictions/{run_name}_{split}.npz` with `paths`, `labels`, `probs`.
  This is the artefact McNemar, the confound analysis and any re-thresholding consume.

### 6. Persistence and determinism (`run_sweep.py`, `data/prepare_dataset.py`)

- New flags: `--image-size`, `--lr-backbone`, `--lr-new`, `--warmup`, `--grad-clip`, `--tta`,
  `--fusion`.
- Require `--persist-dir`; copy `results/`, `models/`, `predictions/` after each run; print a
  warning if unset. Create a Kaggle Dataset from it before the session ends.
- **Split determinism guard:** `prepare_dataset.py` must sort paths (`sorted(Path.rglob(...))`)
  before shuffling — today the order is filesystem-dependent, so seed 42 alone does not
  reproduce the interim split. S0 additionally verifies the regenerated split against the
  interim counts (9,448 / 1,181 / 1,182 and 5,223 / 4,225 · 903 / 278 · 927 / 255) and a hash
  of the sorted path lists before any training starts. A mismatch stops the plan.

### 7. Analysis scripts (new, `analysis/`)

- `confound.py` — JPEG quality (via `read_jpeg_quality`) and ELA-intensity distributions per
  class; Mann-Whitney U. On the 80 locally available CASIA files the distributions are close
  to disjoint (Au median 82, IQR 73–90; Tp median 57, IQR 46–72), so a "quality-matched"
  correction is not statistically available. The script therefore reports **QF-stratified F1**
  (per-bin, only for bins containing both classes with n ≥ 20 each, labelled as such) and
  states the overlap insufficiency rather than manufacturing a corrected number. The
  quantified ELA-only vs RGB-only gap is the headline confound evidence.
- `mcnemar.py` — exact McNemar on npz predictions, **two declared families**, Holm within
  each: (i) within-architecture mode comparisons, 12 pairs; (ii) cross-architecture
  best-mode comparisons, 6 pairs. Reporting 630 comparisons and correcting across all of them
  would guarantee no significant result.
- `localization.py` — reuse `gradcam.py`; CAM target layer is the last conv/attention block
  (8×8 at 256 px); binarisation threshold selected on **validation** masks over 0.1–0.9 and
  applied to test; IoU/Dice reported descriptively with the 8×8 resolution caveat stated.

Outputs land in `docs/results/` as markdown tables ready for the report.

### 8. Dependency to verify before S2

Grad-CAM needs CASIA 2.0 ground-truth masks. The attached Kaggle dataset mounts `CASIA2/Au`
and `CASIA2/Tp`; mask availability is **unverified**. S0 checks for a `Groundtruth`/`*_gt`
directory. If absent, attach a mask dataset or downgrade G8 to "reported as unavailable" —
decided before the expensive runs, not after.

## Execution stages

| Stage | Where | Cost | Exit criterion |
|---|---|---|---|
| S0 | local CPU | 0 | masks confirmed; split verified against interim counts; dual-stem builds; 2-epoch smoke over all 12 arch×mode combos |
| S1 | Kaggle | ~4 h | 224 vs 256 chosen; Swin ELA no longer degenerate; dual-stem measured against concat on val (winner recorded either way) |
| S2 | Kaggle | ~20 h | 36/36 runs logged with npz predictions, persisted to a dataset |
| S3 | local CPU | 0 | confound, McNemar, Grad-CAM tables in `docs/results/` |
| S4 | local | 0 | report tables regenerated; `128×128` corrected |

**Abort vs adapt.** A pilot "failing" means *broken* (degenerate predictions, shape errors,
OOM), and only that stops the plan. A pilot losing to its baseline is a result: the loser is
recorded, the winner is used for S2, and the negative finding goes in the report. One retry
per pilot is budgeted (e.g. `aux_conv` moved between LR groups, a second Swin recipe).

**Conditional extra ablation.** A concat-vs-dual arm for the two CNNs (+6 runs, ~4 h) runs only
if ≥6 h of the weekly quota remains after S2. Otherwise the fusion claim rests on the S1 pilot
and the report says so.

## Expected outcome (honest)

| Architecture | Now | Expected after B |
|---|---|---|
| EfficientNet-B0 | 0.804 | 0.86–0.91 |
| ResNet-18 + BiLSTM | 0.774 | 0.84–0.89 |
| Swin-Tiny | 0.547 | 0.80–0.86 |
| MLP (lower bound) | 0.539 | ~0.55, unchanged |

"All four above 0.9" is not reachable: the MLP is designed to fail and Swin's deficit is partly
data scale. The honest ceiling on CASIA 2.0 is itself inflated by the JPEG confound — which is
why G6 exists and why the headline must be reported with that caveat rather than as a clean
forensic result.

## Decisions taken here (override if wrong)

- MLP stays a deliberately weak lower bound.
- Dual-stream over late fusion (cost; late fusion doubles the matrix).
- `rgb_ela` and `ela` get spatial-only augmentation; `rgb` keeps the interim policy.
- Reported number = val-tuned threshold + TTA; argmax-0.5 and no-TTA logged.
- Threshold, HPO and CAM threshold selected on validation, never test.
- Two McNemar families, not one 630-pair family.
- QF-matched correction abandoned as statistically unavailable; stratified analysis +
  stated insufficiency instead.
- Report deadline drives the plan; presentation takes whatever is ready.

## Left to the builder

Exact HPO search ranges for the pilot, whether to freeze BN statistics in the backbone, the
concrete CAM target layer per architecture, and the QF bin edges for the stratified analysis.

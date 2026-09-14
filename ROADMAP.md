# ImgForge → ICT-4442 Synopsis Roadmap

**Where you stand (14 Sep 2026):** work-plan milestones 2 (dataset/split/ELA pipeline — due 05 Sep) and 3 (shared harness — due 12 Sep) are already past due. This roadmap front-loads catching those up before the 18 Sep lit review / 22 Sep MLP+EfficientNet milestone.

## Phase 0 — Fix the ImgForge foundation (Done)
Do this before anything else — these bugs get baked into all 4 architectures otherwise.
- [x] `prepare_dataset.py`: perceptual-hash dedup + union-find source grouping before splitting
- [x] `dataset.py`: RGB/ELA alignment via Albumentations `additional_targets` (spatial transforms shared, pixel-level transforms RGB-only)
- [x] `ela.py`: `read_jpeg_quality()` + `--log-qf` confound logging, `ela_quality` configurable
- [x] `train.py`: `seed_everything()`, `--seed` CLI arg, stored in checkpoint metadata
- [ ] **Before Phase 3 runs**: dedup count and QF divergence were verified on a small synthetic fixture — rerun `prepare_dataset.py --log-qf` on the full 12,614-image CASIA set and record the real numbers before the ablation sweep starts

## Phase 1 — Shared harness (Done)
- [x] `build_model(arch, input_mode)` in `model_factory.py` — `efficientnet_b0` implemented for all 3 input modes; `mlp`/`resnet18_bilstm`/`swin_tiny` raise `NotImplementedError` pending Phase 2
- [x] Parameterized `train.py` (`--arch`, `--input-mode`, `--seed`, etc.) — class-weighted loss, macro-F1 selection, early stopping kept
- [x] Batch `evaluate.py` — accuracy/precision/recall/F1/ROC-AUC/PR-AUC + 95% bootstrap CI, reads arch/input_mode from checkpoint
- [x] `experiment_log.py` — appends to `results/experiment_log.csv`

## Phase 2 — Fill in the 3 missing architectures (Done)
- [x] MLP: 16×16 adaptive-avg-pool → flatten → FC, trained from scratch
- [x] ResNet-18 + BiLSTM: pretrained ResNet-18 trunk → raster-flatten feature map → BiLSTM(256) → mean pool → classifier
- [x] Swin-Tiny: ImageNet-pretrained, standard classification head
- [x] All three wired into `build_model()` in `model_factory.py`; all 12 arch×mode combos smoke-tested (2-epoch train + evaluate)

## Phase 3 — Ablation sweep (prep done, runs pending)
- [x] ELA cache: `ela.py` caches precomputed ELA to disk via `IMGFORGE_ELA_CACHE` env var — computed once per (image, quality), reused across all 36 runs
- [x] `run_sweep.py`: resumable driver over 4 archs × 3 input_modes × seeds [1,2,3]; seed-first iteration order; checks `experiment_log.csv` to skip completed combos; `--persist-dir` for Drive/Kaggle persistence; `--ela-cache-dir` for transparent ELA caching; ELA warmup before sweep starts
- [x] MLP confirmed: no conv layers — only AdaptiveAvgPool2d + Linear/ReLU/Dropout
- [x] Dry-run verified: 2 combos trained+evaluated+logged, resume correctly skipped them and ran only new combos
- [ ] 4 architectures × 3 input modes × 3 seeds = 36 runs (to execute on Colab/Kaggle)
- [ ] Confound check: compare JPEG-QF and ELA-intensity distributions, Authentic vs. Tampered; flag if ELA-only performance is suspiciously strong

## Phase 4 — Stats + localization
- [ ] McNemar's test, pairwise, Holm-corrected, across the best run per architecture
- [ ] `gradcam.py`: threshold CAM → binary mask → IoU/Dice vs. CASIA groundtruth masks (best model only)

## Phase 5 — Write-up
- [ ] Comparison table + CIs, McNemar results, confound findings, Grad-CAM IoU/Dice, error analysis
- [ ] Interim report (25 Sep) = Phase 0–2 results; full ablation + stats land in the final report (31 Oct)

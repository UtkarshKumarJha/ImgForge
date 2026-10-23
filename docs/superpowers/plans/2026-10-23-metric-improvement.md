# Metric Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise CASIA 2.0 test macro-F1 to ≥0.90 (EfficientNet-B0), ≥0.88 (ResNet-18+BiLSTM) and ≥0.80 (Swin-Tiny) by fixing the collapsed transformer, the broken ELA fusion, and the missing evaluation artefacts.

**Architecture:** Replace input concatenation with a dual-stream stem (separate 1-channel ELA conv summed with the pretrained RGB conv), give `rgb_ela`/`ela` spatial-only augmentation, train with three parameter groups plus warmup and clipping, and make `evaluate.py` emit per-sample predictions so McNemar, the confound analysis and threshold tuning become possible. All 36 runs re-execute on Kaggle against the verified interim split.

**Tech Stack:** Python 3, PyTorch, torchvision, Albumentations, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-metric-improvement-design.md`

## Global Constraints

- Test split must reproduce **9,448 / 1,181 / 1,182** images (**5,223/4,225 · 903/278 · 927/255** by class) at seed 42. Any mismatch stops the plan.
- Seeds are exactly `[1, 2, 3]`; matrix is 4 archs × 3 modes × 3 seeds = **36 runs**.
- Learning rates: backbone `1e-4` (CNN) / `2e-5` (Swin); all new/random params `1e-3`.
- Warmup **5** epochs, then cosine over `--num-epochs`; max **60** epochs; early stop patience **10** on val macro-F1; `clip_grad_norm_` **1.0**; AdamW weight decay **1e-2**; class-weighted CE.
- Resolution: train `Resize(288,288)` → `RandomCrop(256)`; eval `Resize(256,256)`. 224 remains valid for the pilot.
- Augmentation: `rgb` keeps photometric ops; `ela` and `rgb_ela` are **spatial-only**.
- Threshold, hyperparameters and CAM thresholds are selected on **validation** only.
- QF-stratified F1 requires **≥20 samples per class per bin**; otherwise the bin is skipped and reported as insufficient.
- McNemar: exact binomial, two families — (i) 12 within-architecture mode pairs, (ii) 6 cross-architecture best-mode pairs; Holm within each family.
- Reported number is val-tuned-threshold with TTA on; argmax-0.5 and no-TTA are logged alongside.
- Kaggle free tier: ≤30 GPU-hours/week, ≤12 h sessions. Local machine is CPU-only.

## Review Focus

1. **Small images (CASIA originals run to 240×160).** `Resize(288)` then `RandomCrop(256)` must pad or upscale cleanly; a crop larger than the resized image would crash every `rgb_ela` epoch. Expected: training proceeds and produces a 256×256 tensor.
2. **Non-JPEG inputs (CASIA contains BMP and TIFF).** `read_jpeg_quality` returns `None` for these; the confound analysis must exclude them and say how many, never crash or silently coerce to 0.
3. **Degenerate predictions (all one class).** The interim Swin bug produced byte-identical output for three seeds and would have made IoU, McNemar and bootstrap CIs quietly meaningless. Expected: the code reports it instead of returning `nan`.
4. **Single-class or tiny QF bins.** With authentic QF 73–90 and tampered 46–72, most bins are single-class; dividing by an empty bin would raise or return `nan`. Expected: bin skipped, count reported.
5. **Missing ground-truth mask for a tampered image.** Expected: that image is skipped and the coverage (`n_with_mask / n_tampered`) is printed, not an exception.

---

### Task 1: Test harness and split determinism

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_prepare_dataset.py`
- Modify: `data/prepare_dataset.py:26-27` (`collect_images`)
- Modify: `requirements.txt` (append `pytest`)

**Interfaces:**
- Consumes: nothing.
- Produces: `collect_images(folder, label) -> list[tuple[str, int]]` returning a **sorted** list; `split_fingerprint(csv_dir: str) -> dict` with keys `counts`, `class_counts`, `sha256`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_dataset.py
import os
from pathlib import Path
from data.prepare_dataset import collect_images, split_fingerprint


def test_collect_images_is_sorted_regardless_of_creation_order(tmp_path):
    d = tmp_path / "Au"
    d.mkdir()
    for name in ["c.jpg", "a.jpg", "b.jpg"]:
        (d / name).write_bytes(b"x")
    got = [Path(p).name for p, _ in collect_images(d, 0)]
    assert got == ["a.jpg", "b.jpg", "c.jpg"]


def test_split_fingerprint_reports_counts_and_hash(tmp_path):
    for split, labels in {"train": [0, 0, 1], "val": [0], "test": [1]}.items():
        with open(tmp_path / f"{split}.csv", "w") as f:
            f.write("path,label\n")
            for i, lab in enumerate(labels):
                f.write(f"/x/{split}_{i}.jpg,{lab}\n")
    fp = split_fingerprint(str(tmp_path))
    assert fp["counts"] == {"train": 3, "val": 1, "test": 1}
    assert fp["class_counts"]["train"] == {"0": 2, "1": 1}
    assert len(fp["sha256"]) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'split_fingerprint'` (and the sorted assertion fails on `c,a,b`).

- [ ] **Step 3: Implement**

In `collect_images`, build the list from `sorted(Path(folder).rglob("*"))` so filesystem ordering cannot change the shuffle input. Add `split_fingerprint(csv_dir)` that reads `train.csv`/`val.csv`/`test.csv`, returns per-split row counts, per-split class counts keyed by `"0"`/`"1"`, and the sha256 of the concatenated sorted `path` columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prepare_dataset.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/ data/prepare_dataset.py requirements.txt
git commit -m "test: pin dataset split determinism and add split fingerprint"
```

---

### Task 2: Dual-stream stem

**Files:**
- Modify: `model_factory.py:15-37` (`_adapt_conv`), `:42-50`, `:111-119`, `:124-130`, `:143-150`
- Create: `tests/test_model_factory.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class DualStem(nn.Module)` with `__init__(self, rgb_conv: nn.Conv2d, aux_conv: nn.Conv2d)` and `forward(x: Tensor) -> Tensor` computing `rgb_conv(x[:, :3]) + aux_conv(x[:, 3:])`.
  - `build_model(arch: str, input_mode: str, num_classes: int = 2, fusion: str = "dual") -> nn.Module`, `fusion ∈ {"dual", "concat"}`; `fusion` is inert for `rgb`/`ela`.
  - `_adapt_conv(old_conv, in_channels)` when `in_channels < old_conv.in_channels` uses the **mean across RGB** of the pretrained kernels.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_factory.py
import torch
from model_factory import build_model, VALID_ARCHS, VALID_MODES


def test_all_arch_mode_combos_forward():
    for arch in sorted(VALID_ARCHS):
        for mode in sorted(VALID_MODES):
            ch = {"rgb": 3, "ela": 1, "rgb_ela": 4}[mode]
            m = build_model(arch, mode).eval()
            with torch.no_grad():
                out = m(torch.randn(1, ch, 224, 224))
            assert out.shape == (1, 2), (arch, mode, out.shape)


def test_dual_stem_sums_rgb_and_aux_branches():
    import torch.nn as nn
    from model_factory import DualStem
    rgb = nn.Conv2d(3, 8, 3, padding=1)
    aux = nn.Conv2d(1, 8, 3, padding=1)
    x = torch.randn(1, 4, 8, 8)
    expect = rgb(x[:, :3]) + aux(x[:, 3:])
    assert torch.allclose(DualStem(rgb, aux)(x), expect)


def test_ela_only_conv_initialised_from_rgb_mean():
    m = build_model("efficientnet_b0", "ela", fusion="concat")
    from torchvision import models
    ref = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    got = m.features[0][0].weight
    assert got.shape[1] == 1
    assert torch.allclose(got[:, 0], ref.features[0][0].weight.mean(dim=1), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'DualStem'`; the ELA test fails because the current code takes only channel 0.

- [ ] **Step 3: Implement**

`DualStem.forward` slices and sums. In each builder, when `fusion == "dual"` and `input_mode == "rgb_ela"`, wrap the pretrained stem and a fresh `Conv2d(1, out_ch, k, s, p)` in `DualStem`, keeping the original module's stride/padding. Concat path keeps today's `_adapt_conv(conv, 4)`. Fix `_adapt_conv`'s narrowing branch to `old_conv.weight.mean(dim=1, keepdim=True)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_factory.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add model_factory.py tests/test_model_factory.py
git commit -m "feat: dual-stream ELA stem and mean-initialised 1-channel ELA conv"
```

---

### Task 3: Augmentation policy by input mode

**Files:**
- Modify: `data/dataset.py:18-47` (`get_transforms`), `:55` (call site)
- Create: `tests/test_dataset_transforms.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_transforms(split: str, image_size: int = 224, input_mode: str = "rgb") -> tuple[A.Compose, A.Compose | None]`. For `input_mode in {"ela", "rgb_ela"}` the returned pixel transform is `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_transforms.py
import numpy as np
from data.dataset import get_transforms


def test_rgb_keeps_photometric_augmentation():
    _, pixel = get_transforms("train", 256, "rgb")
    assert pixel is not None


def test_ela_modes_have_no_photometric_augmentation():
    for mode in ("ela", "rgb_ela"):
        _, pixel = get_transforms("train", 256, mode)
        assert pixel is None, mode


def test_spatial_transform_accepts_small_images_and_returns_square():
    spatial, _ = get_transforms("train", 256, "rgb_ela")
    rgb = np.random.randint(0, 255, (160, 240, 3), dtype=np.uint8)
    ela = np.random.randint(0, 255, (160, 240, 3), dtype=np.uint8)
    out = spatial(image=rgb, ela=ela)
    assert out["image"].shape[:2] == (256, 256)
    assert out["ela"].shape[:2] == (256, 256)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_transforms.py -v`
Expected: FAIL — `get_transforms() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement**

Add the `input_mode` parameter defaulting to `"rgb"`; return `None` for the pixel transform when the mode is `ela` or `rgb_ela`. Keep `A.Resize(288, 288)` before `A.RandomCrop(256)` for training when `image_size == 256`, `Resize(image_size, image_size)` otherwise. Pass `input_mode` from `ForgeryDataset.__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_transforms.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py tests/test_dataset_transforms.py
git commit -m "fix: spatial-only augmentation for ELA input modes"
```

---

### Task 4: Parameter groups, warmup, clipping, AMP

**Files:**
- Modify: `train.py:29-45` (args), `:82-96` (model/optim), `:98-142` (loop), `:154-163` (checkpoint)
- Create: `tests/test_train_optim.py`

**Interfaces:**
- Consumes: `build_model(..., fusion=...)` from Task 2.
- Produces:
  - `build_param_groups(model: nn.Module, arch: str, lr_backbone: float, lr_new: float) -> list[dict]` — three groups when the arch has new modules, each `{"params": [...], "lr": float}`; `aux_conv`, the classifier head and the BiLSTM must appear in an `lr_new` group, pretrained backbone weights in the `lr_backbone` group, and **no parameter may appear twice**.
  - `make_scheduler(optimizer, warmup_epochs: int, total_epochs: int)` — linear warmup then cosine; at epoch 0 the LR is `base_lr / warmup_epochs` of its group.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_optim.py
import torch
from model_factory import build_model
from train import build_param_groups, make_scheduler


def _ids(groups):
    seen = []
    for g in groups:
        seen += [id(p) for p in g["params"]]
    return seen


def test_no_parameter_is_in_two_groups():
    for arch in ["efficientnet_b0", "resnet18_bilstm", "swin_tiny", "mlp"]:
        m = build_model(arch, "rgb_ela")
        ids = _ids(build_param_groups(m, arch, 1e-4, 1e-3))
        assert len(ids) == len(set(ids)), arch


def test_aux_stem_is_in_the_new_group():
    m = build_model("efficientnet_b0", "rgb_ela")
    groups = build_param_groups(m, "efficientnet_b0", 1e-4, 1e-3)
    aux_id = id(m.features[0][0].aux_conv.weight)
    new_ids = [id(p) for g in groups if g["lr"] == 1e-3 for p in g["params"]]
    assert aux_id in new_ids


def test_warmup_ramps_then_cosine_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p], "lr": 1e-3}])
    sched = make_scheduler(opt, warmup_epochs=5, total_epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[4]
    assert lrs[5] > lrs[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_optim.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_param_groups'`.

- [ ] **Step 3: Implement**

Split parameters by identity against a set built from the pretrained backbone module; everything else is "new". Replace `CosineAnnealingLR` with a `LambdaLR` implementing `(epoch+1)/warmup` for `epoch < warmup`, then cosine. Add `--lr-backbone`, `--lr-new`, `--warmup`, `--grad-clip`, `--fusion`, `--tta` args; wrap the forward/loss in `torch.amp.autocast` under CUDA; call `clip_grad_norm_` before `optimizer.step()`; store all values in the checkpoint `cfg`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_optim.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_optim.py
git commit -m "feat: discriminative learning rates, warmup, clipping and AMP"
```

---

### Task 5: Threshold tuning, TTA, prediction artefacts

**Files:**
- Modify: `evaluate.py:55-121` (`evaluate`), `:124-150` (`main`)
- Modify: `experiment_log.py:9-18` (`LOG_COLS`)
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `ForgeryDataset` order (`ds.samples[i][0]` is the path for sample `i`).
- Produces:
  - `tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, lo: float = 0.01, hi: float = 0.99, step: float = 0.01) -> float`
  - `predict(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]` returning `(y_true, y_prob, paths)`; with `tta=True` the probability is the mean of the image and its horizontal flip.
  - `save_predictions(path: str, paths: list[str], y_true: np.ndarray, y_prob: np.ndarray) -> None` writing an `.npz` with keys `paths`, `labels`, `probs`.
  - `is_degenerate(pred: np.ndarray) -> bool` — `True` when every prediction is the same class.
  - `evaluate(..., tta: bool = False, threshold: float | None = None)` returns a dict that also contains `decision_threshold`, `f1_macro_argmax`, `f1_macro_tuned`, `f1_macro_tuned_no_tta` and `degenerate: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
import pytest
from evaluate import tune_threshold, save_predictions


def test_tune_threshold_maximises_macro_f1_on_validation():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.55, 0.60, 0.65, 0.70, 0.80])
    t = tune_threshold(y, p)
    from sklearn.metrics import f1_score
    achieved = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
    assert achieved == pytest.approx(1.0)


def test_save_predictions_roundtrip(tmp_path):
    f = tmp_path / "r.npz"
    save_predictions(str(f), ["a.jpg", "b.jpg"], np.array([0, 1]), np.array([0.2, 0.8]))
    d = np.load(f, allow_pickle=False)
    assert list(d["paths"]) == ["a.jpg", "b.jpg"]
    assert d["labels"].tolist() == [0, 1]
    assert d["probs"].tolist() == pytest.approx([0.2, 0.8])


def test_degenerate_predictions_are_flagged():
    from evaluate import is_degenerate
    assert is_degenerate(np.array([0, 0, 0, 0]))
    assert not is_degenerate(np.array([0, 0, 1, 1]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ImportError: cannot import name 'tune_threshold'`.

- [ ] **Step 3: Implement**

`tune_threshold` sweeps `np.arange(lo, hi + 1e-9, step)` and returns the value maximising `f1_score(..., average="macro")`. `is_degenerate(pred)` returns `len(np.unique(pred)) == 1`. `evaluate` computes metrics at argmax 0.5 and at the tuned threshold, writes `predictions/{run_name}_{split}.npz`, and passes the tuned decision into `experiment_log`. Append `decision_threshold`, `f1_macro_argmax`, `f1_macro_tuned`, `degenerate` to `LOG_COLS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluate.py experiment_log.py tests/test_evaluate.py
git commit -m "feat: val-tuned threshold, TTA and per-sample prediction artefacts"
```

---

### Task 6: Sweep driver flags and persistence

**Files:**
- Modify: `run_sweep.py:121-193` (`run_combo`), `:196-250` (`main`)
- Create: `tests/test_run_sweep_cmds.py`

**Interfaces:**
- Consumes: the CLI flags added in Task 4, and the `predictions/` directory written in Task 5.
- Produces:
  - `build_train_cmd(args, arch: str, input_mode: str, seed: int) -> list[str]`
  - `build_eval_cmd(args, ckpt_path: str, json_path: str) -> list[str]`
  - `persist_to_drive` also copies `predictions/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_sweep_cmds.py
from run_sweep import build_train_cmd, build_eval_cmd


class A:
    data_dir = "data"; num_epochs = 60; batch_size = 16; num_workers = 2
    ela_quality = 75; out_dir = "models"; wandb = False
    image_size = 256; lr_backbone = 1e-4; lr_new = 1e-3; warmup = 5
    grad_clip = 1.0; fusion = "dual"; tta = True


def test_train_cmd_carries_every_new_flag():
    cmd = build_train_cmd(A, "swin_tiny", "rgb_ela", 2)
    for flag, val in [("--image-size", "256"), ("--lr-backbone", "0.0001"),
                      ("--lr-new", "0.001"), ("--warmup", "5"),
                      ("--grad-clip", "1.0"), ("--fusion", "dual")]:
        assert flag in cmd and cmd[cmd.index(flag) + 1] == val, flag
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "2"


def test_eval_cmd_enables_tta_and_json_output():
    cmd = build_eval_cmd(A, "models/m.pth", "results/m.json")
    assert "--tta" in cmd
    assert cmd[cmd.index("--out-json") + 1] == "results/m.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_sweep_cmds.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_train_cmd'`.

- [ ] **Step 3: Implement**

Extract the two command builders from `run_combo` verbatim plus the new flags. Add `--image-size`, `--lr-backbone`, `--lr-new`, `--warmup`, `--grad-clip`, `--fusion`, `--tta` to `main`'s parser and thread them through. Extend `persist_to_drive` to copy `predictions/`. Warn loudly when `--persist-dir` is unset.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_sweep_cmds.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add run_sweep.py tests/test_run_sweep_cmds.py
git commit -m "feat: sweep driver threads new flags and persists predictions"
```

---

### Task 7: Confound analysis

**Files:**
- Create: `analysis/__init__.py`, `analysis/confound.py`, `tests/test_confound.py`

**Interfaces:**
- Consumes: `data.ela.read_jpeg_quality(path)`; `.npz` predictions from Task 5.
- Produces:
  - `mann_whitney(a: list[int], b: list[int]) -> dict` with keys `n_a`, `n_b`, `median_a`, `median_b`, `p_value`.
  - `stratified_f1(y_true, y_pred, qf: list[int | None], min_per_class: int = 20) -> dict` with keys `bins` (list of `{lo, hi, n_ok, n_tampered, f1}`), `usable_bins`, `insufficient_bins`.
  - CLI: `python -m analysis.confound --data-dir data --predictions predictions/*.npz --out docs/results/confound.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confound.py
import numpy as np
from analysis.confound import mann_whitney, stratified_f1


def test_mann_whitney_separates_disjoint_quality_groups():
    r = mann_whitney([80, 82, 85, 88], [50, 55, 60, 65])
    assert r["p_value"] < 0.05
    assert r["median_a"] > r["median_b"]


def test_none_quality_values_are_excluded_not_coerced():
    r = mann_whitney([80, None, 85], [50, None, 60])
    assert r["n_a"] == 2 and r["n_b"] == 2


def test_stratified_f1_skips_single_class_and_small_bins():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0, 0, 1, 1, 1, 0])
    qf = [50, 50, 50, 80, 80, 80]      # two bins, each single-class
    r = stratified_f1(y, p, qf, min_per_class=1)
    assert r["usable_bins"] == 0
    assert r["insufficient_bins"] == 2


def test_stratified_f1_computes_when_a_bin_has_both_classes():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    qf = [50, 50, 50, 50, 60, 60, 60, 60]
    r = stratified_f1(y, p, qf, min_per_class=2)
    assert r["usable_bins"] == 2
    assert all(0.0 <= b["f1"] <= 1.0 for b in r["bins"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_confound.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis'`.

- [ ] **Step 3: Implement**

`mann_whitney` filters `None`, calls `scipy.stats.mannwhitneyu(..., alternative="two-sided")` (add `scipy` to `requirements.txt`). `stratified_f1` buckets quality into 10-wide bins, skips bins where either class has fewer than `min_per_class` samples, and macro-averages F1 across usable bins. The CLI writes a markdown table including how many images had no readable QF.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_confound.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/ tests/test_confound.py requirements.txt
git commit -m "feat: JPEG-quality confound analysis with insufficiency reporting"
```

---

### Task 8: McNemar with Holm correction

**Files:**
- Create: `analysis/mcnemar.py`, `tests/test_mcnemar.py`

**Interfaces:**
- Consumes: `.npz` predictions from Task 5.
- Produces:
  - `mcnemar_exact(b: int, c: int) -> float` — two-sided exact binomial p-value at 0.5.
  - `mcnemar_pair(y_true, pred_a, pred_b) -> tuple[int, int, float]` returning `(b, c, p)` where `b` = A correct and B wrong.
  - `holm(pvals: dict[str, float]) -> dict[str, float]`.
  - `within_arch_pairs(run_names: list[str]) -> list[tuple[str, str]]` and `cross_arch_pairs(best_run_per_arch: dict[str, str]) -> list[tuple[str, str]]`, producing 12 and 6 pairs respectively.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcnemar.py
import numpy as np
from analysis.mcnemar import mcnemar_exact, mcnemar_pair, holm, within_arch_pairs


def test_exact_p_value_matches_binomial_for_known_counts():
    assert abs(mcnemar_exact(10, 2) - 0.038574) < 1e-5
    assert mcnemar_exact(5, 5) == 1.0


def test_pair_counts_discordant_cells_only():
    y = np.array([1, 1, 0, 0])
    a = np.array([1, 1, 0, 1])   # wrong on last
    b = np.array([1, 0, 0, 0])   # wrong on second
    bb, cc, _ = mcnemar_pair(y, a, b)
    assert (bb, cc) == (1, 1)


def test_holm_is_monotone_and_bounded():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert out["a"] <= out["b"] <= out["c"]
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_within_arch_pairs_yields_twelve():
    runs = [f"{a}_{m}_s1" for a in ["efficientnet_b0", "mlp", "resnet18_bilstm", "swin_tiny"]
            for m in ["ela", "rgb", "rgb_ela"]]
    assert len(within_arch_pairs(runs)) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcnemar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.mcnemar'`.

- [ ] **Step 3: Implement**

`mcnemar_exact` uses `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue` clamped to `[0, 1]`; if `b + c == 0` return `1.0`. `holm` sorts ascending and applies `max(prev, (m - i) * p)` with cumulative minimum from the end. Parse a run name by matching the **longest** prefix in `VALID_ARCHS` (names contain underscores: `resnet18_bilstm_ela_s1`) and then the mode against `VALID_MODES` — do not split on `_` positionally.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcnemar.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/mcnemar.py tests/test_mcnemar.py
git commit -m "feat: exact McNemar with Holm correction over two declared families"
```

---

### Task 9: Localization IoU/Dice

**Files:**
- Create: `analysis/localization.py`, `tests/test_localization.py`
- Modify: `gradcam.py` (add `get_gradcam_target_layer(model, arch) -> nn.Module`)

**Interfaces:**
- Consumes: `gradcam.GradCAM`; checkpoints from Task 4.
- Produces:
  - `cam_to_mask(cam: np.ndarray, threshold: float) -> np.ndarray` — boolean array, CAM resized to `gt.shape`.
  - `iou_dice(pred_mask, gt_mask) -> tuple[float, float]`, returning `(0.0, 0.0)` for empty ground truth.
  - `choose_threshold(cams, gts, grid: np.ndarray) -> tuple[float, float]` — threshold maximising mean IoU, evaluated on validation masks only.
  - `coverage(tampered_paths, mask_paths) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_localization.py
import numpy as np
from analysis.localization import cam_to_mask, iou_dice, choose_threshold, coverage


def test_perfect_and_disjoint_masks():
    gt = np.zeros((8, 8), bool); gt[2:6, 2:6] = True
    assert iou_dice(gt, gt) == (1.0, 1.0)
    assert iou_dice(~gt, gt) == (0.0, 0.0)


def test_empty_ground_truth_returns_zero_not_nan():
    gt = np.zeros((8, 8), bool)
    iou, dice = iou_dice(np.zeros((8, 8), bool), gt)
    assert iou == 0.0 and dice == 0.0


def test_threshold_selection_picks_the_separable_value():
    cam = np.zeros((8, 8)); cam[2:6, 2:6] = 1.0
    gt = np.zeros((8, 8), bool); gt[2:6, 2:6] = True
    best, score = choose_threshold([cam], [gt], np.array([0.3, 0.5, 0.7]))
    assert best in (0.3, 0.5, 0.7) and score == 1.0


def test_coverage_reports_fraction_not_exception():
    assert coverage(["a", "b", "c"], ["a"]) == 1 / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_localization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.localization'`.

- [ ] **Step 3: Implement**

Use `cv2.resize(cam, (gt.shape[1], gt.shape[0]))` before thresholding. `iou_dice` computes `|∩|/|∪|` and `2|∩|/(|A|+|B|)`, guarding zero denominators. `get_gradcam_target_layer` returns `features[-1]` for both EfficientNet and Swin, and the last block of `layer4` for ResNet. The CLI skips tampered images without a mask and prints coverage.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_localization.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/localization.py gradcam.py tests/test_localization.py
git commit -m "feat: Grad-CAM mask agreement with val-selected threshold"
```

---

### Task 10: S0 verification gate and docs

**Files:**
- Create: `scripts/verify_setup.py`, `tests/test_verify_setup.py`
- Modify: `ROADMAP.md` (tick phases 3–4), `docs/results/*.md` (regenerate after S3)

**Interfaces:**
- Consumes: `split_fingerprint` (Task 1).
- Produces: `verify_setup(data_dir: str, raw_dir: str) -> dict` with keys `split_ok`, `mask_available`, `counts`, `problems: list[str]`; exit code 1 when `problems` is non-empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_setup.py
import json
from scripts.verify_setup import verify_setup


def test_mismatched_counts_are_reported_as_problems(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    with open(d / "train.csv", "w") as f:
        f.write("path,label\n/x/1.jpg,0\n")
    for s in ("val", "test"):
        (d / f"{s}.csv").write_text("path,label\n")
    r = verify_setup(str(d), str(tmp_path / "missing_raw"))
    assert r["split_ok"] is False
    assert any("train" in p for p in r["problems"])


def test_expected_counts_pass(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    spec = {"train": (9448, 5223), "val": (1181, 903), "test": (1182, 927)}
    for split, (n, n_auth) in spec.items():
        with open(d / f"{split}.csv", "w") as f:
            f.write("path,label\n")
            for i in range(n):
                f.write(f"/x/{i}.jpg,{0 if i < n_auth else 1}\n")
    r = verify_setup(str(d), str(tmp_path / "missing_raw"))
    assert r["split_ok"] is True
    assert r["mask_available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verify_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`.

- [ ] **Step 3: Implement**

Compare `split_fingerprint` output against the Global Constraints table; record each mismatch as a `problems` string. Probe `raw_dir` for a `Groundtruth`/`*_gt` directory and set `mask_available` accordingly. Print a JSON summary and exit 1 on problems.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests, 24 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_setup.py tests/test_verify_setup.py ROADMAP.md
git commit -m "feat: S0 verification gate for split integrity and mask availability"
```

---

## Execution notes

- S0 runs tasks 1–10 locally and ends with `python scripts/verify_setup.py --data-dir data --raw-dir data/raw` on Kaggle, where the real split exists.
- S1/S2 are Kaggle-only and need no new code after Task 10. S1 opens with the resolution pilot — `run_sweep.py --archs efficientnet_b0 --input-modes ela --seeds 1 --image-size 224` and the same at 256 — and keeps whichever wins on validation. Then the Swin ELA check (`--archs swin_tiny --input-modes ela --seeds 1`) confirms the collapse is gone; if it is still degenerate, change the backbone LR (2e-5 → 1e-5) and retry once.
- S3 runs `python -m analysis.confound`, `python -m analysis.mcnemar`, `python -m analysis.localization` against the persisted `predictions/`.
- S4 regenerates `docs/results/*.md` from the S3 output and rewrites every number in the Part B report that came from the interim 36-run table, including replacing the `128×128` resolution claim with the size actually used.

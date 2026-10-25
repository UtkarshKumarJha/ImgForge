# efficientnet_b0 — CASIA 2.0 sweep

Protocol: 224 px, 60-epoch cap with early stopping (patience 10), batch 16, dual-stream stem, mode-aware augmentation, discriminative LRs. Reported number is the **val-tuned threshold, TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and no-TTA are shown for comparison.

Measured on the held-out test split: 1,181 test images (878 authentic / 303 tampered); split fingerprint `79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`.

### efficientnet_b0

| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | macro-F1 (no TTA) | Accuracy | ROC-AUC |
|---|---:|---|---|---|---|---|
| rgb | 3 | 0.6709 ± 0.0031 | 0.6776 ± 0.0025 | 0.6729 ± 0.0059 | 0.7324 ± 0.0082 | 0.7729 ± 0.0028 |
| ela | 3 | 0.8376 ± 0.0169 | 0.8320 ± 0.0192 | 0.8307 ± 0.0231 | 0.8718 ± 0.0129 | 0.9327 ± 0.0159 |
| rgb_ela | 3 | 0.7264 ± 0.0109 | 0.7260 ± 0.0031 | 0.7220 ± 0.0010 | 0.7711 ± 0.0055 | 0.8275 ± 0.0021 |

### efficientnet_b0 - per seed (val-tuned threshold, TTA on)

| Input mode | Seed | macro-F1 | 95% bootstrap CI |
|---|---:|---:|---|
| rgb | 1 | 0.6708 | 0.639-0.699 |
| rgb | 2 | 0.6679 | 0.636-0.698 |
| rgb | 3 | 0.6741 | 0.644-0.701 |
| ela | 1 | 0.8185 | 0.793-0.843 |
| ela | 2 | 0.8438 | 0.820-0.866 |
| ela | 3 | 0.8506 | 0.827-0.874 |
| rgb_ela | 1 | 0.7346 | 0.707-0.761 |
| rgb_ela | 2 | 0.7140 | 0.684-0.742 |
| rgb_ela | 3 | 0.7307 | 0.702-0.757 |

### Documented retry

| Run | Resolution | macro-F1 (tuned, TTA) | Accuracy |
|---|---:|---|---|
| efficientnet_b0_ela_p320_s1 | 320 | 0.8680 | 0.8899 |
| efficientnet_b0_ela_p320_s2 | 320 | 0.8674 | 0.8908 |
| efficientnet_b0_ela_p320_s3 | 320 | 0.8648 | 0.8942 |
| **retry mean** | | **0.8667 ± 0.0017** | |

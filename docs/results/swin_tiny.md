# swin_tiny — CASIA 2.0 sweep

Protocol: 224 px, 60-epoch cap with early stopping (patience 10), batch 16, dual-stream stem, mode-aware augmentation, discriminative LRs. Reported number is the **val-tuned threshold, TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and no-TTA are shown for comparison.

Measured on the held-out test split: 1,181 test images (878 authentic / 303 tampered); split fingerprint `79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`.

### swin_tiny

| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | macro-F1 (no TTA) | Accuracy | ROC-AUC |
|---|---:|---|---|---|---|---|
| rgb | 3 | 0.7432 ± 0.0013 | 0.7411 ± 0.0069 | 0.7415 ± 0.0017 | 0.7818 ± 0.0040 | 0.8512 ± 0.0078 |
| ela | 3 | 0.8440 ± 0.0135 | 0.8513 ± 0.0093 | 0.8518 ± 0.0090 | 0.8798 ± 0.0066 | 0.9427 ± 0.0026 |
| rgb_ela | 3 | 0.7992 ± 0.0279 | 0.7817 ± 0.0145 | 0.8022 ± 0.0305 | 0.8307 ± 0.0237 | 0.8911 ± 0.0170 |

### swin_tiny - per seed (val-tuned threshold, TTA on)

| Input mode | Seed | macro-F1 | 95% bootstrap CI |
|---|---:|---:|---|
| rgb | 1 | 0.7447 | 0.717-0.770 |
| rgb | 2 | 0.7426 | 0.715-0.770 |
| rgb | 3 | 0.7422 | 0.714-0.769 |
| ela | 1 | 0.8390 | 0.815-0.862 |
| ela | 2 | 0.8338 | 0.809-0.857 |
| ela | 3 | 0.8593 | 0.837-0.881 |
| rgb_ela | 1 | 0.8122 | 0.788-0.837 |
| rgb_ela | 2 | 0.8182 | 0.793-0.842 |
| rgb_ela | 3 | 0.7671 | 0.740-0.793 |

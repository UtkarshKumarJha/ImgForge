# swin_tiny — CASIA 2.0 sweep

Protocol: 224 px, 60-epoch cap with early stopping (patience 10), batch 16, dual-stream stem, mode-aware augmentation, discriminative LRs. Reported number is the **val-tuned threshold, TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and no-TTA are shown for comparison.

Measured on the held-out test split: 1,181 test images (878 authentic / 303 tampered); split fingerprint `79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`.

### swin_tiny

| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | macro-F1 (no TTA) | Accuracy | ROC-AUC |
|---|---:|---|---|---|---|---|
| rgb | 2 | 0.7437 ± 0.0015 | 0.7406 ± 0.0097 | 0.7414 ± 0.0024 | 0.7802 ± 0.0042 | 0.8467 ± 0.0008 |
| ela | 2 | 0.8364 ± 0.0037 | 0.8561 ± 0.0059 | 0.8541 ± 0.0115 | 0.8760 ± 0.0006 | 0.9412 ± 0.0009 |
| rgb_ela | 2 | 0.8152 ± 0.0042 | 0.7891 ± 0.0098 | 0.8197 ± 0.0011 | 0.8442 ± 0.0048 | 0.9009 ± 0.0007 |

### swin_tiny - per seed (val-tuned threshold, TTA on)

| Input mode | Seed | macro-F1 | 95% bootstrap CI |
|---|---:|---:|---|
| rgb | 1 | 0.7447 | 0.717-0.770 |
| rgb | 2 | 0.7426 | 0.715-0.770 |
| ela | 1 | 0.8390 | 0.815-0.862 |
| ela | 2 | 0.8338 | 0.809-0.857 |
| rgb_ela | 1 | 0.8122 | 0.788-0.837 |
| rgb_ela | 2 | 0.8182 | 0.793-0.842 |

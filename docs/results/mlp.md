# mlp — CASIA 2.0 sweep

Protocol: 224 px, 60-epoch cap with early stopping (patience 10), batch 16, dual-stream stem, mode-aware augmentation, discriminative LRs. Reported number is the **val-tuned threshold, TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and no-TTA are shown for comparison.

Measured on the held-out test split: 1,181 test images (878 authentic / 303 tampered); split fingerprint `79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`.

### mlp

| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | macro-F1 (no TTA) | Accuracy | ROC-AUC |
|---|---:|---|---|---|---|---|
| rgb | 3 | 0.6128 ± 0.0020 | 0.5952 ± 0.0120 | 0.6145 ± 0.0099 | 0.6927 ± 0.0088 | 0.6649 ± 0.0024 |
| ela | 3 | 0.4910 ± 0.0647 | 0.4478 ± 0.0373 | 0.4940 ± 0.0655 | 0.7398 ± 0.0285 | 0.5817 ± 0.1050 |
| rgb_ela | 3 | 0.5995 ± 0.0096 | 0.5878 ± 0.0018 | 0.5962 ± 0.0094 | 0.6771 ± 0.0168 | 0.6585 ± 0.0089 |

### mlp - per seed (val-tuned threshold, TTA on)

| Input mode | Seed | macro-F1 | 95% bootstrap CI |
|---|---:|---:|---|
| rgb | 1 | 0.6117 | 0.581-0.640 |
| rgb | 2 | 0.6115 | 0.580-0.642 |
| rgb | 3 | 0.6151 | 0.585-0.643 |
| ela | 1 | 0.4264 | 0.418-0.434 |
| ela | 2 | 0.5558 | 0.523-0.590 |
| ela | 3 | 0.4909 | 0.463-0.520 |
| rgb_ela | 1 | 0.6038 | 0.574-0.632 |
| rgb_ela | 2 | 0.6063 | 0.576-0.635 |
| rgb_ela | 3 | 0.5885 | 0.557-0.616 |

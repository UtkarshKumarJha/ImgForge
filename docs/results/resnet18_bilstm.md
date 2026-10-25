# resnet18_bilstm — CASIA 2.0 sweep

Protocol: 224 px, 60-epoch cap with early stopping (patience 10), batch 16, dual-stream stem, mode-aware augmentation, discriminative LRs. Reported number is the **val-tuned threshold, TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and no-TTA are shown for comparison.

Measured on the held-out test split: 1,181 test images (878 authentic / 303 tampered); split fingerprint `79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`.

### resnet18_bilstm

| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | macro-F1 (no TTA) | Accuracy | ROC-AUC |
|---|---:|---|---|---|---|---|
| rgb | 3 | 0.6662 ± 0.0109 | 0.6623 ± 0.0132 | 0.6612 ± 0.0188 | 0.7180 ± 0.0082 | 0.7520 ± 0.0092 |
| ela | 3 | 0.8242 ± 0.0151 | 0.8289 ± 0.0173 | 0.8200 ± 0.0133 | 0.8589 ± 0.0149 | 0.9109 ± 0.0052 |
| rgb_ela | 3 | 0.7415 ± 0.0098 | 0.7323 ± 0.0140 | 0.7313 ± 0.0038 | 0.7855 ± 0.0078 | 0.8364 ± 0.0120 |

### resnet18_bilstm - per seed (val-tuned threshold, TTA on)

| Input mode | Seed | macro-F1 | 95% bootstrap CI |
|---|---:|---:|---|
| rgb | 1 | 0.6674 | 0.637-0.695 |
| rgb | 2 | 0.6547 | 0.624-0.684 |
| rgb | 3 | 0.6764 | 0.648-0.703 |
| ela | 1 | 0.8332 | 0.808-0.856 |
| ela | 2 | 0.8067 | 0.781-0.831 |
| ela | 3 | 0.8326 | 0.808-0.856 |
| rgb_ela | 1 | 0.7478 | 0.721-0.774 |
| rgb_ela | 2 | 0.7302 | 0.700-0.757 |
| rgb_ela | 3 | 0.7466 | 0.719-0.773 |

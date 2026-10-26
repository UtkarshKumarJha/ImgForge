# Final comparison — 384 px, each architecture on its best input mode

Test split: 1,181 images (878 authentic / 303 tampered). Reported number: val-tuned threshold, TTA on; mean ± sd over seeds. Precision and recall are for the tampered class.

| Model | Input | Seeds | Macro-F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---|---|---|---|---|---|
| EfficientNet-B0 | ela | 3 | 0.8753 ± 0.0084 | 0.8981 ± 0.0066 | 0.7450 ± 0.0086 | 0.9164 ± 0.0199 | 0.9582 ± 0.0023 | 0.8670 ± 0.0070 |
| Swin-Tiny | ela | 3 | 0.8922 ± 0.0056 | 0.9153 ± 0.0037 | 0.8069 ± 0.0205 | 0.8823 ± 0.0391 | 0.9690 ± 0.0012 | 0.8913 ± 0.0078 |
| ResNet-18 + BiLSTM | ela | 3 | 0.8789 ± 0.0065 | 0.9004 ± 0.0050 | 0.7437 ± 0.0116 | 0.9340 ± 0.0293 | 0.9571 ± 0.0041 | 0.8540 ± 0.0123 |
| MLP | rgb | 3 | 0.5910 ± 0.0179 | 0.6912 ± 0.0395 | 0.4106 ± 0.0601 | 0.3971 ± 0.1273 | 0.6571 ± 0.0135 | 0.4167 ± 0.0213 |

## Per seed

| Model | Seed | Val macro-F1 | Test macro-F1 | 95% bootstrap CI | JPEG-only macro-F1 |
|---|---:|---:|---:|---|---:|
| EfficientNet-B0 | 1 | 0.8736 | 0.8660 | 0.844-0.886 | 0.7512 |
| EfficientNet-B0 | 2 | 0.8579 | 0.8773 | 0.856-0.897 | 0.7713 |
| EfficientNet-B0 | 3 | 0.8705 | 0.8825 | 0.862-0.902 | 0.7509 |
| Swin-Tiny | 1 | 0.8857 | 0.8936 | 0.874-0.913 | 0.8101 |
| Swin-Tiny | 2 | 0.8874 | 0.8860 | 0.865-0.905 | 0.7950 |
| Swin-Tiny | 3 | 0.8884 | 0.8970 | 0.877-0.916 | 0.7908 |
| ResNet-18 + BiLSTM | 1 | 0.8614 | 0.8782 | 0.857-0.898 | 0.7537 |
| ResNet-18 + BiLSTM | 2 | 0.8564 | 0.8857 | 0.865-0.904 | 0.7698 |
| ResNet-18 + BiLSTM | 3 | 0.8622 | 0.8727 | 0.851-0.893 | 0.7729 |
| MLP | 1 | 0.5758 | 0.5915 | 0.559-0.623 | 0.5657 |
| MLP | 2 | 0.5751 | 0.5729 | 0.543-0.603 | 0.4812 |
| MLP | 3 | 0.5849 | 0.6087 | 0.579-0.637 | 0.5104 |

## 3-seed ensembles (mean probability, mean of the members' val thresholds)

| Model | Threshold | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| EfficientNet-B0 | 0.337 | 0.8802 | 0.9018 |
| Swin-Tiny | 0.520 | 0.8991 | 0.9204 |
| ResNet-18 + BiLSTM | 0.410 | 0.8947 | 0.9136 |
| MLP | 0.530 | 0.6248 | 0.7146 |

## McNemar, Family B (representative = best validation seed), Holm-corrected

| Pair | b (A right, B wrong) | c (A wrong, B right) | p | p (Holm) |
|---|---:|---:|---:|---:|
| efficientnet_b0_ela_p384_s1 vs swin_tiny_ela_p384_s3 | 19 | 50 | 0.000244 | 0.000733 |
| efficientnet_b0_ela_p384_s1 vs resnet18_bilstm_ela_p384_s3 | 35 | 42 | 0.494 | 0.494 |
| efficientnet_b0_ela_p384_s1 vs mlp_rgb_p384_s3 | 317 | 58 | 2.53e-44 | 1.01e-43 |
| swin_tiny_ela_p384_s3 vs resnet18_bilstm_ela_p384_s3 | 46 | 22 | 0.0049 | 0.00981 |
| swin_tiny_ela_p384_s3 vs mlp_rgb_p384_s3 | 329 | 39 | 2.68e-58 | 1.61e-57 |
| resnet18_bilstm_ela_p384_s3 vs mlp_rgb_p384_s3 | 317 | 51 | 4.86e-48 | 2.43e-47 |

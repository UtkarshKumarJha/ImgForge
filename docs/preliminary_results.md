# Preliminary results - all four architectures

Every architecture family trained under one evaluation protocol on CASIA 2.0: 3 seeds x 3
input modes, mean +/- standard deviation over the three seeds, scored on the held-out test
split (1,182 images: 927 authentic, 255 tampered).

## Models and owners

| Model | Owner | Status | Best input mode | Test F1 (macro) | Test accuracy |
|---|---|---|---|---:|---:|
| EfficientNet-B0 | Mayur R Das | Trained (3 seeds x 3 modes) | ELA | 0.804 ± 0.014 | 0.857 ± 0.009 |
| MLP | Utkarsh Kumar Jha | Trained (3 seeds x 3 modes) | RGB | 0.539 ± 0.004 | 0.643 ± 0.024 |
| ResNet-18 + BiLSTM | Dyaneshwar M. Hawal | Trained (3 seeds x 3 modes) | ELA | 0.774 ± 0.006 | 0.838 ± 0.006 |
| Swin-Tiny | Kabir Panda | Trained (3 seeds x 3 modes) | RGB+ELA | 0.547 ± 0.015 | 0.633 ± 0.002 |

## Architecture x input mode (test macro-F1)

| Architecture | RGB (F1) | ELA (F1) | RGB+ELA (F1) |
|---|---:|---:|---:|
| EfficientNet-B0 | RGB 0.652 ± 0.019 | ELA 0.804 ± 0.014 | RGB+ELA 0.665 ± 0.013 |
| MLP | RGB 0.539 ± 0.004 | ELA 0.524 ± 0.000 | RGB+ELA 0.534 ± 0.015 |
| ResNet-18 + BiLSTM | RGB 0.652 ± 0.015 | ELA 0.774 ± 0.006 | RGB+ELA 0.714 ± 0.013 |
| Swin-Tiny | RGB 0.544 ± 0.026 | ELA 0.440 ± 0.000 | RGB+ELA 0.547 ± 0.015 |

## Notes

- **EfficientNet-B0** - Best overall; the ELA mode significantly outperforms RGB and RGB+ELA.
- **MLP** - Near-chance F1 as expected; confirms that spatial bias is necessary.
- **ResNet-18 + BiLSTM** - Second-best; BiLSTM sequential modelling adds marginal benefit over ResNet alone.
- **Swin-Tiny** - Underperforming; likely needs more data or longer training for fine-grained forensic features.

The JPEG quality-factor confound analysis and the McNemar pairwise comparison with Holm
correction are still outstanding; see ROADMAP.md.

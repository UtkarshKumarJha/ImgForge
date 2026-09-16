# EfficientNet-B0 - preliminary results on CASIA 2.0

Owner: Mayur R Das
Status: trained, 3 seeds x 3 input modes, preliminary (unoptimised) hyperparameters

| Input mode | Test macro-F1 |
|---|---:|
| RGB 0.652 ± 0.019
| ELA 0.804 ± 0.014
| RGB+ELA 0.665 ± 0.013 |

Best input mode: ELA (macro-F1 0.804 ± 0.014, accuracy 0.857 ± 0.009)

Best overall; the ELA mode significantly outperforms RGB and RGB+ELA.

Measured on the held-out test split (1,182 images: 927 authentic, 255 tampered).

# Swin-Tiny - preliminary results on CASIA 2.0

Owner: Kabir Panda
Status: trained, 3 seeds x 3 input modes, preliminary (unoptimised) hyperparameters

| Input mode | Test macro-F1 |
|---|---:|
| RGB 0.544 ± 0.026
| ELA 0.440 ± 0.000
| RGB+ELA 0.547 ± 0.015 |

Best input mode: RGB+ELA (macro-F1 0.547 ± 0.015, accuracy 0.633 ± 0.002)

Underperforming; likely needs more data or longer training for fine-grained forensic features.

Measured on the held-out test split (1,182 images: 927 authentic, 255 tampered).

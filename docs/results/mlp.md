# MLP - preliminary results on CASIA 2.0

Owner: Utkarsh Kumar Jha
Status: trained, 3 seeds x 3 input modes, preliminary (unoptimised) hyperparameters

| Input mode | Test macro-F1 |
|---|---:|
| RGB 0.539 ± 0.004
| ELA 0.524 ± 0.000
| RGB+ELA 0.534 ± 0.015 |

Best input mode: RGB (macro-F1 0.539 ± 0.004, accuracy 0.643 ± 0.024)

Near-chance F1 as expected; confirms that spatial bias is necessary.

Measured on the held-out test split (1,182 images: 927 authentic, 255 tampered).

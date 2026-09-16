# ResNet-18 + BiLSTM - preliminary results on CASIA 2.0

Owner: Dyaneshwar M. Hawal
Status: trained, 3 seeds x 3 input modes, preliminary (unoptimised) hyperparameters

| Input mode | Test macro-F1 |
|---|---:|
| RGB 0.652 ± 0.015
| ELA 0.774 ± 0.006
| RGB+ELA 0.714 ± 0.013 |

Best input mode: ELA (macro-F1 0.774 ± 0.006, accuracy 0.838 ± 0.006)

Second-best; BiLSTM sequential modelling adds marginal benefit over ResNet alone.

Measured on the held-out test split (1,182 images: 927 authentic, 255 tampered).

# Literature Review

Source papers behind the architecture choices and the evaluation methodology of
ImgForge (ICT-4442, Comparative Evaluation of Deep Learning Architectures for
Image Forgery Detection and Tamper Attribution).

| Paper (Author, Year) | Method | Dataset | Key Result | Relevance to Project |
|---|---|---|---|---|
| Dong et al., 2013 [1] | CASIA 2.0 benchmark dataset | CASIA 2.0 (12,614 images) | Standard evaluation benchmark for splicing and copy-move detection | Primary dataset used in this project; enables comparison with published baselines |
| Rao & Ni, 2016 [2] | CNN with steganalysis-derived high-pass filters on patches | CASIA 1.0, Columbia | ~97% accuracy on CASIA 1.0 patch classification | Established that low-level residuals, not semantic content, carry the discriminative signal for forgery detection |
| Bappy et al., 2017 [3] | LSTM over CNN patch features for boundary-aware detection | CASIA 2.0, NIST16 | F1 ~0.64 on CASIA 2.0 pixel-level localization | Direct antecedent of our ResNet-18 + BiLSTM architecture; validates sequential modelling of spatial feature maps |
| Bappy et al., 2019 [4] | Hybrid LSTM + encoder-decoder for forgery detection and localization | CASIA 2.0, COVER, Columbia | Improved pixel-level F1 over prior LSTM approaches | Extended LSTM-based approach with encoder-decoder; motivates our BiLSTM design for image-level classification |
| Zhou et al., 2018 [5] | Two-stream Faster R-CNN (RGB + SRM noise) | CASIA 1.0/2.0, NIST16, Columbia | mAP 0.467 on CASIA, outperforming single-stream | Showed that fusing RGB with noise-domain features improves detection; analogous to our RGB+ELA fusion strategy |
| Tan & Le, 2019 [6] | EfficientNet compound scaling for CNNs | ImageNet | 84.1% top-1 accuracy with 8× fewer parameters than prior CNNs | Architecture basis for our CNN model (EfficientNet-B0); provides strong pretrained backbone for transfer learning |
| Wu et al., 2019 [7] | ManTra-Net: anomaly-based manipulation tracing | 385 manipulation types | Generalized detection without per-type supervision | Represents the state-of-the-art in manipulation-agnostic detection; contextualizes our architecture comparison |
| Liu et al., 2021 [8] | Swin Transformer with shifted-window self-attention | ImageNet, COCO, ADE20K | 87.3% top-1 on ImageNet; state-of-the-art on downstream tasks | Architecture basis for our Swin-Tiny model; tests whether global attention helps with local forensic artifacts |
| Kwon et al., 2022 [9] | CAT-Net: JPEG artifact learning in RGB + DCT domains | CASIA 2.0, Columbia, COVER, NIST16 | State-of-the-art localization on CASIA (F1 ~0.82 pixel-level) | Exploits compression artifacts explicitly; validates ELA-based features as informative for forgery detection |
| Guillaro et al., 2023 [10] | TruFor: RGB + learned noise fingerprint + confidence map | Multiple benchmarks including CASIA | State-of-the-art across multiple benchmarks with reliability estimation | Latest generation of forensic detectors; sets the upper bound for comparison with our simpler architectures |

## References

1. Dong et al., 2013. CASIA image tampering detection evaluation database.
2. Rao & Ni, 2016. A deep learning approach to detection of splicing and copy-move forgeries in images.
3. Bappy et al., 2017. Exploiting spatial structure for localizing manipulated image regions.
4. Bappy et al., 2019. Hybrid LSTM + encoder-decoder architecture for forgery detection and localization.
5. Zhou et al., 2018. Learning rich features for image manipulation detection.
6. Tan & Le, 2019. EfficientNet: Rethinking model scaling for convolutional networks.
7. Wu et al., 2019. ManTra-Net: anomaly-based manipulation tracing.
8. Liu et al., 2021. Swin Transformer: Hierarchical vision transformer using shifted windows.
9. Kwon et al., 2022. CAT-Net: JPEG artifact learning in RGB and DCT domains.
10. Guillaro et al., 2023. TruFor: RGB + learned noise fingerprint + confidence map.

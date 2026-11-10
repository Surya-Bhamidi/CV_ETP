**Detail-preserving SR refinement**

The same 479,109-parameter SR architecture was fine-tuned from the original full-trained checkpoint. Inference still maps an LR image to an SR image; the feature networks are training/evaluation tools, not part of the deployed SR model.

The selected checkpoint adds 4 fine-tune epochs to 30 base epochs. The run completed 5 fine-tune epochs on 67,000 training images. Selection used 1,000 validation images; no test examples were used for selecting the checkpoint or loss weights.

Loss weights: {"pixel": 1.0, "identity": 0.05, "perceptual": 0.05, "edge": 0.2}. Pixel L1 is combined with frozen VGG16 feature L1, Sobel edge L1, and cosine feature distance from the frozen ResNet50 ArcFace network. MobileFaceNet is not used in the training loss.

**Held-out FFHQ results: 2,000 images**

| Method | PSNR-Y ↑ | SSIM-Y ↑ | VGG feature error ↓ | Edge error ↓ | MobileFace feature similarity ↑ |
|---|---:|---:|---:|---:|---:|
| bicubic | 24.6016 | 0.6641 | 1.14740 | 0.02237 | 0.32295 |
| original_sr | 25.6829 | 0.7345 | 0.98645 | 0.01938 | 0.37204 |
| detail_sr | 25.4873 | 0.7190 | 0.95981 | 0.01975 | 0.39916 |

**LFW: 6,000 pairs**

These are downstream recognizer verification scores, not classification accuracies belonging to the SR network.

| Frozen recognizer | Processing | Accuracy (%) | AUC | TAR@FAR 1% (%) |
|---|---|---:|---:|---:|
| arcface_training_prior | lr_bicubic | 94.067 | 0.98246 | 86.667 |
| arcface_training_prior | original_sr | 89.167 | 0.95496 | 70.167 |
| arcface_training_prior | detail_sr | 93.267 | 0.97799 | 84.533 |
| arcface_training_prior | hr_ceiling | 98.950 | 0.99752 | 98.700 |
| mobileface_not_in_loss | lr_bicubic | 86.700 | 0.94143 | 61.833 |
| mobileface_not_in_loss | original_sr | 85.100 | 0.92749 | 52.467 |
| mobileface_not_in_loss | detail_sr | 87.750 | 0.94711 | 59.867 |
| mobileface_not_in_loss | hr_ceiling | 96.850 | 0.99194 | 94.367 |

**QMUL-SurvFace: 10,640 pairs**

These are downstream recognizer verification scores, not classification accuracies belonging to the SR network.

| Frozen recognizer | Processing | Accuracy (%) | AUC | TAR@FAR 1% (%) |
|---|---|---:|---:|---:|
| arcface_training_prior | lr_bicubic | 69.427 | 0.75649 | 14.267 |
| arcface_training_prior | original_sr | 68.064 | 0.74582 | 12.368 |
| arcface_training_prior | detail_sr | 68.393 | 0.75280 | 14.812 |
| mobileface_not_in_loss | lr_bicubic | 69.756 | 0.76957 | 18.872 |
| mobileface_not_in_loss | original_sr | 69.192 | 0.75774 | 17.293 |
| mobileface_not_in_loss | detail_sr | 68.553 | 0.74942 | 16.241 |

**Release checks**

- validation_passed: PASS
- heldout_perceptual_error_improves: PASS
- heldout_independent_feature_similarity_improves: PASS
- lfw_second_recognizer_no_material_regression: PASS
- qmul_second_recognizer_no_material_regression: FAIL

The UI keeps the original pixel model available. Default promotion requires all release checks to pass. A failed candidate may be exposed only as an explicitly labelled experimental comparison. The reference/original image is never supplied to the SR network at inference.

Some details were removed by blur, noise, compression, and downsampling to 28 × 28. Refinement estimates missing information; these measurements do not certify a person's identity or guarantee that every texture in the output is accurate. Human recognizability is not established merely by a feature-similarity score.

The independent recognizer has a different architecture, but both recognizers were pretrained using WebFace600K. Benchmark pretraining identity overlap and identity-disjoint FFHQ splits are not established. No new QMUL identification Rank-1/Rank-5 results are claimed for this refinement.

**Sources informing the training objectives**

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Super-Identity Convolutional Neural Network for Face Hallucination](https://openaccess.thecvf.com/content_ECCV_2018/html/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.html)
- [Official InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)

This implementation is an extension of the project's SR training objective, not a reproduction of those complete methods.

**Experimental UI comparison**

The release gate failed because MobileFaceNet QMUL AUC regressed. The original model remains the default. The new checkpoint is exposed only as an explicitly labelled experimental comparison, with the failed check visible in the UI. It is not presented as a validated CCTV improvement.

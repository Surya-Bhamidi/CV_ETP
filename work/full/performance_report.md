**Full training and performance report**

Both the proposed SR network and convolution baseline completed the notebook's full 30-epoch schedule from fresh random initialization. Pilot checkpoints were not reused. The full run uses the corrected rgb_jpeg_seeded_v2 protocol described in ../audit/pretraining_audit.md.

Measured outcome: proposed SR test PSNR is 25.683 dB versus 24.602 for bicubic. LFW verification accuracy is 89.18% versus 94.07%; QMUL verification accuracy is 68.04% versus 69.46%. Full training is complete, but the model has not demonstrated its intended recognition benefit. The pixel-only objective is a limitation to investigate; the identity-loss variant remains untrained.

Dataset: FFHQ128 from nuwandaa/ffhq128 on Hugging Face; 67,000 training images, 1,000 validation images, and 2,000 held-out test images. Input 28 × 28 RGB; output 112 × 112 RGB. Batch size 64; 1,046 batches per epoch; 31,380 optimizer updates per model. AdamW, starting learning rate 0.0002, cosine schedule, L1 reconstruction loss, and gradient clipping. Best checkpoint selected only using validation PSNR-Y.

**Held-out FFHQ reconstruction**

PSNR/SSIM use luminance and exclude a four-pixel border. Higher is better.

| Method | Parameters | PSNR-Y (dB) | SSIM-Y | Best validation PSNR-Y |
|---|---:|---:|---:|---:|
| bicubic | 0 | 24.60163 | 0.664125 | — |
| baseline_conv | 371,715 | 25.57442 | 0.727664 | 25.4589 |
| ours_ssc_attn | 479,109 | 25.68286 | 0.734488 | 25.5665 |

Proposed minus bicubic: PSNR +1.08122 dB; SSIM +0.070363.

Proposed minus baseline_conv: PSNR +0.10844 dB; SSIM +0.006824.

**Training completion and convergence**

- baseline_conv: 30 completed epochs; best validation epoch 30; final L1 0.046650; validation PSNR change over the last five logged epochs +0.00368 dB.
- ours_ssc_attn: 30 completed epochs; best validation epoch 30; final L1 0.045704; validation PSNR change over the last five logged epochs +0.00551 dB.

Completion of the planned schedule is established by checkpoint histories. It does not by itself establish a globally optimal model or guarantee recognition gains.

**LFW: all 6,000 verification pairs with fixed synthetic degradation**

| Condition | Verification accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |
|---|---:|---:|---|---:|
| hr_ceiling | 98.950 | 0.997515 | [0.996141, 0.998728] | 98.700 |
| lr_bicubic | 94.067 | 0.982462 | [0.979583, 0.985220] | 86.667 |
| baseline_conv | 89.217 | 0.957002 | [0.952286, 0.961343] | 70.400 |
| ours_ssc_attn | 89.183 | 0.954966 | [0.950868, 0.959213] | 70.167 |

Proposed minus lr_bicubic: verification accuracy -4.883 percentage points; AUC -0.027496.

Proposed minus baseline_conv: verification accuracy -0.033 percentage points; AUC -0.002037.

**QMUL-SurvFace: real surveillance verification and closed-set identification**

| Condition | Verification accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |
|---|---:|---:|---|---:|
| native_bicubic | 68.637 | 0.753754 | [0.744506, 0.762762] | 13.195 |
| lr_bicubic | 69.464 | 0.756486 | [0.746966, 0.765421] | 14.267 |
| baseline_conv | 68.477 | 0.747197 | [0.738351, 0.757180] | 12.387 |
| ours_ssc_attn | 68.036 | 0.745820 | [0.737123, 0.755709] | 12.368 |

Proposed minus lr_bicubic: verification accuracy -1.429 percentage points; AUC -0.010666.

Proposed minus baseline_conv: verification accuracy -0.442 percentage points; AUC -0.001376.

| Condition | Closed-set Rank-1 (%) | Closed-set Rank-5 (%) | Probes | Gallery identities |
|---|---:|---:|---:|---:|
| native_bicubic | 10.906 | 19.456 | 60,423 | 3,000 |
| lr_bicubic | 10.983 | 19.597 | 60,423 | 3,000 |
| baseline_conv | 10.700 | 19.271 | 60,423 | 3,000 |
| ours_ssc_attn | 10.579 | 18.870 | 60,423 | 3,000 |

**Interpretation and limits**

QMUL coverage: 125,031/125,031 required unique images present; 0 missing. The provided protocol contains 10,640 verification pairs. No missing image pairs were silently discarded.

Recognition uses the same frozen pretrained ArcFace model for every condition. Accuracy thresholds are calibrated on other folds. AUC confidence intervals use 200 pair bootstrap resamples. Closed-set gallery matching does not implement the official challenge's full open-set protocol; these Rank-1/Rank-5 figures must be labelled closed-set.

The native_bicubic QMUL condition resizes the original crop directly to 112 × 112. The other conditions share the same 28 × 28 input. Their preprocessing therefore differs from native_bicubic.

The objective is pixel L1. No identity-loss variant or additional architecture ablations were trained in this run. FFHQ splits are image-disjoint, with no exact duplicate images across them; identity-disjointness and near-duplicate separation have not been established. Frozen recognizer pretraining overlap with benchmark identities is not established.

The implemented architecture differs from the supplied slides: GroupNorm, hard top-k attention-key masking, 28→112 crops, and 479,109 parameters. The historical 29.58 dB slide result is not directly comparable. Earlier pilot scores also predate the RGB JPEG and tied-threshold fixes and should not be used as a controlled comparison against this run.

**Artifacts**

- ckpt/ours_ssc_attn_best.pt: proposed model selected on validation PSNR.
- ckpt/baseline_conv_best.pt: corresponding trained comparison model.
- *_last.pt: optimizer, scheduler, RNG state, and epoch history for resumption.
- manifest.json and split_ids.npz: exact configuration and image indices.
- part1_results.json, part2a_lfw.json, part2a_qmul.json: measured scores.
- comparison.png: nearest LR, bicubic, convolution baseline, proposed SR, clean HR, top to bottom.
- training_curves.png: training and validation histories.
- ../audit/: audit report, original notebook backups, and regression checks.
- ../full_training.log, ../full_lfw.log, ../full_qmul.log: execution logs.

Resume or regenerate the complete experiment with `.venv\Scripts\python.exe -u full_experiment.py`. Configuration/protocol checks prevent reusing incompatible checkpoints.

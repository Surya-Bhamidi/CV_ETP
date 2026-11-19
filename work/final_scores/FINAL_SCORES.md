# Current model: complete measured scorecard

Model: identity-pair v4, 479,109 parameters, x4 output. Raw FP32 parameter storage is 1,916,436 bytes (1.916 MB), excluding buffers, runtime activations and the heavy comparison models. No new training or evaluation was performed to assemble this scorecard.

**Honest assessment:** a compact, fast research prototype with useful distortion-score results. It is not demonstrated to restore 90% of a person, achieve universally better visual quality than heavy models, or reliably recover identity from severely degraded crops. Eye/spectacle fidelity remains a known weakness. There is room to improve.

## What each score means

| Metric | Direction | Meaning and limit |
| --- | --- | --- |
| PSNR-Y (dB) | Higher | Pixel-error measure on luminance versus a clean reference; not visual realism or identity accuracy. A 3.01 dB increase approximately halves mean squared luminance error under the same protocol. |
| SSIM-Y | Higher | Local luminance, contrast and structure similarity to the reference; values near 1 are closer. 0.85 does not mean 85% identity/detail restored. |
| LPIPS-Alex / LPIPS-Squeeze | Lower | Learned perceptual-feature distance to a reference. The backbone matters; Alex and Squeeze scores must not be compared numerically as one metric. No universal pass/fail threshold. |
| MobileFace cosine | Higher | Normalized face-feature similarity to the reference, generally between -1 and 1; a diagnostic proxy, not a probability, human-recognition score or restoration percentage. |
| Verification accuracy | Higher | Percent of labeled same/different-person pairs classified correctly at a specified threshold. Original controlled pair sets are balanced, giving a 50% constant-class baseline; rejecting unsupported pairs can change class balance. Not open-set person-identification accuracy. |
| ROC AUC | Higher | How well similarity scores rank same-person above different-person pairs over thresholds. 0.5 is chance ranking, 1 is perfect ranking. Not a percent restored. |
| Pair coverage | Higher | Fraction of pairs supported by the app input-size policy. Unsupported pairs must remain visible. |
| All-pair success | Higher | Correct supported decisions divided by every original pair; unsupported pairs count as unresolved/unsuccessful. Not ordinary forced-decision accuracy. |
| Latency (ms) | Lower | Time for the stated processing path on the stated hardware. Median is typical; p95 means 95% of recorded requests were no slower. Crop inference is not camera FPS. |
| Parameters | Smaller for compactness | Learned tensor count; neither an accuracy score nor the full app memory footprint. |

## Image quality: 512 FFHQ test images per condition

Original v4 evaluation on 112x112 clean targets reconstructed from 28x28 inputs. Four-pixel-border Y-channel PSNR/SSIM. These are synthetic input degradations, not QMUL real-camera targets.

| Condition | PSNR-Y | SSIM-Y | LPIPS-Alex | LPIPS-Squeeze |
| --- | --- | --- | --- | --- |
| standard28 | 25.4573 | 0.73581 | 0.25682 | 0.19676 |
| clean28 | 28.1194 | 0.85336 | 0.17172 | 0.11314 |

## Latest optional clean-only correction: 512 FFHQ test images

One neural pass plus three residual consistency corrections. Clean-bicubic scope was selected on validation before this test. Single pass remains the default; this mode is disabled for native uploads, QMUL, noisy/mild inputs and external models.

| Mode | PSNR-Y | SSIM-Y | LPIPS-Alex | MobileFace cosine |
| --- | --- | --- | --- | --- |
| single | 28.11941 | 0.85336 | 0.17141 | 0.71519 |
| backprojection3 | 28.31575 | 0.85567 | 0.16540 | 0.73475 |

PSNR change: +0.1963dB; SSIM change: +0.00231; relative LPIPS reduction: 3.51%; cosine change: +0.01956.

The original quality report computes LPIPS on float predictions; the latest pass study uses actual 8-bit export pixels for LPIPS and face diagnostics. Hence the older clean baseline LPIPS 0.17172 and the latest 0.17141 are different measurement pipelines, not a new model improvement. These test images have been examined historically; no claim of a new untouched test cohort is made.

## Current three-model comparison: 128 FFHQ validation images, standard28

The exact same degraded pixels go into every model. External models resize internally to 512 and return 112. Published training sources share FFHQ but exact splits, training budgets and camera adaptation differ. External FFHQ pretraining overlap is not excluded. Descriptive diagnostics, not an independent matched-training benchmark.

| Method | Parameters | PSNR-Y | SSIM-Y | LPIPS-Alex | Face cosine | Median model ms |
| --- | --- | --- | --- | --- | --- | --- |
| identity_v4 | 479,109 | 25.56080 | 0.73305 | 0.25442 | 0.36648 | 10.20 |
| restoration | 86,444,252 | 23.64957 | 0.68390 | 0.12230 | 0.38691 | 60.60 |
| codeformer | 94,112,707 | 23.65464 | 0.63597 | 0.15577 | 0.35721 | 134.15 |
| bicubic | 0 | 24.78072 | 0.66542 | 0.57447 | 0.33077 | Not timed |

v4 better than both heavy baselines on: psnr: 128/128, ssim: 125/128, lpips_alex: 0/128, mobileface_cosine: 41/128.

The heavy models have better mean perceptual distance. GFPGAN also has higher mean face-feature cosine in this comparison. v4 has higher PSNR/SSIM and lower measured crop-model latency. No human likeness/preference study has been completed.

## Small fixed UI test gallery: 12 faces, all three methods

These small diagnostic gallery means are not the 512-image headline. Clean and standard inputs are different experiments.

| Condition | Method | PSNR-Y | SSIM-Y | Median model ms |
| --- | --- | --- | --- | --- |
| test/standard28 | identity_v4 | 24.58933 | 0.70987 | 10.20 |
| test/standard28 | restoration | 23.31892 | 0.68583 | 60.80 |
| test/standard28 | codeformer | 22.79989 | 0.61841 | 135.65 |
| test/standard28 | bicubic_reference | 23.86475 | 0.64147 | Not timed |
| test/clean28 | identity_v4 | 27.37026 | 0.83668 | 10.15 |
| test/clean28 | restoration | 25.65209 | 0.80425 | 60.85 |
| test/clean28 | codeformer | 25.78342 | 0.80417 | 134.95 |
| test/clean28 | bicubic_reference | 25.95604 | 0.77746 | Not timed |

## Controlled QMUL verification: forced28x28 inputs

Main: 9,913 images, 10,640 labeled pairs. Harder additional cohort: 1,423 images from 456 identities, 3,416 pairs. Original 512-crop official-training-validation calibration is primary. The SR image is unchanged when recognition features are mirrored or recognizer scores are fused.

| Cohort | Recognizer pipeline | Accuracy | AUC | Correct / total |
| --- | --- | --- | --- | --- |
| prior_9913_images | Single-view arcface | 68.3459% | 0.756030 | 7272/10640 |
| prior_9913_images | Single-view mobileface | 68.1015% | 0.770208 | 7246/10640 |
| prior_9913_images | Single-view fusion | 69.9060% | 0.776945 | 7438/10640 |
| prior_9913_images | Mirrored-feature arcface | 67.7726% | 0.757180 | 7211/10640 |
| prior_9913_images | Mirrored-feature mobileface | 70.0282% | 0.773178 | 7451/10640 |
| prior_9913_images | Mirrored-feature fusion | 68.8252% | 0.777595 | 7323/10640 |
| additional_identity_disjoint | Single-view arcface | 61.8267% | 0.666720 | 2112/3416 |
| additional_identity_disjoint | Single-view mobileface | 60.6265% | 0.691297 | 2071/3416 |
| additional_identity_disjoint | Single-view fusion | 62.3536% | 0.689329 | 2130/3416 |
| additional_identity_disjoint | Mirrored-feature arcface | 61.8852% | 0.668924 | 2114/3416 |
| additional_identity_disjoint | Mirrored-feature mobileface | 62.4415% | 0.692664 | 2133/3416 |
| additional_identity_disjoint | Mirrored-feature fusion | 60.3337% | 0.690451 | 2061/3416 |

The 70.0282% mirrored-MobileFace result is 7451/10640 correct pairs, only 3 above exactly 70%. Approximate fixed-model anchor-cluster 95% interval: 69.14–70.91%. It excludes some uncertainty from shared negative identities, threshold fitting, seeds, historical test reuse and unknown upstream pretraining. It is not reliable evidence of a general above 70% guarantee.

## Secondary expanded validation calibration

Different threshold-calibration pool, not new SR weights or the same primary benchmark. Included for completeness; do not silently substitute its scores for the primary calibration.

| Cohort | Recognizer | Accuracy | AUC | Correct / total |
| --- | --- | --- | --- | --- |
| prior_9913_images | arcface | 68.9004% | 0.756030 | 7331/10640 |
| prior_9913_images | mobileface | 69.2763% | 0.770208 | 7371/10640 |
| prior_9913_images | fusion | 69.8402% | 0.777391 | 7431/10640 |
| additional_identity_disjoint | arcface | 60.8314% | 0.666720 | 2078/3416 |
| additional_identity_disjoint | mobileface | 63.8173% | 0.691297 | 2180/3416 |
| additional_identity_disjoint | fusion | 64.2857% | 0.690059 | 2196/3416 |

## Native QMUL crop replay: matches the app input policy

Preserves native aspect ratio, rejects crops below 8px, fits large crops within 256px, exports 8-bit x4 predictions and resizes to 112 for recognition. Thresholds are calibrated separately on supported native validation pairs. This is a different pipeline from forced28, not an extra improvement to the same benchmark.

| Cohort | Recognition | Supported accuracy | All-pair success | Supported AUC | Pair coverage | Correct / supported |
| --- | --- | --- | --- | --- | --- | --- |
| main | single | 70.6707% | 70.1128% | 0.778380 | 99.2105% | 7460/10556 supported |
| main | mirror | 70.4718% | 69.9154% | 0.780666 | 99.2105% | 7439/10556 supported |
| additional | single | 63.1579% | 59.7190% | 0.696146 | 94.5550% | 2040/3230 supported |
| additional | mirror | 62.0433% | 58.6651% | 0.698412 | 94.5550% | 2004/3230 supported |

Main: 41 unsupported images and 84 affected pairs. Harder cohort: 46 unsupported images and 186 affected pairs. Report unsupported inputs, not just favorable supported accuracy.

Bicubic is a meaningful reference: its native mirrored-feature supported accuracy was 70.4339% on main and 64.3653% on the harder cohort. Native v4 mirrored-feature accuracy was 70.4718% and 62.0433%. Thus v4 does not universally improve recognition over simple enlargement.

No clean HR QMUL references exist for these crops, so QMUL PSNR/SSIM/LPIPS-to-truth and a restoration percentage are unavailable. No human recognition accuracy has been measured.

## Runtime: recorded native crop spot-check

RTX 4060 Laptop GPU, warm batch 1, FP32, TF32 off. These are prior recorded measurements, not a new benchmark of the modified UI. Historical browser-comparison timing included two custom forward passes and PNG/JSON transport, without recognition or final rendering.

| Input crop | Path | Median ms | p95 ms | Samples |
| --- | --- | --- | --- | --- |
| 19x25 | sr_only | 8.674 | 10.628 | 50 |
| 19x25 | sr_plus_single_mobileface | 12.218 | 15.095 | 50 |
| 59x80 | sr_only | 8.924 | 9.601 | 50 |
| 59x80 | sr_plus_single_mobileface | 11.741 | 12.672 | 50 |
| 19x25 | Historical browser request | 25.250 | 27.700 | 30 |
| 59x80 | Historical browser request | 49.300 | 54.000 | 30 |

Full camera capture, detection, tracking, temporal consistency and sustained multi-face/video FPS remain untested.

## All repeated-pass results: 128 validation images per condition

Same v4 weights. Fixed-size feedback downsamples the previous prediction to the original LR dimensions before each new neural pass. Ensemble methods average inverse-transformed predictions from the original input. Backprojection3 uses one neural pass plus 3 residual corrections. No method passed the original all-profile release guards; single remains the default.

| Profile | Method | PSNR-Y | SSIM-Y | LPIPS-Alex | Face cosine |
| --- | --- | --- | --- | --- | --- |
| standard28 | single | 25.56068 | 0.73305 | 0.25443 | 0.36644 |
| standard28 | feedback2 | 25.13927 | 0.73141 | 0.23719 | 0.35669 |
| standard28 | feedback3 | 24.59274 | 0.72180 | 0.23519 | 0.34328 |
| standard28 | feedback5 | 23.48592 | 0.69539 | 0.24047 | 0.31416 |
| standard28 | feedback8 | 22.00847 | 0.65073 | 0.25811 | 0.27531 |
| standard28 | ensemble2 | 25.62072 | 0.73533 | 0.27169 | 0.36893 |
| standard28 | ensemble4 | 25.56030 | 0.72836 | 0.30868 | 0.36668 |
| standard28 | ensemble8 | 25.48492 | 0.72135 | 0.34310 | 0.36892 |
| standard28 | backprojection3 | 25.52461 | 0.71850 | 0.27192 | 0.35230 |
| mild28 | single | 27.69906 | 0.82827 | 0.18938 | 0.60658 |
| mild28 | feedback2 | 26.95353 | 0.81589 | 0.19187 | 0.56222 |
| mild28 | feedback3 | 26.16895 | 0.79978 | 0.19826 | 0.51829 |
| mild28 | feedback5 | 24.69418 | 0.76315 | 0.21435 | 0.44640 |
| mild28 | feedback8 | 22.85186 | 0.70648 | 0.24134 | 0.37102 |
| mild28 | ensemble2 | 27.79873 | 0.83084 | 0.20329 | 0.61103 |
| mild28 | ensemble4 | 27.76400 | 0.82747 | 0.22860 | 0.61386 |
| mild28 | ensemble8 | 27.69127 | 0.82331 | 0.24989 | 0.61287 |
| mild28 | backprojection3 | 27.87716 | 0.82860 | 0.19385 | 0.61822 |
| clean28 | single | 28.22431 | 0.85096 | 0.17033 | 0.70712 |
| clean28 | feedback2 | 27.40138 | 0.83529 | 0.18038 | 0.64171 |
| clean28 | feedback3 | 26.50700 | 0.81547 | 0.19094 | 0.58295 |
| clean28 | feedback5 | 24.84196 | 0.77181 | 0.21222 | 0.49075 |
| clean28 | feedback8 | 22.83399 | 0.70735 | 0.24518 | 0.39779 |
| clean28 | ensemble2 | 28.32999 | 0.85385 | 0.18416 | 0.71242 |
| clean28 | ensemble4 | 28.30359 | 0.85170 | 0.20345 | 0.71518 |
| clean28 | ensemble8 | 28.22827 | 0.84860 | 0.21771 | 0.71074 |
| clean28 | backprojection3 | 28.40992 | 0.85321 | 0.16473 | 0.72605 |

## Literal repeated x4: 16 validation images per condition

Dimensions grow 28→112→448→1792. All variants are resized to the same 112 target for scoring. These 16-image means are separate from the 128-image table, with matching inputs within this table.

| Profile | Method | PSNR-Y | SSIM-Y | LPIPS-Alex | Face cosine |
| --- | --- | --- | --- | --- | --- |
| standard28 | cascade1 | 25.95191 | 0.72873 | 0.25071 | 0.38095 |
| standard28 | cascade2 | 25.92872 | 0.72852 | 0.24921 | 0.38017 |
| standard28 | cascade3 | 25.93133 | 0.72843 | 0.24937 | 0.38048 |
| mild28 | cascade1 | 28.14014 | 0.82355 | 0.18697 | 0.59730 |
| mild28 | cascade2 | 28.11705 | 0.82386 | 0.18521 | 0.59516 |
| mild28 | cascade3 | 28.11801 | 0.82374 | 0.18542 | 0.59519 |
| clean28 | cascade1 | 28.72368 | 0.85085 | 0.16522 | 0.73111 |
| clean28 | cascade2 | 28.69978 | 0.85132 | 0.16459 | 0.72792 |
| clean28 | cascade3 | 28.70002 | 0.85122 | 0.16471 | 0.72853 |

Repeating the identical original input produced a maximum output difference of 0.0 in the recorded deterministic repeat check.

## Pass-count runtime: 28x28 batch 1

| Method | Median GPU ms | Trials |
| --- | --- | --- |
| single | 8.084 | 30 |
| feedback2 | 16.715 | 30 |
| feedback3 | 24.857 | 30 |
| feedback5 | 40.779 | 30 |
| feedback8 | 63.564 | 30 |
| ensemble2 | 16.371 | 30 |
| ensemble4 | 32.694 | 30 |
| ensemble8 | 67.820 | 30 |
| backprojection3 | 8.063 | 30 |
| cascade2 | 21.142 | 10 |
| cascade3 | 287.009 | 10 |

Tiny differences around the one-pass baseline are timing noise; the clean correction is not demonstrated to speed up inference. Literal3 incurs large intermediate tensors, not new observed detail.

## Final assessment and improvement priorities

- Strongest demonstrated contribution: a 479k-parameter network with fast crop inference and higher distortion scores on the declared synthetic degradation than these heavy pretrained baselines.
- Main weakness: perceptual detail and human identity likeness. Heavy models scored better on mean LPIPS; recognition falls to roughly 60–64% on harder crops.
- The clean-only correction is a modest conditional improvement, not a general camera restoration fix.
- Better native observations and aligned actual video frames could supply more evidence. Repeating the same pixels does not.
- Future teacher-distillation or explicitly trained recurrent refinement should be separate experiments with validation selection, an untouched final evaluation cohort and human likeness/eye-frame review. No further improvement is guaranteed.

## Data and evaluation limits

No direct project split overlap was detected under the recorded identity/pixel/strict-near-duplicate checks. Unknown pretrained overlap and unlabelled FFHQ person overlap are not certified absent. External FFHQ pretraining may include project test images. Historical repeated-test research choices remain a limitation. No score supports a claim of 85–90% true detail restored.

## Source records

- [v4](../../work/recognition_v4/evaluation.json) — SHA256 `944ebd2a9165c7046e1b6f2a3a97d39dc1dcab0739c79670f743e15cc02a645a`
- [mirrored_recognition](../../work/recognition_flip_v5/evaluation.json) — SHA256 `e92b5b5a57bf4da4d81caf83f93690a2023a84e318e49370fa470b62c5fb7b63`
- [mirrored_uncertainty](../../work/recognition_flip_v5/uncertainty.json) — SHA256 `66bd1d5e0ead17cbdb88a8570ecaffdadde4d0b943a6ebe7c24891457f6e256d`
- [native](../../work/field_check_v2/native_results.json) — SHA256 `541cc87c0b5d751afc38b6afd727d620d0131db2a316001e135e9fb6a6ddba7e`
- [native_uncertainty](../../work/field_check_v2/uncertainty.json) — SHA256 `2da4301d7a49acfb8935ade81624d1cd6bf38d825a6479f0ea075987d3c953d5`
- [native_runtime](../../work/field_check_v2/runtime_spotcheck.json) — SHA256 `7d358eb96407e9aff6f72ef929c29f9fc783aabb089a597548455c69b3bce1af`
- [native_latency_sweep](../../work/field_check_v2/latency.json) — SHA256 `146710d49e27546f20a8ac231bb820804de462db23aa51a328b10322c9cb5e6e`
- [three_model_quality](../../work/visual_tradeoff/summary.json) — SHA256 `e59b300933ccd26f5eeb6cc044e7c7f62f43842c9eac600e051bbb1d5ad14d5e`
- [three_model_conditions](../../work/three_model_comparison/summary.json) — SHA256 `0b52ddc03195b228b95751e15f4684d211d7b0151bac41d7af50cc0bc38b0bff`
- [passes](../../work/restoration_passes_v1/evaluation.json) — SHA256 `a53e872c8f9bd49b863811550b02c6470d28be26a331392f56dce92def6e98e0`
- [pass_runtime](../../work/restoration_passes_v1/runtime.json) — SHA256 `6a4fcd6b0924389fd9964696a8a01d6f4f9c7191315de29a0d70d37dc2989b7b`
- [clean_correction](../../work/restoration_passes_v1/clean_only_followup/evaluation.json) — SHA256 `63b6024619c5f17aeb7e864012ec4a3f31873159c19e7c0685e68167187238d0`
- [review](../../work/recognition_v4/review.json) — SHA256 `8d72c86b2496150ada1f4a0c1dbeec4245be34611d85f4ec94b5dca140113209`

Exact source values, including thresholds, raw timing samples and uncertainty records, are retained in [SCORES.json](SCORES.json).

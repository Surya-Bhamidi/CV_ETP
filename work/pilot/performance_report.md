**Project execution and preliminary performance — 10 November 2026**

The project was executed on an NVIDIA GeForce RTX 4060 Laptop GPU using CUDA-enabled PyTorch. These are measured results from a short pilot, not the full notebook training schedule or the earlier slide experiment.

**Experiment scope**

FFHQ128: 4,096 training images, 256 validation images, and 256 held-out test images. Both trainable models ran for 5 epochs with batch size 64, L1 loss, AdamW, and the notebook's cosine schedule. Input/output: 28 × 28 → 112 × 112. Best checkpoints were selected on validation PSNR. Test images were not used to select checkpoints.

The pilot uses fewer images and epochs than the full experiment. It contains 320 optimizer steps per trainable model. Results do not establish the models' final attainable performance.

**Held-out reconstruction quality**

PSNR and SSIM are measured on luminance with a four-pixel border excluded. Higher is better.

| Method | Parameters | PSNR-Y (dB) | SSIM-Y |
|---|---:|---:|---:|
| bicubic | 0 | 24.49509 | 0.665554 |
| baseline_conv | 371,715 | 24.51595 | 0.668158 |
| ours_ssc_attn | 479,109 | 24.51271 | 0.666467 |

The proposed model improves PSNR over bicubic by 0.01762 dB and SSIM by 0.000913. Its PSNR difference from the convolution baseline is -0.00324 dB. These pilot reconstruction gains are very small; the proposed architecture has not demonstrated a useful advantage over the baseline in this run.

**Identity verification**

The same frozen ArcFace model evaluates each processing condition. Accuracy uses thresholds selected on the other nine folds; AUC summarizes the similarity-score ranking. TAR@FAR is computed from the test ROC as in the notebook. AUC confidence intervals use 200 bootstrap resamples.

**LFW: all 6,000 official pairs, synthetic degradation**

| Condition | Accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |
|---|---:|---:|---|---:|
| hr_ceiling | 98.950 | 0.99752 | [0.99614, 0.99873] | 98.700 |
| lr_bicubic | 93.383 | 0.97895 | [0.97586, 0.98192] | 85.200 |
| baseline_conv | 93.600 | 0.97853 | [0.97532, 0.98135] | 84.133 |
| ours_ssc_attn | 93.300 | 0.97808 | [0.97483, 0.98110] | 84.000 |

Proposed minus lr_bicubic: accuracy -0.083 percentage points; AUC -0.00086.

Proposed minus baseline_conv: accuracy -0.300 percentage points; AUC -0.00045.

**QMUL-SurvFace: all 10,640 pairs, real surveillance images**

| Condition | Accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |
|---|---:|---:|---|---:|
| native_bicubic | 68.590 | 0.75376 | [0.74451, 0.76276] | 13.195 |
| lr_bicubic | 69.398 | 0.75649 | [0.74697, 0.76543] | 14.267 |
| baseline_conv | 68.853 | 0.75194 | [0.74318, 0.76124] | 12.932 |
| ours_ssc_attn | 68.543 | 0.75147 | [0.74247, 0.76120] | 12.444 |

Proposed minus lr_bicubic: accuracy -0.855 percentage points; AUC -0.00502.

Proposed minus baseline_conv: accuracy -0.310 percentage points; AUC -0.00047.

native_bicubic resizes each original surveillance crop directly to 112 × 112. The other conditions first use the same 28 × 28 input. The native condition and fixed-size LR condition therefore have different preprocessing.

**Execution fixes and limits**

- Installed project dependencies and CUDA-enabled PyTorch/torchvision into the workspace .venv; existing global packages were not replaced.
- Added a Windows runner using the notebooks' model, training, and metric definitions, with DataLoader workers set to zero.
- Built the FFHQ cache directly from the same dataset's source image archive; image ordering is recorded in work/data/source.json.
- Fixed QMUL download fallback when the main Drive link returns an empty folder listing because it is a file link.
- Fixed onnx2torch's Windows temporary-file locking by loading the ONNX model into memory before conversion, in both recognition notebooks.
- Included the QMUL verification_images directory in image lookup. Full required-image coverage was checked before evaluation; no missing images were silently discarded.
- The larger gallery-identification run was stopped to honor the requested short assessment. Rank-1/Rank-5, extra ablations, identity-loss training, and full 30-epoch training were not completed or claimed.
- The slides' 29.58 dB result uses a different configuration and is not reproduced or directly comparable here.

**Artifacts and rerunning**

- part1_results.json: reconstruction scores.
- part2a_lfw.json and part2a_qmul_verification.json: completed recognition scores, when available.
- ckpt/: best and latest weights, optimizer states, and training histories.
- manifest.json and ../runtime_versions.json: configuration and software versions.
- comparison.png: rows are nearest-enlarged LR, bicubic, convolution baseline, proposed model, and clean HR; columns use the same six held-out examples.
- training_curves.png: training loss and validation PSNR.

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe -u run_project.py train --pilot
.\.venv\Scripts\python.exe -u run_project.py recognition --pilot --benchmark lfw
.\.venv\Scripts\python.exe -u run_project.py recognition --pilot --benchmark qmul --verification-only
.\.venv\Scripts\python.exe write_report.py
```

Omit --pilot to use the full training configuration and separate work/full checkpoints. Omit --verification-only on QMUL to include the full gallery-identification evaluation. Full runs require substantially more time.

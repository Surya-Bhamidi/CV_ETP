**Reconstruction-first investigation, 2026-11-10**

The previous recognizer-guided refinement regressed in PSNR, SSIM, pixel error and edge error despite better VGG/recognition features. Original checkpoint, colour handling, deterministic degradation, RGB tensor conversion, data splits, architecture settings and residual interpolation passed the new audit. See audit/audit.json.

Three independent two-epoch pilots used the same 8,192 training images and 256 validation images: structure, perceptual, visual_gan. Every pilot started from the original 30-epoch custom checkpoint. The perceptual mode had the lowest validation VGG error (0.95199), with SSIM 0.73205 versus original 0.73280; it was selected over the softer structure trial and over the adversarial trial. This choice used validation data and visual comparisons, not test data. The screenshot sample is never used for training or checkpoint selection.

Full run: full_reconstruction, five additional epochs on all 67,000 training images, batch 64, 1,000 validation images, fresh AdamW without weight decay, learning rate 3e-5 decaying to 1e-6. Same 479,109-parameter architecture. Objective weights: pixel L1=1, local SSIM loss=0.15, VGG feature L1=0.08, multiscale Laplacian L1=0.1; no recognizer and no discriminator in the selected full run. Perceptual supervision covers the entire batch, compared with the previous refinement's eight-image subbatch.

Checkpoint eligibility is tightened to PSNR >= original-0.20 dB and SSIM >= original-0.003. Among eligible checkpoints maximize SSIM+0.01*PSNR-0.05*VGG. Keep the original if no checkpoint beats it. Test acceptance additionally requires improvement in official LPIPS-Squeeze, which is not used during training or selection. These checks apply to reconstruction, not identity verification. Original remains the default unless explicitly changed after review.

GFPGAN 1.4 is a separately labelled external pretrained restoration baseline (86.4 million parameters). Pinned source, official weights, licenses and hashes are saved under work/restoration_assets and restoration_vendor. It is not the custom project model. Upstream training uses FFHQ, so our FFHQ split is not proven disjoint from its pretraining: external FFHQ scores are diagnostic only. Inference gets only the low-resolution face crop, never the clean reference. A blend control trades off the original custom reconstruction and the generated restoration; generated features may differ from the actual original.

Sources:
- https://github.com/TencentARC/GFPGAN
- https://github.com/TencentARC/GFPGAN/blob/master/options/train_gfpgan_v1.yml
- https://github.com/richzhang/PerceptualSimilarity

Keep original checkpoints/results, the previous refinement and every pilot intact. Do not present an external pretrained model as newly trained project work or score improvements as proof of human recognizability.

Additional external checks: CodeFormer was forward-tested at fidelity=1 and a validation sweep of 0/0.5/1 was run. A YuNet five-landmark alignment trial used low-resolution input only, on eight validation images. Alignment worsened PSNR and SSIM for both external models and introduced visible distortions, so it is rejected and not enabled. Evidence: work/alignment_assets/trial.json and validation_alignment.png.

The CodeFormer 0/0.5/1 sweep remained inferior in validation structure and showed ghosting; ui_review.json disables it in the active app. A compact Real-ESRGAN general model was also tested at denoising weights 0.5/1 on 256 validation images. It regressed strongly (23.32 dB / 0.657 at denoise 0.5 versus original 25.70 dB / 0.733) and showed blocking. Reflection padding 8/16/24 pixels on eight validation images did not help. That baseline is archived and not exposed as an improvement. No additional external baselines are being explored.

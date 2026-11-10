**Reconstruction quality investigation**

The previous refinement was a measured reconstruction regression: lower PSNR and SSIM and higher pixel/edge error. Its selection objective favoured recognizer features. Colour order, seed handling, checkpoint/config loading and residual/bicubic paths passed the new audit.

Three 8,192-image, two-epoch pilots were compared on 256 validation images. The selected perceptual setting then ran 5 full epochs on 67,000 images; checkpoint selection used 1,000 validation images and chose extra epoch 5. No test image or supplied screenshot was used to train or choose weights.

New objective: pixel L1, local structural similarity, multiscale detail and frozen VGG features. No recognizer appears in training or checkpoint selection. The adversarial pilot includes a separate patch discriminator; see manifest for the selected full run.

**Identical degraded inputs: 2000 FFHQ test images**

| Model | PSNR-Y ↑ | SSIM-Y ↑ | LPIPS-Squeeze ↓ | VGG error ↓ |
|---|---:|---:|---:|---:|
| bicubic | 24.6016 | 0.6641 | 0.53845 | 1.14740 |
| original_custom | 25.6829 | 0.7345 | 0.32220 | 0.98646 |
| previous_refinement | 25.4873 | 0.7190 | 0.29429 | 0.95982 |
| reconstruction_custom | 25.6224 | 0.7346 | 0.25089 | 0.93979 |
| gfpgan_pretrained | 23.5629 | 0.6871 | 0.11701 | 1.02793 |
| gfpgan_blend_75 | 24.5222 | 0.7140 | 0.12769 | 0.97764 |

LPIPS uses the official pretrained SqueezeNet v0.1 calibrated metric, excluded from training and checkpoint selection. PSNR/SSIM retain the original luminance and four-pixel border protocol. LPIPS measures the full RGB image. Paired bootstrap intervals are in paired_differences.json; they describe image-sampling uncertainty, not training-seed uncertainty.

**Custom model acceptance checks**

- full_training_completed: PASS
- validation_reconstruction_guards: PASS
- test_psnr_within_0_20_db: PASS
- test_ssim_within_0_003: PASS
- test_independent_lpips_improves: PASS

**External restoration option**

GFPGAN 1.4 and CodeFormer are external pretrained models, not the custom 479,109-parameter project network. Their upstream training uses FFHQ; overlap with these FFHQ images is not excluded. Their FFHQ scores are diagnostic, not evidence of held-out generalization. Generative detail may differ from the actual person. No original/reference image is used at inference.

Restoration takes only the same low-resolution crop, resizes it internally to 512 pixels, synthesizes a face and returns exactly ×4 input dimensions. A user-controlled blend can trade the original custom result against the generative result. CodeFormer was also checked on eight validation images at fidelity 0/0.5/1 and with landmark alignment; visible ghosting and poorer validation structure led to rejection from the active UI. Its experiments remain in work/codeformer_assets and work/alignment_assets; it is not included in this final full-test table. A compact general Real-ESRGAN baseline was tested on 256 validation images with denoising weights 0.5/1, plus a small reflection-padding trial. It produced blocking and poorer PSNR/SSIM (about 23.32 dB / 0.657 at weight 0.5 versus the original custom model 25.70 dB / 0.733 on those same images), so it was also rejected. See work/realesrgan_assets for those results. GFPGAN is a separate selectable option, never reported as a newly trained custom model.

**Limits**

These images are held out from custom-model training, but earlier models have already been evaluated on this test split. Scores here are repeated comparisons on that fixed split, not a fresh benchmark. Image splits are disjoint; identity-disjointness is not established. Human recognizability and CCTV verification are not established by these reconstruction scores. Examples remain difficult where the LR image does not contain enough information.

[GFPGAN official implementation](https://github.com/TencentARC/GFPGAN) · [Official FFHQ training configuration](https://github.com/TencentARC/GFPGAN/blob/master/options/train_gfpgan_v1.yml) · [CodeFormer official implementation](https://github.com/sczhou/CodeFormer) · [LPIPS official implementation](https://github.com/richzhang/PerceptualSimilarity)

**Final review and UI handoff**

The new custom model reduces LPIPS by 22.13% versus the original, with essentially unchanged SSIM and a 0.0605 dB PSNR reduction. The SSIM paired confidence interval crosses zero, so no meaningful SSIM improvement is claimed. Five full fine-tuning epochs took 26.45 minutes including validation. The new custom checkpoint is now the UI/API default; the original remains selectable.

On the supplied sunglasses example, the custom reconstruction remains soft. GFPGAN produces clearer-looking features but changes dark sunglasses into clearer lenses and alters hair appearance; a lower perceptual distance is not proof of exact recovery. It stays a separately labelled external option.

Final interface verification: all 11 general browser checks, six restoration checks and eight final API checks passed on the GPU server. The UI checkpoint matches the evaluated checkpoint; the same supplied low-resolution input produces the recorded test result. Screenshots and checks are in work/ui.

# Using the comparison UI

Follow the [setup instructions](README.md#run-the-comparison-app) and open the local server. With FFHQ prepared, the demo starts on four validation cases. The first two have higher v4 face-feature similarity than both pretrained baselines. The last two show large perceptual advantages for the heavier models. The gallery labels this selection.

1. Pick a face or upload a cropped image. On FFHQ, pick a degradation profile. QMUL examples use their native pixels.
2. Select **Ours v4**, **GFPGAN 1.4** or **CodeFormer**. Switching models sends the same crop to the selected model.
3. Choose **Input**, **Bicubic** or **Original** and move the divider. **Side by side** shows both full images. For a pretrained output, **Our model** shows v4 on the same input.
4. Use **Downsampling only** with v4 to enable the optional three-step consistency correction. **Single-pass v4** compares the correction with the ordinary prediction.
5. Read PSNR and SSIM against the supplied clean reference. Native camera examples have no paired reference. Download the selected output as PNG.

Choose **FFHQ test** to explore the fixed test gallery. The [case studies](work/visual_tradeoff/case_studies.html) and [pass-count comparison](work/restoration_passes_v1/comparison.html) can also be opened offline.

Stop the local server with `python comparison_app.py --stop`.

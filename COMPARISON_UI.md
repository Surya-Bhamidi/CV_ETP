Double-click `Open_Comparison_UI.cmd` to open the local app, or run:

```powershell
.\.venv\Scripts\python.exe -u comparison_app.py --open-browser
```

Open http://127.0.0.1:8765. The first load warms up the full-trained SR model.

- Select a held-out test face to compare its 28 × 28 input, 112 × 112 reconstruction, bicubic enlargement, and clean original.
- Drag the divider, switch to side-by-side, zoom, and pan the image to inspect facial details.
- Upload a PNG, JPEG, or WebP cropped face to reconstruct it at ×4. Inputs larger than 256 pixels on the longest edge are resized, with the processed dimensions shown.
- For a clear photo, choose **Simulate CCTV from a clear photo**. It creates a 112 × 112 clean reference and a degraded 28 × 28 input using the project's degradation method.
- An optional clean original must exactly match the ×4 output dimensions. PSNR/SSIM are shown only with a matching reference; they use the same luminance and border-cropping settings as the full experiment.
- Download the reconstructed PNG using **Download result**.

The **Reconstruction model** selector defaults to **Reconstruction refinement (ours)** after the new checkpoint passes reconstruction checks. **Pixel model (original)** remains selectable as a fallback. The earlier recognizer-guided refinement was retired after confirmed visual/structural regression; its files remain archived.

The new custom refinement retains the same 479,109-parameter architecture and uses pixel, structure, VGG and multiscale-detail losses. No face recognizer is used in its training or checkpoint selection. Results and evaluation limits are in `work/visual_sr/full_reconstruction/quality_report.md`.

**GFPGAN 1.4 (pretrained)** is a separate external restoration option when its official weights are installed. CodeFormer was also tested at three fidelity settings and rejected from active UI choices after visible ghosting and poorer validation structure. These are not newly trained versions of our SR model. Generative restoration can make features clearer while changing the actual face: inspect glasses, eyes, mouth and hair against a clean reference when available. Their FFHQ pretraining may overlap these example images, so external FFHQ scores do not demonstrate independent generalization.

Pretrained restoration requires a square, approximately aligned face crop. Its **Restoration strength** slider blends the original custom result (0%) with the external restoration (100%). It does not measure accuracy or recovered detail. Output remains exactly ×4 input dimensions, even though external models process 512-pixel faces internally. Their source licenses and pinned provenance are included in `restoration_vendor`, `codeformer_vendor` and the corresponding `work` asset folders. CodeFormer is licensed for noncommercial use.

**Previous SR** compares any alternate result with the original custom model on identical input pixels. **Original** always means the clean reference, never an input to the reconstruction network. Zoom starts at **Fit**, relative to the viewer size.

The comparison app needs no recognition network at inference. Estimated detail is not guaranteed to match the original. Browser/API checks cover model switching, strength endpoints, PNG output dimensions, clean-reference isolation, invalid inputs, and desktop/mobile layout. Official LPIPS is used only by the offline quality evaluation; its additional dependency is in `requirements-quality.txt`.

The app loads `work/full/ckpt/ours_ssc_attn_best.pt`; it does not retrain the model. Uploads are handled in memory on this computer. The service listens only on loopback.

To stop a background instance:

```powershell
.\.venv\Scripts\python.exe comparison_app.py --stop
```

To use a different port, add `--port 8766` when starting or stopping.

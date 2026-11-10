**Pretraining audit ? 10 November 2026**

Dataset: FFHQ128, Hugging Face nuwandaa/ffhq128. Source archive: https://huggingface.co/datasets/nuwandaa/ffhq128/resolve/main/thumbnails128x128.zip . The 70,000 RGB images are cached at work/data/ffhq128_u8.npy. All images were hashed; no exact duplicate groups or cross-split exact duplicates were found. A random sample of 256 cached images matched their archive source pixels exactly. Near-duplicate and identity-level separation have not been established.

**Confirmed issues fixed before full training**

1. RGB arrays were passed directly to OpenCV JPEG encoding, which interprets channels as BGR. This changed the JPEG colour transform and compression artifacts. Explicit RGB-to-BGR and BGR-to-RGB conversion now preserves the intended colour semantics. The tested image showed a mean absolute discrepancy of 3.48 pixel levels under the previous implementation.
2. Training augmentation created unseeded NumPy generators, bypassing the configured seed. It now uses seed + image index + epoch. Epoch state is shared with DataLoader workers; epoch shuffling uses an explicit generator. Both models receive the same augmentations and ordering for a given epoch.
3. Checkpoints did not validate configuration or protocol and were not written atomically. They now save configuration/protocol and Torch RNG states, reject incompatible resumption, and replace completed checkpoint files atomically. The runner also verifies the data/source manifest before resume. Pilot checkpoints will not initialize the full run.
4. Recognition caches validated only file count and first/last names, allowing stale embeddings after weight changes. Caches now fingerprint weights, full file ordering, source/preprocessing, and file metadata.
5. Verification threshold selection could split a group of equal scores at an impossible boundary. Calibration now considers only distinct-score boundaries. The previous test case achieved 37.5% where 62.5% was attainable; the corrected method passed 2,000 brute-force tied-score cases.
6. Read-only NumPy image arrays could produce Torch warnings; tensors now receive writable contiguous copies. Windows notebook DataLoader configuration now defaults to zero workers.

**Tests**

Notebook syntax, tensor shapes/ranges, seeded augmentation, correct RGB JPEG reference, zero-offset deformable-convolution equivalence, offset/scorer gradients, explicit-versus-linear attention equivalence, and identity PSNR/SSIM references all passed. Interrupted/resumed CPU training exactly matched uninterrupted training, and incompatible checkpoints were rejected. Cache invalidation was verified by changing model weights.

A diagnostic SR model learned a single fixed training face in 200 updates: bicubic PSNR 24.18 dB versus model PSNR 26.75 dB; pixel L1 fell from 0.0540 to 0.0320. This demonstrates learning capacity on that diagnostic example; it is not a held-out performance result.

**Remaining methodology limits, disclosed before starting**

- The current notebook architecture differs from the slides: 28-to-112 images, GroupNorm, hard top-k attention-key masking, and 479,109 parameters. The full run trains the current notebook model, not an exact reproduction of the slide architecture.
- The main objective remains L1 reconstruction. Recognition gains are not guaranteed; identity-loss training is a separate, unrun variant.
- Logical top-k key masking does not gather or prune positions from the dense tensor operations, so reduced attention computation from sparsity has not been demonstrated.
- FFHQ uses image-level splits, not verified identity-disjoint splits. LFW/ArcFace pretraining-identity overlap is not established.
- QMUL Rank-1/Rank-5 code implements closed-set identification, not the complete open-set challenge. This distinction must remain in reports.
- Historical pilot metrics predate the RGB/threshold fixes; full-run numbers must be compared with newly evaluated full-run baselines rather than presented as directly comparable pilot gains.

**Full training configuration**

Start from fresh random weights. Train the proposed model and the convolution baseline for 30 epochs each on 67,000 images, with 1,000 validation images and 2,000 held-out test images. Batch size 64; 31,380 optimizer updates per model. AdamW, initial learning rate 0.0002, cosine schedule, pixel L1, and gradient clipping are retained. Best weights are selected only on validation PSNR. Full artifacts are separate from pilot artifacts at work/full/.

**Training throughput adjustment**

After epoch 1 was saved, the run resumed with one background CPU producer overlapping image preparation and GPU training. A 128-image, eight-batch regression check confirmed that order and tensor values exactly match synchronous loading. Model, loss, sample ordering, seed policy, optimizer, and epoch count are unchanged. Full-batch GPU forward/backward checks passed with finite gradients.

During training, the independent evaluation audit also confirmed a float32 midpoint-rounding edge case. Verification calibration and score comparisons now use float64, and the adjacent-float32-score regression check passes. This does not alter the training model or checkpoints.

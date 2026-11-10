**Detail-preserving refinement experiment**

The existing 479,109-parameter model is retained. The original full-trained checkpoint and its results are not overwritten. Fine-tuning starts from its weights, with a fresh optimizer and the same 67,000 / 1,000 / 2,000 image split and degradation protocol.

Training adds VGG16 feature L1, Sobel edge L1, and frozen ArcFace feature cosine loss to pixel L1. The selected balanced weights are pixel=1, identity=0.05, perceptual=0.05, edge=0.20. Feature losses are evaluated on eight images per training batch of 64 to fit GPU memory. MobileFaceNet is excluded from the training loss and used as an additional validation/evaluation network.

Trials use training images only. Their loss balance and checkpoints are selected on validation images; the screenshot/test example is not used to train or select a model. Full fine-tuning runs five extra epochs, starting again from the original full checkpoint, not from a trial checkpoint.

Release checks are declared in refine_model.py and evaluate_refinement.py. A candidate is not added to the UI until it passes these checks. Even an accepted model cannot guarantee that missing high-frequency details or a person's identity are recovered exactly.

The UI will retain both versions and compare their outputs on identical inputs. Original ground-truth images are used only for reference/metrics, never as an input to the SR model during inference.

Motivation:
- https://arxiv.org/abs/1603.08155 ? perceptual training for image transformation and SR.
- https://openaccess.thecvf.com/content_ECCV_2018/html/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.html ? identity-preserving face SR; this project does not reproduce its complete method.
- https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md ? feature network provenance.

Outcome: the selected refinement passed FFHQ validation/test feature checks and improved LFW, but failed the independent QMUL release check. It is available only as an experimental UI option. The original remains the default; no universal recognizability improvement is claimed.

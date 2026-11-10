"""Summarize the completed full training and evaluation artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "work" / "full"


def main():
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((OUT / "manifest.json").read_text())
    scores = json.loads((OUT / "part1_results.json").read_text())
    histories = {}
    for tag in ["baseline_conv", "ours_ssc_attn"]:
        checkpoint = torch.load(OUT / "ckpt" / f"{tag}_last.pt", map_location="cpu")
        assert checkpoint["epoch"] == 30, f"{tag} has not completed full training"
        histories[tag] = checkpoint["hist"]
    lines = [
        "**Full training and performance report**", "",
        "Both the proposed SR network and convolution baseline completed the notebook's full 30-epoch schedule from fresh random initialization. Pilot checkpoints were not reused. The full run uses the corrected rgb_jpeg_seeded_v2 protocol described in ../audit/pretraining_audit.md.", "",
        f"Dataset: FFHQ128 from nuwandaa/ffhq128 on Hugging Face; {manifest['train_images']:,} training images, {manifest['validation_images']:,} validation images, and {manifest['test_images']:,} held-out test images. Input 28 × 28 RGB; output 112 × 112 RGB. Batch size 64; 1,046 batches per epoch; 31,380 optimizer updates per model. AdamW, starting learning rate 0.0002, cosine schedule, L1 reconstruction loss, and gradient clipping. Best checkpoint selected only using validation PSNR-Y.", "",
        "**Held-out FFHQ reconstruction**", "",
        "PSNR/SSIM use luminance and exclude a four-pixel border. Higher is better.", "",
        "| Method | Parameters | PSNR-Y (dB) | SSIM-Y | Best validation PSNR-Y |",
        "|---|---:|---:|---:|---:|",
    ]
    lfw = json.loads((OUT / "part2a_lfw.json").read_text())
    qmul = json.loads((OUT / "part2a_qmul.json").read_text())
    summary = (f"Measured outcome: proposed SR test PSNR is {scores['ours_ssc_attn']['PSNR_Y']:.3f} dB versus "
               f"{scores['bicubic']['PSNR_Y']:.3f} for bicubic. LFW verification accuracy is "
               f"{100*lfw['ours_ssc_attn']['ver']['acc']:.2f}% versus {100*lfw['lr_bicubic']['ver']['acc']:.2f}%; "
               f"QMUL verification accuracy is {100*qmul['ours_ssc_attn']['ver']['acc']:.2f}% versus "
               f"{100*qmul['lr_bicubic']['ver']['acc']:.2f}%.")
    if all(result['ours_ssc_attn']['ver']['acc'] < result['lr_bicubic']['ver']['acc'] for result in [lfw, qmul]):
        summary += " Full training is complete, but the model has not demonstrated its intended recognition benefit. The pixel-only objective is a limitation to investigate; the identity-loss variant remains untrained."
    lines[4:4] = [summary, ""]
    for name in ["bicubic", "baseline_conv", "ours_ssc_attn"]:
        r = scores[name]
        best = f"{r['best_validation_psnr_y']:.4f}" if "best_validation_psnr_y" in r else "—"
        lines.append(f"| {name} | {r['params']:,} | {r['PSNR_Y']:.5f} | {r['SSIM_Y']:.6f} | {best} |")
    ours = scores["ours_ssc_attn"]
    for ref in ["bicubic", "baseline_conv"]:
        r = scores[ref]
        lines += ["", f"Proposed minus {ref}: PSNR {ours['PSNR_Y']-r['PSNR_Y']:+.5f} dB; SSIM {ours['SSIM_Y']-r['SSIM_Y']:+.6f}."]
    lines += ["", "**Training completion and convergence**", ""]
    for name, history in histories.items():
        best = max(history, key=lambda r: r["val_psnr"])
        lines += [f"- {name}: {len(history)} completed epochs; best validation epoch {best['epoch']}; final L1 {history[-1]['l1']:.6f}; validation PSNR change over the last five logged epochs {history[-1]['val_psnr']-history[-5]['val_psnr']:+.5f} dB."]
    lines += ["", "Completion of the planned schedule is established by checkpoint histories. It does not by itself establish a globally optimal model or guarantee recognition gains.", ""]
    for filename, title in [("part2a_lfw.json", "LFW: all 6,000 verification pairs with fixed synthetic degradation"), ("part2a_qmul.json", "QMUL-SurvFace: real surveillance verification and closed-set identification")]:
        path = OUT / filename
        assert path.exists(), f"Evaluation missing: {path}"
        results = json.loads(path.read_text())
        lines += [f"**{title}**", "", "| Condition | Verification accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |", "|---|---:|---:|---|---:|"]
        for name, result in results.items():
            v = result["ver"]
            lines.append(f"| {name} | {100*v['acc']:.3f} | {v['auc']:.6f} | [{v['auc_lo']:.6f}, {v['auc_hi']:.6f}] | {100*v['tar_far1']:.3f} |")
        proposed = results["ours_ssc_attn"]["ver"]
        for ref in ["lr_bicubic", "baseline_conv"]:
            baseline = results[ref]["ver"]
            lines += ["", f"Proposed minus {ref}: verification accuracy {100*(proposed['acc']-baseline['acc']):+.3f} percentage points; AUC {proposed['auc']-baseline['auc']:+.6f}."]
        if "ident" in results["ours_ssc_attn"]:
            lines += ["", "| Condition | Closed-set Rank-1 (%) | Closed-set Rank-5 (%) | Probes | Gallery identities |", "|---|---:|---:|---:|---:|"]
            for name, result in results.items():
                i = result["ident"]
                lines.append(f"| {name} | {100*i['rank1']:.3f} | {100*i['rank5']:.3f} | {i['n_probes']:,} | {i['n_ids']:,} |")
        lines += [""]
    coverage = json.loads((OUT / "qmul_coverage.json").read_text())
    lines += [
        "**Interpretation and limits**", "",
        f"QMUL coverage: {coverage['available_images']:,}/{coverage['required_images']:,} required unique images present; {coverage['missing_images']} missing. The provided protocol contains 10,640 verification pairs. No missing image pairs were silently discarded.", "",
        "Recognition uses the same frozen pretrained ArcFace model for every condition. Accuracy thresholds are calibrated on other folds. AUC confidence intervals use 200 pair bootstrap resamples. Closed-set gallery matching does not implement the official challenge's full open-set protocol; these Rank-1/Rank-5 figures must be labelled closed-set.", "",
        "The native_bicubic QMUL condition resizes the original crop directly to 112 × 112. The other conditions share the same 28 × 28 input. Their preprocessing therefore differs from native_bicubic.", "",
        "The objective is pixel L1. No identity-loss variant or additional architecture ablations were trained in this run. FFHQ splits are image-disjoint, with no exact duplicate images across them; identity-disjointness and near-duplicate separation have not been established. Frozen recognizer pretraining overlap with benchmark identities is not established.", "",
        "The implemented architecture differs from the supplied slides: GroupNorm, hard top-k attention-key masking, 28→112 crops, and 479,109 parameters. The historical 29.58 dB slide result is not directly comparable. Earlier pilot scores also predate the RGB JPEG and tied-threshold fixes and should not be used as a controlled comparison against this run.", "",
        "**Artifacts**", "",
        "- ckpt/ours_ssc_attn_best.pt: proposed model selected on validation PSNR.",
        "- ckpt/baseline_conv_best.pt: corresponding trained comparison model.",
        "- *_last.pt: optimizer, scheduler, RNG state, and epoch history for resumption.",
        "- manifest.json and split_ids.npz: exact configuration and image indices.",
        "- part1_results.json, part2a_lfw.json, part2a_qmul.json: measured scores.",
        "- comparison.png: nearest LR, bicubic, convolution baseline, proposed SR, clean HR, top to bottom.",
        "- training_curves.png: training and validation histories.",
        "- ../audit/: audit report, original notebook backups, and regression checks.",
        "- ../full_training.log, ../full_lfw.log, ../full_qmul.log: execution logs.",
        "",
        "Resume or regenerate the complete experiment with `.venv\\Scripts\\python.exe -u full_experiment.py`. Configuration/protocol checks prevent reusing incompatible checkpoints.",
    ]
    (OUT / "performance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, label in [("baseline_conv", "Convolution baseline"), ("ours_ssc_attn", "Proposed SR")]:
        h = histories[name]
        epochs = [r["epoch"] for r in h]
        axes[0].plot(epochs, [r["l1"] for r in h], label=label)
        axes[1].plot(epochs, [r["val_psnr"] for r in h], label=label)
    axes[0].set(title="Training loss", ylabel="Mean L1")
    axes[1].set(title="Validation reconstruction", ylabel="PSNR-Y (dB)")
    for a in axes:
        a.set(xlabel="Epoch", xticks=[1, 5, 10, 15, 20, 25, 30])
        a.grid(alpha=.2)
        a.legend()
    fig.suptitle("Full training: 67,000 train / 1,000 validation images")
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)
    print("FULL REPORT", OUT / "performance_report.md", flush=True)


if __name__ == "__main__":
    main()

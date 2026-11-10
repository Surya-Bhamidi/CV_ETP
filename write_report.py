"""Write a reviewable report and training plot from saved run artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "work" / "pilot"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    sr = json.loads((OUT / "part1_results.json").read_text())
    lines = [
        "**Project execution and preliminary performance — 10 November 2026**",
        "",
        "The project was executed on an NVIDIA GeForce RTX 4060 Laptop GPU using CUDA-enabled PyTorch. These are measured results from a short pilot, not the full notebook training schedule or the earlier slide experiment.",
        "",
        "**Experiment scope**",
        "",
        f"FFHQ128: {manifest['train_images']:,} training images, {manifest['validation_images']} validation images, and {manifest['test_images']} held-out test images. Both trainable models ran for {manifest['config']['epochs']} epochs with batch size 64, L1 loss, AdamW, and the notebook's cosine schedule. Input/output: 28 × 28 → 112 × 112. Best checkpoints were selected on validation PSNR. Test images were not used to select checkpoints.",
        "",
        "The pilot uses fewer images and epochs than the full experiment. It contains 320 optimizer steps per trainable model. Results do not establish the models' final attainable performance.",
        "",
        "**Held-out reconstruction quality**",
        "",
        "PSNR and SSIM are measured on luminance with a four-pixel border excluded. Higher is better.",
        "",
        "| Method | Parameters | PSNR-Y (dB) | SSIM-Y |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sr.items():
        lines.append(f"| {name} | {value['params']:,} | {value['PSNR_Y']:.5f} | {value['SSIM_Y']:.6f} |")
    ours, bic, base = sr["ours_ssc_attn"], sr["bicubic"], sr["baseline_conv"]
    lines += [
        "",
        f"The proposed model improves PSNR over bicubic by {ours['PSNR_Y'] - bic['PSNR_Y']:.5f} dB and SSIM by {ours['SSIM_Y'] - bic['SSIM_Y']:.6f}. Its PSNR difference from the convolution baseline is {ours['PSNR_Y'] - base['PSNR_Y']:+.5f} dB. These pilot reconstruction gains are very small; the proposed architecture has not demonstrated a useful advantage over the baseline in this run.",
        "",
        "**Identity verification**",
        "",
        "The same frozen ArcFace model evaluates each processing condition. Accuracy uses thresholds selected on the other nine folds; AUC summarizes the similarity-score ranking. TAR@FAR is computed from the test ROC as in the notebook. AUC confidence intervals use 200 bootstrap resamples.",
    ]
    for filename, title in [("part2a_lfw.json", "LFW: all 6,000 official pairs, synthetic degradation"), ("part2a_qmul_verification.json", "QMUL-SurvFace: all 10,640 pairs, real surveillance images")]:
        path = OUT / filename
        lines += ["", f"**{title}**", ""]
        if not path.exists():
            lines += ["Evaluation is still in progress; no scores are claimed."]
            continue
        results = json.loads(path.read_text())
        lines += ["| Condition | Accuracy (%) | AUC | AUC 95% CI | TAR@FAR 1% (%) |", "|---|---:|---:|---|---:|"]
        for name, result in results.items():
            v = result["ver"]
            lines.append(f"| {name} | {100*v['acc']:.3f} | {v['auc']:.5f} | [{v['auc_lo']:.5f}, {v['auc_hi']:.5f}] | {100*v['tar_far1']:.3f} |")
        proposed = results["ours_ssc_attn"]["ver"]
        for ref in ["lr_bicubic", "baseline_conv"]:
            v = results[ref]["ver"]
            lines += ["", f"Proposed minus {ref}: accuracy {100*(proposed['acc']-v['acc']):+.3f} percentage points; AUC {proposed['auc']-v['auc']:+.5f}."]
        if "native_bicubic" in results:
            lines += ["", "native_bicubic resizes each original surveillance crop directly to 112 × 112. The other conditions first use the same 28 × 28 input. The native condition and fixed-size LR condition therefore have different preprocessing."]
    lines += [
        "",
        "**Execution fixes and limits**",
        "",
        "- Installed project dependencies and CUDA-enabled PyTorch/torchvision into the workspace .venv; existing global packages were not replaced.",
        "- Added a Windows runner using the notebooks' model, training, and metric definitions, with DataLoader workers set to zero.",
        "- Built the FFHQ cache directly from the same dataset's source image archive; image ordering is recorded in work/data/source.json.",
        "- Fixed QMUL download fallback when the main Drive link returns an empty folder listing because it is a file link.",
        "- Fixed onnx2torch's Windows temporary-file locking by loading the ONNX model into memory before conversion, in both recognition notebooks.",
        "- Included the QMUL verification_images directory in image lookup. Full required-image coverage was checked before evaluation; no missing images were silently discarded.",
        "- The larger gallery-identification run was stopped to honor the requested short assessment. Rank-1/Rank-5, extra ablations, identity-loss training, and full 30-epoch training were not completed or claimed.",
        "- The slides' 29.58 dB result uses a different configuration and is not reproduced or directly comparable here.",
        "",
        "**Artifacts and rerunning**",
        "",
        "- part1_results.json: reconstruction scores.",
        "- part2a_lfw.json and part2a_qmul_verification.json: completed recognition scores, when available.",
        "- ckpt/: best and latest weights, optimizer states, and training histories.",
        "- manifest.json and ../runtime_versions.json: configuration and software versions.",
        "- comparison.png: rows are nearest-enlarged LR, bicubic, convolution baseline, proposed model, and clean HR; columns use the same six held-out examples.",
        "- training_curves.png: training loss and validation PSNR.",
        "",
        "Run from the project directory:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe -u run_project.py train --pilot",
        r".\.venv\Scripts\python.exe -u run_project.py recognition --pilot --benchmark lfw",
        r".\.venv\Scripts\python.exe -u run_project.py recognition --pilot --benchmark qmul --verification-only",
        r".\.venv\Scripts\python.exe write_report.py",
        "```",
        "",
        "Omit --pilot to use the full training configuration and separate work/full checkpoints. Omit --verification-only on QMUL to include the full gallery-identification evaluation. Full runs require substantially more time.",
    ]
    (OUT / "performance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, label in [("baseline_conv", "Convolution baseline"), ("ours_ssc_attn", "Proposed model")]:
        history = torch.load(OUT / "ckpt" / f"{name}_last.pt", map_location="cpu")["hist"]
        epochs = [v["epoch"] for v in history]
        axes[0].plot(epochs, [v["l1"] for v in history], marker="o", label=label)
        axes[1].plot(epochs, [v["val_psnr"] for v in history], marker="o", label=label)
    axes[0].set(ylabel="Mean training L1", title="Training loss")
    axes[1].set(ylabel="Validation PSNR-Y (dB)", title="Validation reconstruction quality")
    for a in axes:
        a.set(xlabel="Epoch", xticks=range(1, 6))
        a.grid(alpha=.2)
        a.legend()
    fig.suptitle("Short pilot: 4,096 training / 256 validation images")
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)
    print(OUT / "performance_report.md")


if __name__ == "__main__":
    main()

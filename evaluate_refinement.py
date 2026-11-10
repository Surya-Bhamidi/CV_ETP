"""Evaluate a validation-selected refinement before making it available in the UI."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from refine_model import BASE, Priors, evaluate, sha, write_json
from run_project import ROOT, notebook_definitions, runtime


def load_model(n, cfg, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = n.SRNet(cfg).to(n.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def recognition(n, priors, old, new, folder, benchmark):
    r = notebook_definitions("sr_part2a_recognition.ipynb", [1, 6, 7, 8, 9, 11], "refinement_recognition")
    r.device = n.device
    conds = ["lr_bicubic", "original_sr", "detail_sr"]
    nets = {"arcface_training_prior": priors.arc, "mobileface_not_in_loss": priors.mobile}
    bank = {net: {name: [] for name in conds} for net in nets}
    if benchmark == "lfw":
        experiment = r.LFWDegradedExperiment({}, priors.arc, n.Degrader())
        pairs, labels = experiment.load(ROOT / "work/recognition/lfw")
        hr = pairs.reshape(-1, 112, 112, 3)
        lrs = np.stack([n.Degrader()(image, np.random.default_rng(5000 + i)) for i, image in enumerate(hr)])
        for net in bank: bank[net]["hr_ceiling"] = []
        total = len(hr)
        print("LFW EVALUATION", len(labels), "pairs", flush=True)
        def batches():
            for i in range(0, total, 64):
                tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float().div(255)
                yield tensor(lrs[i:i + 64]), tensor(hr[i:i + 64])
        source = batches()
    else:
        qmul = r.QMULData(str(ROOT / "work/recognition"))
        if qmul.missing: raise RuntimeError("Missing required QMUL images")
        files = sorted({name for pair in qmul.pairs for name in pair})
        dataset = r.ImageSet([qmul.paths[name] for name in files], 28)
        loader = n.DataLoader(dataset, batch_size=64, num_workers=0)
        source = ((low, None) for native, low in loader)
        total = len(files)
        labels = qmul.pair_labels
        print("QMUL EVALUATION", len(files), "images", len(labels), "verification pairs", flush=True)
    start = last = time.monotonic()
    count = 0
    with torch.inference_mode():
        for low, clean in source:
            low = low.to(n.device)
            views = {"lr_bicubic": F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1),
                     "original_sr": old(low).clamp(0, 1), "detail_sr": new(low).clamp(0, 1)}
            if clean is not None: views["hr_ceiling"] = clean.to(n.device)
            for net_name, net in nets.items():
                for name, image in views.items(): bank[net_name][name].append(net(image).cpu())
            count += len(low)
            if time.monotonic() - last > 25:
                print(f"{benchmark.upper()} embedded {count}/{total}", flush=True); last = time.monotonic()
    results = {}
    for net_name, conditions in bank.items():
        results[net_name] = {}
        for name, values in conditions.items():
            embeddings = torch.cat(values).numpy()
            if benchmark == "lfw":
                folds = np.arange(len(labels)) * 10 // len(labels)
                value = r.VerificationEval()(embeddings[0::2], embeddings[1::2], np.asarray(labels), folds)
            else:
                index = {name: i for i, name in enumerate(files)}
                ia = np.array([index[a] for a, b in qmul.pairs]); ib = np.array([index[b] for a, b in qmul.pairs])
                value = r.VerificationEval()(embeddings[ia], embeddings[ib], labels)
            results[net_name][name] = value
    write_json(folder / f"{benchmark}_verification.json", results)
    print("VERIFICATION RESULT", benchmark, json.dumps(results), "seconds", time.monotonic() - start, flush=True)
    return results


def visual_comparison(n, old, new, test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    low, hr = next(test.batches(6))
    with torch.inference_mode():
        rows = [hr, F.interpolate(low, scale_factor=4, mode="nearest"),
                old(low.to(n.device)).clamp(0, 1).cpu(), new(low.to(n.device)).clamp(0, 1).cpu()]
    labels = ["Clean original", "Low-res input", "Previous SR", "Detail-refined SR"]
    fig, axes = plt.subplots(4, 6, figsize=(11, 7.8), layout="constrained")
    for row, name, axes_row in zip(rows, labels, axes):
        for i, (image, axis) in enumerate(zip(row, axes_row)):
            axis.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_visible(False)
            if i == 0: axis.set_ylabel(name, fontsize=9)
    fig.suptitle("Same held-out inputs · original and refined checkpoints", fontsize=13)
    fig.savefig(path, dpi=160); plt.close(fig)


def make_report(folder, test_metrics, validation, lfw, qmul, acceptance, checkpoint):
    lines = ["**Detail-preserving SR refinement**", "",
        "The same 479,109-parameter SR architecture was fine-tuned from the original full-trained checkpoint. Inference still maps an LR image to an SR image; the feature networks are training/evaluation tools, not part of the deployed SR model.", "",
        f"The selected checkpoint adds {checkpoint['finetune_epoch']} fine-tune epochs to 30 base epochs. The run completed 5 fine-tune epochs on 67,000 training images. Selection used 1,000 validation images; no test examples were used for selecting the checkpoint or loss weights.", "",
        "Loss weights: " + json.dumps(checkpoint["loss_weights"]) + ". Pixel L1 is combined with frozen VGG16 feature L1, Sobel edge L1, and cosine feature distance from the frozen ResNet50 ArcFace network. MobileFaceNet is not used in the training loss.", "",
        "**Held-out FFHQ results: 2,000 images**", "",
        "| Method | PSNR-Y ↑ | SSIM-Y ↑ | VGG feature error ↓ | Edge error ↓ | MobileFace feature similarity ↑ |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, v in test_metrics.items():
        lines.append(f"| {name} | {v['psnr_y']:.4f} | {v['ssim_y']:.4f} | {v['vgg_l1']:.5f} | {v['edge_l1']:.5f} | {v['mobileface_cos']:.5f} |")
    for title, results in [("LFW: 6,000 pairs", lfw), ("QMUL-SurvFace: 10,640 pairs", qmul)]:
        lines += ["", f"**{title}**", "", "These are downstream recognizer verification scores, not classification accuracies belonging to the SR network.", "",
                  "| Frozen recognizer | Processing | Accuracy (%) | AUC | TAR@FAR 1% (%) |", "|---|---|---:|---:|---:|"]
        for net, conditions in results.items():
            for name, v in conditions.items():
                lines.append(f"| {net} | {name} | {100*v['acc']:.3f} | {v['auc']:.5f} | {100*v['tar_far1']:.3f} |")
    lines += ["", "**Release checks**", ""]
    for key, value in acceptance.items(): lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines += ["", "The UI keeps the original pixel model available. Default promotion requires all release checks to pass. A failed candidate may be exposed only as an explicitly labelled experimental comparison. The reference/original image is never supplied to the SR network at inference.", "",
        "Some details were removed by blur, noise, compression, and downsampling to 28 × 28. Refinement estimates missing information; these measurements do not certify a person's identity or guarantee that every texture in the output is accurate. Human recognizability is not established merely by a feature-similarity score.", "",
        "The independent recognizer has a different architecture, but both recognizers were pretrained using WebFace600K. Benchmark pretraining identity overlap and identity-disjoint FFHQ splits are not established. No new QMUL identification Rank-1/Rank-5 results are claimed for this refinement.", "",
        "**Sources informing the training objectives**", "",
        "- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)",
        "- [Super-Identity Convolutional Neural Network for Face Hallucination](https://openaccess.thecvf.com/content_ECCV_2018/html/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.html)",
        "- [Official InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)", "",
        "This implementation is an extension of the project's SR training objective, not a reproduction of those complete methods.", ""]
    (folder / "refinement_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="detail_full")
    args = parser.parse_args()
    folder = ROOT / "work/refinement" / args.run
    validation = json.loads((folder / "validation_result.json").read_text())
    if not validation["passes_validation"]:
        raise RuntimeError("This run has not passed validation release checks; test evaluation was not started.")
    n, cfg = runtime(); priors = Priors(n.device)
    old, original_checkpoint = load_model(n, cfg, BASE)
    new, checkpoint = load_model(n, cfg, folder / "best.pt")
    assert checkpoint["manifest"]["base_checkpoint_sha256"] == sha(BASE)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    assert last["finetune_epoch"] == 5
    del last, original_checkpoint
    hr = np.load(ROOT / "work/data/ffhq128_u8.npy", mmap_mode="r")
    with np.load(ROOT / "work/full/split_ids.npz") as split: ids = split["test"]
    test = n.FixedEvalSet(hr, ids, n.Degrader(), 112, 2000)
    methods = {"bicubic": lambda x: F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False), "original_sr": old, "detail_sr": new}
    metrics = {}
    for name, model in methods.items():
        metrics[name] = evaluate(n, model, test, priors)
        print("HELDOUT RESULT", name, json.dumps(metrics[name]), flush=True)
    write_json(folder / "heldout_metrics.json", metrics)
    visual_comparison(n, old, new, test, folder / "test_comparison.png")
    lfw = recognition(n, priors, old, new, folder, "lfw")
    qmul = recognition(n, priors, old, new, folder, "qmul")
    mobile = "mobileface_not_in_loss"
    acceptance = {"validation_passed": bool(validation["passes_validation"]),
        "heldout_perceptual_error_improves": metrics["detail_sr"]["vgg_l1"] < metrics["original_sr"]["vgg_l1"],
        "heldout_independent_feature_similarity_improves": metrics["detail_sr"]["mobileface_cos"] > metrics["original_sr"]["mobileface_cos"],
        "lfw_second_recognizer_no_material_regression": lfw[mobile]["detail_sr"]["acc"] >= lfw[mobile]["original_sr"]["acc"] - .005,
        "qmul_second_recognizer_no_material_regression": qmul[mobile]["detail_sr"]["auc"] >= qmul[mobile]["original_sr"]["auc"] - .005}
    release = {"checkpoint": str((folder / "best.pt").resolve()), "checkpoint_sha256": sha(folder / "best.pt"),
               "passes_release": all(acceptance.values()), "checks": acceptance, "heldout_metrics": metrics,
               "best_finetune_epoch": checkpoint["finetune_epoch"]}
    write_json(folder / "release_checks.json", release)
    if release["passes_release"]: write_json(ROOT / "work/refinement/promoted.json", release)
    make_report(folder, metrics, validation, lfw, qmul, acceptance, checkpoint)
    print("REFINEMENT RELEASE", json.dumps(release), flush=True)


if __name__ == "__main__": main()

"""Fine-tune the existing SR network for feature and edge preservation.

Training/selection use the original train/validation split. The original full
checkpoint is read-only. MobileFaceNet is never part of the training loss.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2torch import convert
from torch import nn
from torch.nn import functional as F
from torchvision.models.vgg import make_layers

from run_project import ROOT, ProgressLoader, notebook_definitions, runtime

ASSETS = ROOT / "work/refinement_assets"
BASE = ROOT / "work/full/ckpt/ours_ssc_attn_best.pt"
PRESETS = {
    "identity": {"pixel": 1., "identity": .05, "perceptual": 0., "edge": 0.},
    "detail": {"pixel": 1., "identity": .025, "perceptual": .02, "edge": .10},
    "detail_strong": {"pixel": 1., "identity": .02, "perceptual": .08, "edge": .30},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


class FrozenFace(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.net = convert(onnx.load(str(path))).eval().requires_grad_(False)

    def forward(self, image):
        value = self.net((image - .5) / .5)
        if isinstance(value, (tuple, list)): value = value[0]
        return F.normalize(value, dim=1)


class FrozenFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256])
        cache = ASSETS / "vgg16_features16.pt"
        if cache.exists():
            state = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            full = torch.load(ASSETS / "vgg16-397923af.pth", map_location="cpu", weights_only=True)
            state = {key: full["features." + key] for key in self.net.state_dict()}
            torch.save(state, cache)
            del full
        self.net.load_state_dict(state, strict=True)
        self.net.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image):
        value, outputs = (image - self.mean) / self.std, []
        for i, layer in enumerate(self.net):
            value = layer(value)
            if i in {3, 8, 15}: outputs.append(value)
        return outputs


class Priors(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.arc = FrozenFace(ROOT / "work/recognition/w600k_r50.onnx")
        self.mobile = FrozenFace(ASSETS / "w600k_mbf.onnx")
        self.vgg = FrozenFeatures()
        edge = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        self.register_buffer("sobel", torch.stack([edge, edge.T]).unsqueeze(1))
        self.register_buffer("luma", torch.tensor([.299, .587, .114]).view(1, 3, 1, 1))
        self.to(device).eval().requires_grad_(False)

    def edges(self, image):
        return F.conv2d((image * self.luma).sum(1, keepdim=True), self.sobel, padding=1)[:, :, 1:-1, 1:-1]

    @staticmethod
    def feature_distance(a, b):
        return sum((x - y).abs().flatten(1).mean(1) for x, y in zip(a, b)) / len(a)

    def loss(self, sr, hr, weights, subbatch):
        pixel = F.l1_loss(sr, hr)
        total = weights["pixel"] * pixel
        parts = {"pixel": pixel.detach().item()}
        if weights["edge"]:
            edge = F.l1_loss(self.edges(sr), self.edges(hr))
            total = total + weights["edge"] * edge
            parts["edge"] = edge.detach().item()
        small_sr, small_hr = sr[:subbatch].clamp(0, 1), hr[:subbatch]
        if weights["identity"]:
            with torch.no_grad(): target = self.arc(small_hr)
            identity = 1 - (self.arc(small_sr) * target).sum(1).mean()
            total = total + weights["identity"] * identity
            parts["identity"] = identity.detach().item()
        if weights["perceptual"]:
            with torch.no_grad(): target = self.vgg(small_hr)
            perceptual = self.feature_distance(self.vgg(small_sr), target).mean()
            total = total + weights["perceptual"] * perceptual
            parts["perceptual"] = perceptual.detach().item()
        return total, parts


@torch.no_grad()
def evaluate(n, fn, data, priors, bs=32):
    if isinstance(fn, nn.Module): fn.eval()
    values = {key: [] for key in ["psnr_y", "ssim_y", "pixel_l1", "edge_l1", "vgg_l1", "arcface_cos", "mobileface_cos"]}
    for lr, hr in data.batches(bs):
        low, target = lr.to(n.device), hr.to(n.device)
        sr = fn(low).clamp(0, 1)
        values["psnr_y"].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
        values["ssim_y"].extend(n.Metrics.ssim_y(sr.cpu(), hr, 4).tolist())
        values["pixel_l1"].extend((sr - target).abs().flatten(1).mean(1).cpu().tolist())
        values["edge_l1"].extend((priors.edges(sr) - priors.edges(target)).abs().flatten(1).mean(1).cpu().tolist())
        values["vgg_l1"].extend(priors.feature_distance(priors.vgg(sr), priors.vgg(target)).cpu().tolist())
        values["arcface_cos"].extend((priors.arc(sr) * priors.arc(target)).sum(1).cpu().tolist())
        values["mobileface_cos"].extend((priors.mobile(sr) * priors.mobile(target)).sum(1).cpu().tolist())
    result = {key: float(np.mean(items)) for key, items in values.items()}
    assert all(math.isfinite(v) for v in result.values()), result
    return result


def score(metrics):
    return metrics["mobileface_cos"] - .04 * metrics["vgg_l1"] - .2 * metrics["edge_l1"]


def eligible(metrics, original):
    return metrics["psnr_y"] >= original["psnr_y"] - 1.5 and metrics["ssim_y"] >= original["ssim_y"] - .05


def promotion_check(metrics, original):
    return {"pixel_quality_guard": eligible(metrics, original),
            "independent_feature_similarity_improves": metrics["mobileface_cos"] > original["mobileface_cos"] + .005,
            "perceptual_error_improves": metrics["vgg_l1"] < original["vgg_l1"] * .98,
            "edge_error_not_materially_worse": metrics["edge_l1"] <= original["edge_l1"] * 1.05}


def save_grid(n, model, old, data, out):
    low, hr = next(data.batches(6))
    with torch.no_grad():
        a = old(low.to(n.device)).clamp(0, 1).cpu()
        b = model(low.to(n.device)).clamp(0, 1).cpu()
        bic = F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    rows = [hr, F.interpolate(low, scale_factor=4, mode="nearest"), bic, a, b]
    grid = np.concatenate([np.concatenate([(image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8) for image in row], 1) for row in rows], 0)
    n.Image.fromarray(grid).save(out)
    out.with_suffix(".txt").write_text("Rows: clean HR; nearest LR; bicubic; original L1 SR; refined SR. These are validation images.\n")


def fine_tune(n, priors, args, mode, hr, train_ids, val_ids):
    out = ROOT / "work/refinement" / (args.name or ("trial_" + mode))
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    weights = PRESETS[mode]
    cfg = dataclasses.replace(n.cfg, work_dir=str(out), epochs=args.epochs, lr=args.lr, batch=args.batch)
    torch.manual_seed(cfg.seed)
    val = n.FixedEvalSet(hr, val_ids[:args.val_count], n.Degrader(), 112, 1000)
    original_ckpt = torch.load(BASE, map_location="cpu", weights_only=True)
    original = n.SRNet(cfg).to(n.device).eval()
    original.load_state_dict(original_ckpt["model"])
    model = n.SRNet(cfg).to(n.device)
    model.load_state_dict(original_ckpt["model"])
    del original_ckpt
    selected_train = train_ids[:args.train_limit] if args.train_limit else train_ids
    loader = n.DataLoader(n.FaceTrainSet(ROOT / "work/data/ffhq128_u8.npy", selected_train, n.Degrader(), 112, seed=cfg.seed),
                          batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True, pin_memory=n.device == "cuda", generator=torch.Generator().manual_seed(cfg.seed))
    manifest = {"mode": mode, "weights": weights, "epochs": args.epochs, "batch": args.batch, "subbatch": args.subbatch,
                "learning_rate": args.lr, "train_images": len(selected_train), "validation_images": len(val),
                "base_checkpoint_sha256": sha(BASE), "model_source_sha256": sha(ROOT / "sr_part1_clean.ipynb"),
                "runner_sha256": sha(__file__), "train_ids_sha256": hashlib.sha256(selected_train.tobytes()).hexdigest(),
                "validation_ids_sha256": hashlib.sha256(val_ids[:args.val_count].tobytes()).hexdigest(),
                "selection_score": "mobileface_cos - 0.04*vgg_l1 - 0.2*edge_l1; PSNR >= old-1.5 dB and SSIM >= old-0.05",
                "independent_evaluator": "MobileFaceNet (not included in training loss)", "test_images_used_for_selection": 0}
    manifest_file = out / "manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise ValueError(f"Run configuration changed: choose a new --name ({out})")
    write_json(manifest_file, manifest)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=1e-6)
    baseline_path = out / "baseline_validation.json"
    if baseline_path.exists(): baseline = json.loads(baseline_path.read_text())
    else:
        baseline = {"original_sr": evaluate(n, original, val, priors), "bicubic": evaluate(n, lambda x: F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False), val, priors)}
        write_json(baseline_path, baseline)
    history, epoch, best_score = [], 0, score(baseline["original_sr"])
    best_metrics = baseline["original_sr"]
    def save(path, completed):
        temp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "epoch": 30 + completed,
                    "pretrain_epochs": 30, "finetune_epoch": completed, "loss_weights": weights, "mode": mode,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                    "best_score": best_score, "best_metrics": best_metrics, "manifest": manifest,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if n.device == "cuda" else []}, temp)
        temp.replace(path)
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=n.device, weights_only=True)
        assert ck["manifest"] == manifest
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        epoch, history, best_score, best_metrics = ck["finetune_epoch"], ck["history"], ck["best_score"], ck["best_metrics"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if n.device == "cuda": torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        print(f"RESUME {mode} epoch {epoch}", flush=True)
    else: save(out / "best.pt", 0)
    print("BASELINE", mode, json.dumps(baseline), flush=True)
    for ep in range(epoch, args.epochs):
        loader.dataset.set_epoch(30 + ep); loader.generator.manual_seed(cfg.seed + 30 + ep)
        model.train(); sums = {}; start = time.perf_counter()
        for i, (low, target) in enumerate(ProgressLoader(loader)):
            low, target = low.to(n.device, non_blocking=True), target.to(n.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, parts = priors.loss(model(low), target, weights, args.subbatch)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite refinement loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); sched.step()
            for key, value in parts.items(): sums[key] = sums.get(key, 0.) + value
        metrics = evaluate(n, model, val, priors)
        record = {"epoch": ep + 1, "losses": {k: v / len(loader) for k, v in sums.items()}, "validation": metrics, "seconds": time.perf_counter() - start}
        history.append(record)
        if eligible(metrics, baseline["original_sr"]) and score(metrics) > best_score:
            best_score, best_metrics = score(metrics), metrics
            save(out / "best.pt", ep + 1)
        save(out / "last.pt", ep + 1)
        write_json(out / "history.json", history)
        print("REFINEMENT EPOCH", mode, json.dumps(record), flush=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=n.device, weights_only=True)["model"])
    model.eval()
    checks = promotion_check(best_metrics, baseline["original_sr"])
    result = {"mode": mode, "best_validation": best_metrics, "baseline_validation": baseline,
              "promotion_checks": checks, "passes_validation": all(checks.values()), "checkpoint": str(out / "best.pt")}
    write_json(out / "validation_result.json", result)
    save_grid(n, model, original, val, out / "validation_comparison.png")
    print("REFINEMENT RESULT", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["identity", "detail", "detail_strong", "both"], default="both")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=2048, help="0 uses all 67,000 training images")
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--subbatch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    args = parser.parse_args()
    if args.name and args.mode == "both": parser.error("Use one mode with --name")
    n, cfg = runtime()
    if n.device != "cuda": raise RuntimeError("This refinement run expects CUDA; the current environment has no GPU support")
    print("Loading frozen feature networks…", flush=True)
    priors = Priors(n.device)
    # Confirm gradients pass through the frozen training priors into the image.
    x = torch.rand(2, 3, 112, 112, device=n.device, requires_grad=True)
    arc_gradient = torch.autograd.grad(priors.arc(x)[:, 0].sum(), x)[0]
    assert torch.isfinite(arc_gradient).all() and arc_gradient.abs().sum() > 0
    loss = sum(f.mean() for f in priors.vgg(x))
    loss.backward(); assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with torch.no_grad(): assert priors.mobile(x).shape == (2, 512)
    assert not any(p.requires_grad for p in priors.parameters())
    del x, loss
    print("Frozen-prior forward/gradient checks passed", flush=True)
    hr = np.load(ROOT / "work/data/ffhq128_u8.npy", mmap_mode="r")
    with np.load(ROOT / "work/full/split_ids.npz") as split:
        train_ids, val_ids = split["train"], split["validation"]
    for mode in (["identity", "detail"] if args.mode == "both" else [args.mode]):
        fine_tune(n, priors, args, mode, hr, train_ids, val_ids)
        torch.cuda.empty_cache()


if __name__ == "__main__": main()

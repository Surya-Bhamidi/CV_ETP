r"""Run the existing notebook definitions on Windows without notebook side effects.

Full experiment: .venv\Scripts\python.exe -u run_project.py train
Fast, explicitly separate pilot: append --pilot
The original notebooks are the source of model, loss, and metric definitions.
"""
import argparse
import ast
import dataclasses
import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def notebook_definitions(filename, cells, name):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / filename)
    sys.modules[name] = module
    notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    for i in cells:
        tree = ast.parse("".join(notebook["cells"][i]["source"]))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        exec(compile(tree, f"{filename}:cell{i}", "exec"), module.__dict__)
    return module


def runtime(pilot=False):
    n = notebook_definitions("sr_part1_clean.ipynb", range(1, 7), "project_part1")
    cfg = n.Config()
    cfg.work_dir = str(ROOT / "work" / ("pilot" if pilot else "full"))
    cfg.num_workers = 0  # Windows cannot spawn notebook-defined DataLoader workers.
    if pilot:
        cfg.n_val, cfg.n_test, cfg.epochs = 256, 256, 5
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    n.cfg = cfg
    n.device = "cuda" if n.torch.cuda.is_available() else "cpu"
    n.torch.set_num_threads(4)
    n.torch.manual_seed(cfg.seed)
    n.np.random.seed(cfg.seed)
    n.torch.backends.cudnn.benchmark = True
    return n, cfg


def sanity(n, cfg):
    n.sanity_check()
    stats = {"device": n.device, "torch": n.torch.__version__, "torchvision": __import__("torchvision").__version__, "models": {}}
    if n.device == "cuda":
        stats["gpu"] = n.torch.cuda.get_device_name(0)
    for tag, options in [("baseline_conv", dict(use_ssc=False, use_attn=False)), ("ours_ssc_attn", {})]:
        model = n.SRNet(cfg, **options).to(n.device)
        x = n.torch.rand(cfg.batch, 3, 28, 28, device=n.device)
        target = n.torch.rand(cfg.batch, 3, 112, 112, device=n.device)
        optimizer = n.torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        times = []
        for i in range(4):
            if n.device == "cuda": n.torch.cuda.synchronize()
            t = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = n.F.l1_loss(prediction, target)
            loss.backward()
            optimizer.step()
            if n.device == "cuda": n.torch.cuda.synchronize()
            if i: times.append(time.perf_counter() - t)
        assert prediction.shape == target.shape
        assert n.torch.isfinite(loss)
        assert all(n.torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        stats["models"][tag] = {"parameters": n.n_params(model), "batch": cfg.batch, "training_step_seconds": sum(times) / len(times), "finite_gradients": True}
        print(tag, json.dumps(stats["models"][tag]), flush=True)
        del model, optimizer, x, target, prediction, loss
        if n.device == "cuda": n.torch.cuda.empty_cache()
    (Path(cfg.work_dir) / "sanity.json").write_text(json.dumps(stats, indent=2))
    return stats


class ProgressLoader:
    def __init__(self, loader):
        self.loader = loader
    def __len__(self):
        return len(self.loader)
    @property
    def dataset(self):
        return self.loader.dataset
    @property
    def generator(self):
        return self.loader.generator
    def __iter__(self):
        last = time.monotonic()
        iterator = iter(self.loader)
        # One CPU producer overlaps preparation with GPU training. It preserves
        # DataLoader order and avoids Windows notebook multiprocessing imports.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(next, iterator)
            for i in range(len(self)):
                batch = pending.result()
                if i + 1 < len(self): pending = pool.submit(next, iterator)
                if time.monotonic() - last >= 25:
                    print(f"  training batch {i + 1}/{len(self)}", flush=True)
                    last = time.monotonic()
                yield batch


def train(n, cfg, pilot):
    data = ROOT / "work" / "data" / "ffhq128_u8.npy"
    if not data.exists():
        raise FileNotFoundError("Run prepare_data.py first")
    hr = n.np.load(data, mmap_mode="r")
    perm = n.np.random.default_rng(cfg.seed).permutation(len(hr))
    test_ids = perm[:cfg.n_test]
    val_ids = perm[cfg.n_test:cfg.n_test + cfg.n_val]
    train_ids = perm[cfg.n_test + cfg.n_val:]
    if pilot:
        train_ids = train_ids[:4096]
    deg = n.Degrader(cfg.scale)
    val_set = n.FixedEvalSet(hr, val_ids, deg, cfg.hr_size, 1000)
    test_set = n.FixedEvalSet(hr, test_ids, deg, cfg.hr_size, 2000)
    loader = n.DataLoader(n.FaceTrainSet(data, train_ids, deg, cfg.hr_size, seed=cfg.seed), batch_size=cfg.batch, shuffle=True, num_workers=0, drop_last=True, pin_memory=n.device == "cuda", generator=n.torch.Generator().manual_seed(cfg.seed))
    manifest = {"protocol": "rgb_jpeg_seeded_v2", "pilot": pilot, "config": dataclasses.asdict(cfg), "device": n.device, "train_images": len(train_ids), "validation_images": len(val_ids), "test_images": len(test_ids), "notebook_sha256": hashlib.sha256((ROOT / "sr_part1_clean.ipynb").read_bytes()).hexdigest(), "execution_changes": ["num_workers=0 for Windows", "FFHQ cache built directly from the same dataset's image archive", "audited RGB JPEG handling, seeded epoch augmentation, atomic validated checkpoints"]}
    integrity = ROOT / "work" / "audit" / "data_integrity.json"
    if integrity.exists(): manifest["data_integrity"] = json.loads(integrity.read_text())
    manifest_path = Path(cfg.work_dir) / "manifest.json"
    if manifest_path.exists() and any((Path(cfg.work_dir) / "ckpt").glob("*.pt")):
        previous = json.loads(manifest_path.read_text())
        if previous != manifest: raise RuntimeError("Existing run has a different configuration, source, or dataset. Use a new run directory.")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n.np.savez(Path(cfg.work_dir) / "split_ids.npz", train=train_ids, validation=val_ids, test=test_ids)
    print(json.dumps(manifest, indent=2), flush=True)
    bicubic = lambda x: n.F.interpolate(x, scale_factor=cfg.scale, mode="bicubic", align_corners=False)
    results = {"bicubic": {**n.evaluate(bicubic, test_set, n.device, ssim=True), "params": 0}}
    output = Path(cfg.work_dir) / "part1_results.json"
    def save():
        output.write_text(json.dumps(results, indent=2))
        print("RESULTS", json.dumps(results), flush=True)
    save()
    models = {}
    for tag, options in [("ours_ssc_attn", {}), ("baseline_conv", dict(use_ssc=False, use_attn=False))]:
        n.torch.manual_seed(cfg.seed)
        model = n.SRNet(cfg, **options)
        trainer = n.Trainer(model, cfg, ProgressLoader(loader), val_set, tag)
        start = time.perf_counter()
        trainer.fit()
        trainer.load_best()
        results[tag] = {**n.evaluate(model, test_set, n.device, ssim=True), "params": n.n_params(model), "best_validation_psnr_y": trainer.best_psnr, "epochs_completed": trainer.epoch, "training_and_evaluation_seconds_this_invocation": time.perf_counter() - start}
        models[tag] = model
        save()
    # Shared, fixed test examples make visible artifacts reviewable.
    lr, clean = next(test_set.batches(6))
    rows = [n.F.interpolate(lr, scale_factor=cfg.scale, mode="nearest"), bicubic(lr)]
    with n.torch.no_grad():
        rows += [models[k](lr.to(n.device)).clamp(0, 1).cpu() for k in ["baseline_conv", "ours_ssc_attn"]]
    rows.append(clean)
    grid = n.np.concatenate([n.np.concatenate([(im.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8") for im in row], axis=1) for row in rows], axis=0)
    n.Image.fromarray(grid).save(Path(cfg.work_dir) / "comparison.png")
    (Path(cfg.work_dir) / "comparison_rows.txt").write_text("Top to bottom: nearest enlarged LR; bicubic; baseline_conv; ours_ssc_attn; clean HR.\n")
    print("COMPLETE", output, flush=True)


def recognition(pilot=False, benchmark="lfw", verification_only=False):
    n, cfg = runtime(pilot)
    r = notebook_definitions("sr_part2a_recognition.ipynb", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11], "project_part2a")
    r.cfg, r.device = cfg, n.device
    r.bicubic = lambda x: r.F.interpolate(x, scale_factor=cfg.scale, mode="bicubic", align_corners=False)
    arc = r.ArcFace(str(ROOT / "work" / "recognition")).to(n.device).eval()
    models = {"baseline_conv": r.load_sr("baseline_conv", use_ssc=False, use_attn=False), "ours_ssc_attn": r.load_sr("ours_ssc_attn")}
    if benchmark == "lfw":
        experiment = r.LFWDegradedExperiment(models, arc, r.Degrader(cfg.scale))
        pairs, labels = experiment.load(ROOT / "work" / "recognition" / "lfw")
        print("LFW pairs:", len(labels), flush=True)
        results = experiment.run(pairs, labels, bs=64)
        r.show(results, "LFW controlled identity evaluation")
    else:
        qmul = r.QMULData(str(ROOT / "work" / "recognition"))
        qmul.size_stats()
        # Verify all files first: the notebook silently excludes missing images.
        coverage = {"required_images": len(qmul.files) + len(qmul.missing), "available_images": len(qmul.files), "missing_images": len(qmul.missing), "missing_examples": qmul.missing[:20]}
        (Path(cfg.work_dir) / "qmul_coverage.json").write_text(json.dumps(coverage, indent=2))
        if qmul.missing:
            raise RuntimeError(f"QMUL has {len(qmul.missing)} missing required images; refusing to report incomplete-protocol results")
        if verification_only:
            qmul.files = sorted({f for pair in qmul.pairs for f in pair})
        bank = r.EmbeddingBank(models, arc, cfg).compute(qmul.files, qmul.paths, tag="qmul_verification" if verification_only else "qmul", bs=64)
        ix = {f: i for i, f in enumerate(qmul.files)}
        ia = r.np.array([ix[a] for a, b in qmul.pairs])
        ib = r.np.array([ix[b] for a, b in qmul.pairs])
        if not verification_only:
            gi = r.np.array([ix[f] for f in qmul.gal_files])
            pi = r.np.array([ix[f] for f in qmul.pro_files])
        results = {}
        for condition in r.EmbeddingBank.CONDS:
            E = bank[condition]
            results[condition] = {"ver": r.VerificationEval()(E[ia], E[ib], qmul.pair_labels)}
            if not verification_only:
                results[condition]["ident"] = r.IdentificationEval(n.device)(E[gi], qmul.gal_ids, E[pi], qmul.pro_ids)
        r.show(results, "QMUL-SurvFace: verification" + ("" if verification_only else " and closed-set identification"))
    suffix = benchmark + ("_verification" if verification_only else "")
    out = Path(cfg.work_dir) / f"part2a_{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print("RECOGNITION RESULTS", json.dumps(results), flush=True)
    print("COMPLETE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sanity", "train", "recognition"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--benchmark", choices=["lfw", "qmul"], default="lfw")
    parser.add_argument("--verification-only", action="store_true", help="Run all QMUL verification pairs without the larger identification experiment")
    parser.add_argument("--log-file", help="Write stdout and stderr to this file as well as the console")
    args = parser.parse_args()
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8", buffering=1)
        class Tee:
            def __init__(self, console): self.console = console
            def write(self, text):
                self.console.write(text); log.write(text)
            def flush(self): self.console.flush(); log.flush()
            def isatty(self): return False
        sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
        print("RUN START UTC", datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    if args.command == "recognition":
        recognition(args.pilot, args.benchmark, args.verification_only)
    else:
        n, cfg = runtime(args.pilot)
        if args.command == "sanity": sanity(n, cfg)
        else: train(n, cfg, args.pilot)


if __name__ == "__main__":
    main()

"""Pretraining regression checks for data, attention, learning, and resume."""
import ast
import copy
import dataclasses
import json
import tempfile
import time
from pathlib import Path

from run_project import ROOT, runtime, notebook_definitions


def main():
    out = ROOT / "work" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    n, cfg = runtime()
    checks = {}
    for path in ROOT.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("\n".join("" if s.lstrip().startswith(("!", "%")) else s for s in "".join(cell["source"]).splitlines()))
    checks["notebook_syntax"] = True
    data = ROOT / "work" / "data" / "ffhq128_u8.npy"
    hr = n.np.load(data, mmap_mode="r")
    ds = n.FaceTrainSet(data, [3001, 3002, 3003], n.Degrader(), 112, seed=42)
    a, b = ds[0], ds[0]
    assert all(n.torch.equal(x, y) for x, y in zip(a, b))
    ds.set_epoch(1)
    c = ds[0]
    assert not n.torch.equal(a[0], c[0])
    ds.set_epoch(0)
    assert n.torch.equal(ds[0][0], a[0])
    assert a[0].shape == (3, 28, 28) and a[1].shape == (3, 112, 112)
    assert all(0 <= t.min() <= t.max() <= 1 for t in a)
    checks["epoch_seeded_augmentation_and_ranges"] = True
    # JPEG should equal an explicitly RGB-aware OpenCV reference.
    image = n.np.asarray(hr[123, 8:120, 8:120]).copy()
    got = n.Degrader(4, blur_p=0, noise=(0, 0), jpeg=(40, 41))(image, n.np.random.default_rng(5))
    lr = n.np.asarray(n.Image.fromarray(image).resize((28, 28), n.Image.Resampling.BICUBIC))
    ok, buf = n.cv2.imencode(".jpg", n.cv2.cvtColor(lr, n.cv2.COLOR_RGB2BGR), [n.cv2.IMWRITE_JPEG_QUALITY, 40])
    reference = n.cv2.cvtColor(n.cv2.imdecode(buf, n.cv2.IMREAD_COLOR), n.cv2.COLOR_BGR2RGB)
    assert ok and n.np.array_equal(got, reference)
    checks["rgb_jpeg_reference"] = True
    ssc = n.SparseSamplingConv(8)
    x = n.torch.randn(2, 8, 8, 8, requires_grad=True)
    assert n.torch.allclose(ssc(x), n.F.conv2d(x, ssc.weight, ssc.bias, padding=1), atol=2e-6)
    ssc(x).square().mean().backward()
    assert ssc.offset.weight.grad.abs().sum() > 0
    checks["deformable_initialization_and_offset_gradients"] = True
    attn = n.AdaptiveSparseLinearAttention(8).double()
    x = n.torch.randn(2, 8, 3, 4, dtype=n.torch.float64, requires_grad=True)
    q, k, v = attn.qkv(x).flatten(2).chunk(3, dim=1)
    q, k = n.F.elu(q) + 1, (n.F.elu(k) + 1) * attn.key_mask(x).unsqueeze(1)
    weights = q.transpose(1, 2) @ k
    expected = attn.proj((v @ (weights / (weights.sum(-1, keepdim=True) + 1e-6)).transpose(1, 2)).reshape_as(x))
    actual = attn(x)
    assert n.torch.allclose(expected, actual, atol=1e-9, rtol=1e-8)
    actual.square().mean().backward()
    assert attn.score[0].weight.grad.abs().sum() > 0
    assert all(n.torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)
    checks["linear_attention_matches_explicit_attention"] = True
    r = notebook_definitions("sr_part2a_recognition.ipynb", [1, 9], "audit_recognition")
    rng = n.np.random.default_rng(8)
    for _ in range(2000):
        scores, labels = rng.choice([-.5, 0., .5], size=8), rng.integers(0, 2, 8)
        threshold = r.VerificationEval.best_threshold(scores, labels)
        best = max(((scores >= t) == labels).mean() for t in [-1, -.25, .25, 1])
        assert ((scores >= threshold) == labels).mean() == best
    checks["verification_ties_bruteforce_2000_cases"] = True
    identical = n.torch.rand(2, 3, 112, 112)
    assert n.torch.allclose(n.Metrics.psnr_y(identical, identical, 4), n.torch.full((2,), 100.))
    ssim = n.Metrics.ssim_y(identical, identical, 4)
    assert n.torch.allclose(ssim, n.torch.ones_like(ssim))
    checks["metric_identity_reference"] = True
    print("Core checks passed", json.dumps(checks), flush=True)
    # Check real learning on one fixed face: an SR network should beat bicubic
    # when allowed to overfit this tiny diagnostic example.
    device = n.device
    n.torch.manual_seed(42)
    fixed = n.FixedEvalSet(hr, [1234], n.Degrader(), 112, 123)
    lr, target = next(fixed.batches(1))
    lr, target = lr.to(device), target.to(device)
    model = n.SRNet(cfg).to(device)
    opt = n.torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    bic = n.F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    base_l1 = n.F.l1_loss(bic, target).item()
    base_psnr = n.Metrics.psnr_y(bic.clamp(0, 1), target, 4).item()
    start = time.perf_counter()
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        prediction = model(lr)
        loss = n.F.l1_loss(prediction, target)
        assert n.torch.isfinite(loss)
        loss.backward()
        n.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    with n.torch.no_grad():
        prediction = model(lr)
        final_l1 = n.F.l1_loss(prediction, target).item()
        final_psnr = n.Metrics.psnr_y(prediction.clamp(0, 1), target, 4).item()
    checks["single_face_overfit"] = {"steps": 200, "bicubic_l1": base_l1, "model_l1": final_l1, "bicubic_psnr": base_psnr, "model_psnr": final_psnr, "seconds": time.perf_counter() - start}
    assert final_l1 < base_l1 * .97, checks["single_face_overfit"]
    print("Learning diagnostic", json.dumps(checks["single_face_overfit"]), flush=True)
    del model, opt, lr, target, prediction, loss
    if device == "cuda": n.torch.cuda.empty_cache()
    # A CPU toy SR network makes interrupted/resumed training exactly testable.
    n.device = "cpu"
    class InterruptDataset(n.FaceTrainSet):
        interrupt = True
        def __getitem__(self, i):
            if self.interrupt and self._epoch.item() == 1:
                raise InterruptedError("intentional audit interruption")
            return super().__getitem__(i)
    initial = n.nn.Sequential(n.nn.Upsample(scale_factor=4, mode="nearest"), n.nn.Conv2d(3, 3, 1))
    val = n.FixedEvalSet(hr, [5001, 5002], n.Degrader(), 112, 1000)
    with tempfile.TemporaryDirectory(dir=out, prefix="resume_check_") as tmp:
        def trainer(folder, interrupted):
            conf = dataclasses.replace(cfg, work_dir=str(Path(tmp) / folder), epochs=2, batch=2)
            Path(conf.work_dir).mkdir(exist_ok=True)
            dataset = InterruptDataset(data, [3001, 3002, 3003, 3004], n.Degrader(), 112)
            dataset.interrupt = interrupted
            loader = n.DataLoader(dataset, batch_size=2, shuffle=True, generator=n.torch.Generator().manual_seed(42))
            return n.Trainer(copy.deepcopy(initial), conf, loader, val, "toy")
        full = trainer("uninterrupted", False)
        full.fit()
        broken = trainer("resumed", True)
        try: broken.fit()
        except InterruptedError: pass
        else: raise AssertionError("Interruption did not occur")
        resumed = trainer("resumed", False)
        assert resumed.epoch == 1
        resumed.fit()
        for key, value in full.model.state_dict().items():
            assert n.torch.equal(value, resumed.model.state_dict()[key]), key
        assert resumed.history == full.history
        mismatched = dataclasses.replace(resumed.cfg, lr=1e-3)
        try: n.Trainer(copy.deepcopy(initial), mismatched, resumed.loader, val, "toy")
        except ValueError: pass
        else: raise AssertionError("Mismatched checkpoint was accepted")
    n.device = device
    checks["interrupted_resume_equals_uninterrupted"] = True
    checks["incompatible_checkpoint_rejected"] = True
    (out / "runtime_checks.json").write_text(json.dumps(checks, indent=2))
    print("ALL PRETRAINING CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()

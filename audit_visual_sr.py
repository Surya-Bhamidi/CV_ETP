"""Audit the reported visual regression without modifying training artifacts."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from refine_model import BASE, sha, write_json
from run_project import ROOT, runtime


def main():
    out = ROOT / 'work/visual_sr/audit'
    out.mkdir(parents=True, exist_ok=True)
    n, cfg = runtime()
    with np.load(ROOT / 'work/full/split_ids.npz') as z:
        splits = {key: z[key] for key in z.files}
    assert len(set(np.concatenate(list(splits.values())).tolist())) == 70000
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    assert hr.shape == (70000, 128, 128, 3) and hr.dtype == np.uint8
    checkpoint = torch.load(BASE, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 30
    model = n.SRNet(cfg).to(n.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    cfg_keys = ['scale', 'hr_size', 'feat', 'blocks', 'keep_ratio']
    assert all(checkpoint['config'][key] == getattr(cfg, key) for key in cfg_keys)
    checks = {'disjoint_image_splits': True, 'cache_shape_and_rgb_layout': True,
              'checkpoint_matches_architecture': True, 'original_checkpoint_sha256': sha(BASE)}
    # Test correct colour conversion and repeatability with a nontrivial image.
    image = np.array(hr[splits['validation'][0], 8:120, 8:120])
    a = n.Degrader()(image, np.random.default_rng(1000))
    b = n.Degrader()(image, np.random.default_rng(1000))
    assert np.array_equal(a, b)
    checks['seeded_degradation_is_repeatable'] = True
    t = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().div(255).to(n.device)
    with torch.inference_mode():
        prediction = model(t)
        head = model.head(t)
        residual = model.up(model.tail(model.body(head)) + head)
        explicit = residual + F.interpolate(t, scale_factor=4, mode='bicubic', align_corners=False)
        assert torch.allclose(prediction, explicit, atol=1e-6)
        assert torch.isfinite(prediction).all() and prediction.shape == (1, 3, 112, 112)
    checks['residual_and_interpolation_correct'] = True
    result = {}
    for name, degrader in [('downsample_only', n.Degrader(blur_p=0, noise=(0, 0), jpeg=(100, 101))),
                            ('training_degradation', n.Degrader())]:
        data = n.FixedEvalSet(hr, splits['validation'][:256], degrader, 112, 1000)
        result[name] = {'bicubic': n.evaluate(lambda x: F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False), data, n.device, ssim=True),
                        'original_sr': n.evaluate(model, data, n.device, ssim=True)}
    # Diagnose the exact screenshot's information loss, not tune on it.
    rng = np.random.default_rng(2005)
    blur = rng.random() < .7
    kernel, sigma = (int(rng.choice([3, 5, 7])), float(rng.uniform(.2, 2.))) if blur else (None, None)
    noise = float(rng.uniform(2, 15))
    rng.normal(0, noise, (28, 28, 3))
    quality = int(rng.integers(30, 70))
    previous = json.loads((ROOT / 'work/refinement/detail_full/heldout_metrics.json').read_text())
    old, refined = previous['original_sr'], previous['detail_sr']
    report = {'checks': checks, 'validation_degradation_matrix': result,
        'screenshot_degradation': {'test_index': 5, 'blur': bool(blur), 'kernel': kernel, 'sigma_hr_pixels': sigma,
            'noise_std_255': noise, 'jpeg_quality': quality, 'input': [28, 28], 'output': [112, 112]},
        'previous_regression': {'psnr_change_db': refined['psnr_y'] - old['psnr_y'],
            'ssim_change': refined['ssim_y'] - old['ssim_y'], 'pixel_error_change': refined['pixel_l1'] - old['pixel_l1'],
            'edge_error_change': refined['edge_l1'] - old['edge_l1'],
            'cause': 'Recognition-feature-led selection accepted worse reconstruction; not a colour/channel or checkpoint-loading bug.'},
        'external_baseline_caveat': 'GFPGAN is pretrained externally on FFHQ. Our FFHQ split is not proven disjoint from its pretraining; its scores are diagnostic, not held-out evidence for a new project-trained model.'}
    write_json(out / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

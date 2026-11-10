"""Compare reconstruction quality on identical inputs, with honest provenance."""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from quality_metric import load_lpips
from refine_model import BASE, FrozenFeatures, sha, write_json
from restoration_model import FaceRestoration
from run_project import ROOT, runtime
from train_visual_sr import grid, selection


def model_from(n, cfg, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = n.SRNet(cfg).to(n.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint


def mean(values):
    return {key: float(np.mean(value)) for key, value in values.items()}


def bootstrap_difference(a, b):
    diff = np.array(a) - np.array(b)
    rng = np.random.default_rng(72)
    estimates = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    return {'mean': float(diff.mean()), '95_percent_interval': np.quantile(estimates, [.025, .975]).tolist()}


def training_plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads((folder / 'history.json').read_text())
    baseline = json.loads((folder / 'baseline_validation.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for axis, key, label in zip(axes, ['psnr_y', 'ssim_y', 'vgg_l1'], ['PSNR-Y (higher)', 'SSIM-Y (higher)', 'VGG feature error (lower)']):
        axis.plot([item['epoch'] for item in history], [item['validation'][key] for item in history], 'o-', color='#426a36', label='New objective')
        axis.axhline(baseline[key], linestyle='--', color='#947257', label='Original checkpoint')
        axis.set_xlabel('Additional full-data epoch'); axis.set_title(label, fontsize=10)
        axis.set_xticks([item['epoch'] for item in history]); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed 1,000-image validation set; test data excluded', fontsize=12)
    fig.savefig(folder / 'training_curves.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='full_reconstruction')
    parser.add_argument('--count', type=int, default=2000)
    args = parser.parse_args()
    folder = ROOT / 'work/visual_sr' / args.run
    validation = json.loads((folder / 'validation_result.json').read_text())
    if not validation['passes_reconstruction_guards']:
        raise RuntimeError('Candidate failed validation; do not tune on test data.')
    n, cfg = runtime()
    torch.backends.cudnn.benchmark = False
    old, old_checkpoint = model_from(n, cfg, BASE)
    new, checkpoint = model_from(n, cfg, folder / 'best.pt')
    failed, _ = model_from(n, cfg, ROOT / 'work/refinement/detail_full/best.pt')
    assert checkpoint['manifest']['base_checkpoint_sha256'] == sha(BASE)
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
    assert last['finetune_epoch'] == checkpoint['manifest']['epochs']
    actual_epochs = int(last['finetune_epoch'])
    del last, old_checkpoint
    restorer = FaceRestoration(n.device)
    lpips = load_lpips(n.device)
    vgg = FrozenFeatures().to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as z:
        ids = z['test'][:args.count]
    data = n.FixedEvalSet(hr, ids, n.Degrader(), 112, 2000)
    names = ['bicubic', 'original_custom', 'previous_refinement', 'reconstruction_custom', 'gfpgan_pretrained', 'gfpgan_blend_75']
    scores = {name: {key: [] for key in ['psnr_y', 'ssim_y', 'lpips_squeeze', 'pixel_l1', 'vgg_l1']} for name in names}
    start = last_log = time.monotonic(); count = 0
    with torch.inference_mode():
        for low, target in data.batches(16):
            low, target = low.to(n.device), target.to(n.device)
            original = old(low).clamp(0, 1)
            restored = restorer(low)
            views = {'bicubic': F.interpolate(low, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1),
                'original_custom': original, 'previous_refinement': failed(low).clamp(0, 1),
                'reconstruction_custom': new(low).clamp(0, 1), 'gfpgan_pretrained': restored,
                'gfpgan_blend_75': .25 * original + .75 * restored}
            clean_features = vgg(target)
            for name, sr in views.items():
                values = scores[name]
                values['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
                values['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
                values['lpips_squeeze'].extend(lpips(sr, target, normalize=True).flatten().cpu().tolist())
                values['pixel_l1'].extend((sr - target).abs().flatten(1).mean(1).cpu().tolist())
                features = vgg(sr)
                values['vgg_l1'].extend((sum((a - b).abs().flatten(1).mean(1) for a, b in zip(features, clean_features)) / 3).cpu().tolist())
            count += len(low)
            if time.monotonic() - last_log > 25:
                print('QUALITY EVALUATION', count, '/', len(data), flush=True); last_log = time.monotonic()
    averages = {name: mean(values) for name, values in scores.items()}
    assert all(math.isfinite(value) for values in averages.values() for value in values.values()), 'Nonfinite reconstruction metric'
    write_json(folder / 'quality_metrics.json', averages)
    write_json(folder / 'per_image_metrics.json', scores)
    differences = {key: bootstrap_difference(scores['reconstruction_custom'][key], scores['original_custom'][key])
                   for key in ['psnr_y', 'ssim_y', 'lpips_squeeze']}
    write_json(folder / 'paired_differences.json', differences)
    original, candidate = averages['original_custom'], averages['reconstruction_custom']
    checks = {'full_training_completed': actual_epochs == checkpoint['manifest']['epochs'],
        'validation_reconstruction_guards': validation['passes_reconstruction_guards'],
        'test_psnr_within_0_20_db': candidate['psnr_y'] >= original['psnr_y'] - .20,
        'test_ssim_within_0_003': candidate['ssim_y'] >= original['ssim_y'] - .003,
        'test_independent_lpips_improves': candidate['lpips_squeeze'] < original['lpips_squeeze']}
    passed = all(checks.values())
    release = {'checkpoint': (folder / 'best.pt').relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(folder / 'best.pt'),
        'passes_reconstruction_checks': passed, 'checks': checks, 'finetune_epochs': checkpoint['finetune_epoch'],
        'epochs_run': actual_epochs, 'mode': checkpoint['manifest']['mode'], 'default_model': 'detail' if passed else 'standard',
        'description': 'Reconstruction-first fine-tuning; no recognition loss. Original remains available.',
        'scope': 'Synthetic FFHQ reconstruction only; no CCTV identity improvement is claimed.'}
    write_json(folder / 'reconstruction_checks.json', release)
    if passed:
        write_json(ROOT / 'work/visual_sr/selected.json', release)
    labels = {'Original custom SR': old, 'Previous refinement': failed, 'New custom SR': new, 'GFPGAN (pretrained)': restorer}
    grid(n, labels, data, folder / 'test_comparison.png')
    # Additional examples beyond the UI gallery avoid relying on six familiar faces.
    extra = n.FixedEvalSet(hr, ids[100:108], n.Degrader(), 112, 2100)
    grid(n, labels, extra, folder / 'additional_examples.png')
    external_note = ('GFPGAN 1.4 and CodeFormer are external pretrained models, not the custom 479,109-parameter project network. '
        'Their upstream training uses FFHQ; overlap with these FFHQ images is not excluded. Their FFHQ scores are diagnostic, '
        'not evidence of held-out generalization. Generative detail may differ from the actual person. No original/reference image is used at inference.')
    lines = ['**Reconstruction quality investigation**', '',
        'The previous refinement was a measured reconstruction regression: lower PSNR and SSIM and higher pixel/edge error. Its selection objective favoured recognizer features. '
        'Colour order, seed handling, checkpoint/config loading and residual/bicubic paths passed the new audit.', '',
        f'Three 8,192-image, two-epoch pilots were compared on 256 validation images. The selected {checkpoint["manifest"]["mode"]} setting then ran {actual_epochs} full epochs on 67,000 images; '
        f'checkpoint selection used 1,000 validation images and chose extra epoch {checkpoint["finetune_epoch"]}. No test image or supplied screenshot was used to train or choose weights.', '',
        'New objective: pixel L1, local structural similarity, multiscale detail and frozen VGG features. '
        'No recognizer appears in training or checkpoint selection. The adversarial pilot includes a separate patch discriminator; see manifest for the selected full run.', '',
        f'**Identical degraded inputs: {len(data)} FFHQ test images**', '',
        '| Model | PSNR-Y ↑ | SSIM-Y ↑ | LPIPS-Squeeze ↓ | VGG error ↓ |', '|---|---:|---:|---:|---:|']
    for name, value in averages.items():
        lines.append(f'| {name} | {value["psnr_y"]:.4f} | {value["ssim_y"]:.4f} | {value["lpips_squeeze"]:.5f} | {value["vgg_l1"]:.5f} |')
    lines += ['', 'LPIPS uses the official pretrained SqueezeNet v0.1 calibrated metric, excluded from training and checkpoint selection. '
        'PSNR/SSIM retain the original luminance and four-pixel border protocol. LPIPS measures the full RGB image. '
        'Paired bootstrap intervals are in paired_differences.json; they describe image-sampling uncertainty, not training-seed uncertainty.', '',
        '**Custom model acceptance checks**', '']
    lines += [f'- {key}: {"PASS" if value else "FAIL"}' for key, value in checks.items()]
    lines += ['', '**External restoration option**', '', external_note, '',
        'Restoration takes only the same low-resolution crop, resizes it internally to 512 pixels, synthesizes a face and returns exactly ×4 input dimensions. '
        'A user-controlled blend can trade the original custom result against the generative result. '
        'CodeFormer was also checked on eight validation images at fidelity 0/0.5/1 and with landmark alignment; '
        'visible ghosting and poorer validation structure led to rejection from the active UI. '
        'Its experiments remain in work/codeformer_assets and work/alignment_assets; it is not included in this final full-test table. '
        'A compact general Real-ESRGAN baseline was tested on 256 validation images with denoising weights 0.5/1, '
        'plus a small reflection-padding trial. It produced blocking and poorer PSNR/SSIM (about 23.32 dB / 0.657 at weight 0.5 '
        'versus the original custom model 25.70 dB / 0.733 on those same images), so it was also rejected. '
        'See work/realesrgan_assets for those results. '
        'GFPGAN is a separate selectable option, never reported as a newly trained custom model.', '',
        '**Limits**', '',
        'These images are held out from custom-model training, but earlier models have already been evaluated on this test split. '
        'Scores here are repeated comparisons on that fixed split, not a fresh benchmark. '
        'Image splits are disjoint; identity-disjointness is not established. Human recognizability and CCTV verification are not established by these reconstruction scores. '
        'Examples remain difficult where the LR image does not contain enough information.', '',
        '[GFPGAN official implementation](https://github.com/TencentARC/GFPGAN) · '
        '[Official FFHQ training configuration](https://github.com/TencentARC/GFPGAN/blob/master/options/train_gfpgan_v1.yml) · '
        '[CodeFormer official implementation](https://github.com/sczhou/CodeFormer) · '
        '[LPIPS official implementation](https://github.com/richzhang/PerceptualSimilarity)', '']
    (folder / 'quality_report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(folder / 'evaluation_summary.json', {'seconds': time.monotonic() - start, 'test_images': len(data),
        'metrics': averages, 'reconstruction_checks': checks, 'external_baseline_caveat': external_note})
    training_plot(folder)
    print('QUALITY RESULT', json.dumps(averages), flush=True)
    print('RECONSTRUCTION CHECKS', json.dumps(release), flush=True)


if __name__ == '__main__':
    main()

"""Reconstruction-first experiments for the existing custom x4 SR model.

No recognition network is used for training or checkpoint selection. Every
candidate starts from the original checkpoint. Validation, never test data,
selects loss settings and checkpoints. Original experiment files are immutable.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from refine_model import BASE, FrozenFeatures, sha, write_json
from run_project import ROOT, ProgressLoader, runtime

PRESETS = {
    'structure': {'pixel': 1., 'structure': .15, 'perceptual': .01, 'laplacian': .10, 'adversarial': 0.},
    'perceptual': {'pixel': 1., 'structure': .15, 'perceptual': .08, 'laplacian': .10, 'adversarial': 0.},
    'visual_gan': {'pixel': 1., 'structure': .15, 'perceptual': .08, 'laplacian': .10, 'adversarial': .005},
}


def structure_loss(a, b):
    # Local SSIM on luminance, stable and differentiable. Validation continues
    # to use the unchanged project's skimage SSIM implementation.
    weights = a.new_tensor([.299, .587, .114])[None, :, None, None]
    a, b = (a * weights).sum(1, keepdim=True), (b * weights).sum(1, keepdim=True)
    mean_a, mean_b = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    var_a = (F.avg_pool2d(a * a, 7, 1) - mean_a.square()).clamp_min(0)
    var_b = (F.avg_pool2d(b * b, 7, 1) - mean_b.square()).clamp_min(0)
    covariance = F.avg_pool2d(a * b, 7, 1) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + .01**2) * (2 * covariance + .03**2)) / (
        (mean_a.square() + mean_b.square() + .01**2) * (var_a + var_b + .03**2))
    return (1 - score).mean()


def laplacian(image):
    # Supervise details at two scales instead of increasing edge contrast blindly.
    outputs = []
    for _ in range(2):
        down = F.avg_pool2d(image, 2)
        outputs.append(image - F.interpolate(down, size=image.shape[-2:], mode='bilinear', align_corners=False))
        image = down
    return outputs


def distance(a, b):
    return sum(F.l1_loss(x, y) for x, y in zip(a, b)) / len(a)


class PatchDiscriminator(nn.Sequential):
    def __init__(self):
        layers = []
        channels = 3
        for out in [32, 64, 128, 128]:
            layers += [spectral_norm(nn.Conv2d(channels, out, 3, 2, 1)), nn.LeakyReLU(.2)]
            channels = out
        layers += [spectral_norm(nn.Conv2d(channels, 1, 3, padding=1))]
        super().__init__(*layers)


@torch.no_grad()
def metrics(n, model, data, vgg):
    model.eval()
    scores = {key: [] for key in ['psnr_y', 'ssim_y', 'pixel_l1', 'vgg_l1', 'laplacian_l1']}
    for low, target in data.batches(32):
        low, target = low.to(n.device), target.to(n.device)
        sr = model(low).clamp(0, 1)
        scores['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
        scores['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
        scores['pixel_l1'].extend((sr - target).abs().flatten(1).mean(1).cpu().tolist())
        scores['vgg_l1'].extend((sum((a - b).abs().flatten(1).mean(1) for a, b in zip(vgg(sr), vgg(target))) / 3).cpu().tolist())
        scores['laplacian_l1'].extend((sum((a - b).abs().flatten(1).mean(1) for a, b in zip(laplacian(sr), laplacian(target))) / 2).cpu().tolist())
    result = {key: float(np.mean(value)) for key, value in scores.items()}
    assert all(math.isfinite(value) for value in result.values())
    return result


def selection(value, baseline):
    # Much tighter reconstruction safeguards than the previous recognizer-led run.
    eligible = value['psnr_y'] >= baseline['psnr_y'] - .20 and value['ssim_y'] >= baseline['ssim_y'] - .003
    score = value['ssim_y'] + .01 * value['psnr_y'] - .05 * value['vgg_l1']
    return eligible, score


def grid(n, models, data, path):
    from PIL import Image, ImageDraw
    low, hr = next(data.batches(8))
    rows = [('Clean original', hr), ('Low-res input', F.interpolate(low, scale_factor=4, mode='nearest'))]
    with torch.inference_mode():
        for name, model in models.items():
            model.eval()
            rows.append((name, model(low.to(n.device)).clamp(0, 1).cpu()))
    canvas = Image.new('RGB', (160 + 8 * 160, len(rows) * 180), '#e8e9e6')
    draw = ImageDraw.Draw(canvas)
    for row, (name, images) in enumerate(rows):
        draw.text((8, row * 180 + 80), name, fill='#141a14')
        for col, image in enumerate(images):
            array = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
            tile = Image.fromarray(array).resize((160, 160), Image.Resampling.NEAREST)
            canvas.paste(tile, (160 + col * 160, row * 180 + 10))
    canvas.save(path)


def train(n, args, vgg, hr, train_ids, val_ids):
    out = ROOT / 'work/visual_sr' / args.name
    out.mkdir(parents=True, exist_ok=True)
    weights = PRESETS[args.mode]
    train_ids = train_ids[:args.train_limit] if args.train_limit else train_ids
    cfg = dataclasses.replace(n.cfg, epochs=args.epochs, batch=args.batch, lr=args.lr, work_dir=str(out))
    manifest = {'mode': args.mode, 'weights': weights, 'epochs': args.epochs, 'batch': args.batch,
        'lr': args.lr, 'train_images': len(train_ids), 'validation_images': args.val_count,
        'base_checkpoint_sha256': sha(BASE), 'source_sha256': sha(__file__),
        'notebook_sha256': sha(ROOT / 'sr_part1_clean.ipynb'),
        'train_ids_sha256': __import__('hashlib').sha256(train_ids.tobytes()).hexdigest(),
        'selection': 'Require PSNR >= original-0.20 dB and SSIM >= original-0.003; maximize SSIM+0.01*PSNR-0.05*VGG.',
        'recognizer_loss_or_selection': False, 'test_used_for_selection': False}
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()) != manifest:
        raise ValueError('Configuration/source changed; use a new run name.')
    write_json(out / 'manifest.json', manifest)
    (out / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    val = n.FixedEvalSet(hr, val_ids[:args.val_count], n.Degrader(), 112, 1000)
    original = n.SRNet(cfg).to(n.device).eval().requires_grad_(False)
    base = torch.load(BASE, map_location='cpu', weights_only=True)
    original.load_state_dict(base['model'])
    model = n.SRNet(cfg).to(n.device)
    model.load_state_dict(base['model'])
    del base
    dataset = n.FaceTrainSet(ROOT / 'work/data/ffhq128_u8.npy', train_ids, n.Degrader(), 112, seed=cfg.seed)
    loader = n.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0,
        pin_memory=n.device == 'cuda', generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=1e-6)
    discriminator = PatchDiscriminator().to(n.device) if weights['adversarial'] else None
    dopt = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.9, .99)) if discriminator else None
    baseline = metrics(n, original, val, vgg)
    write_json(out / 'baseline_validation.json', baseline)
    history = []; best_metrics = baseline; best_epoch = 0
    best_score = selection(baseline, baseline)[1]
    def save(path, epoch):
        state = {'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'epoch': 30 + epoch,
            'pretrain_epochs': 30, 'finetune_epoch': epoch, 'weights': weights, 'manifest': manifest,
            'history': history, 'best_metrics': best_metrics, 'best_epoch': best_epoch, 'best_score': best_score,
            'optimizer': opt.state_dict(), 'scheduler': schedule.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if n.device == 'cuda' else [],
            'discriminator': discriminator.state_dict() if discriminator else None,
            'discriminator_optimizer': dopt.state_dict() if dopt else None}
        temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)
    epoch = 0
    if (out / 'last.pt').exists():
        checkpoint = torch.load(out / 'last.pt', map_location=n.device, weights_only=True)
        assert checkpoint['manifest'] == manifest
        model.load_state_dict(checkpoint['model']); opt.load_state_dict(checkpoint['optimizer']); schedule.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['finetune_epoch']; history = checkpoint['history']; best_metrics = checkpoint['best_metrics']
        best_epoch = checkpoint['best_epoch']; best_score = checkpoint['best_score']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if n.device == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
        if discriminator:
            discriminator.load_state_dict(checkpoint['discriminator']); dopt.load_state_dict(checkpoint['discriminator_optimizer'])
        del checkpoint
    else:
        save(out / 'best.pt', 0)
    print('BASELINE', args.name, json.dumps(baseline), flush=True)
    for ep in range(epoch, args.epochs):
        start = time.monotonic(); sums = {}
        model.train(); dataset.set_epoch(40 + ep); loader.generator.manual_seed(cfg.seed + 40 + ep)
        for low, target in ProgressLoader(loader):
            low, target = low.to(n.device, non_blocking=True), target.to(n.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            raw = model(low); sr = raw.clamp(0, 1)
            parts = {'pixel': F.l1_loss(raw, target), 'structure': structure_loss(sr, target),
                     'laplacian': distance(laplacian(sr), laplacian(target))}
            with torch.no_grad(): features = vgg(target)
            parts['perceptual'] = distance(vgg(sr), features)
            if discriminator:
                discriminator.train().requires_grad_(True); dopt.zero_grad(set_to_none=True)
                dloss = F.softplus(-discriminator(target)).mean() + F.softplus(discriminator(sr.detach())).mean()
                dloss.backward(); nn.utils.clip_grad_norm_(discriminator.parameters(), 1., error_if_nonfinite=True); dopt.step()
                discriminator.eval().requires_grad_(False)
                parts['adversarial'] = F.softplus(-discriminator(sr)).mean()
                sums['discriminator'] = sums.get('discriminator', 0.) + dloss.detach().item()
            total = sum(weights[key] * value for key, value in parts.items())
            if not torch.isfinite(total): raise FloatingPointError('Nonfinite visual SR loss')
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); schedule.step()
            for key, value in parts.items(): sums[key] = sums.get(key, 0.) + value.detach().item()
        value = metrics(n, model, val, vgg)
        eligible, score = selection(value, baseline)
        history.append({'epoch': ep + 1, 'validation': value, 'eligible': eligible,
                        'losses': {k: v / len(loader) for k, v in sums.items()}, 'seconds': time.monotonic() - start})
        if eligible and score > best_score:
            best_metrics, best_epoch, best_score = value, ep + 1, score
            save(out / 'best.pt', ep + 1)
        save(out / 'last.pt', ep + 1)
        write_json(out / 'history.json', history)
        print('VISUAL EPOCH', args.name, json.dumps(history[-1]), flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=n.device, weights_only=True)['model'])
    grid(n, {'Original custom SR': original, 'Candidate custom SR': model}, val, out / 'validation_comparison.png')
    result = {'mode': args.mode, 'baseline': baseline, 'best_validation': best_metrics, 'best_epoch': best_epoch,
              'passes_reconstruction_guards': best_epoch > 0 and selection(best_metrics, baseline)[0],
              'trained_epochs': args.epochs, 'checkpoint': str(out / 'best.pt')}
    write_json(out / 'validation_result.json', result)
    print('VISUAL RESULT', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=PRESETS, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train-limit', type=int, default=8192)
    parser.add_argument('--val-count', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    args = parser.parse_args()
    n, cfg = runtime()
    if n.device != 'cuda': raise RuntimeError('CUDA required for this training run')
    vgg = FrozenFeatures().to(n.device).eval().requires_grad_(False)
    assert structure_loss(torch.ones(1, 3, 20, 20), torch.ones(1, 3, 20, 20)).abs() < 1e-6
    x = torch.rand(1, 3, 20, 20, requires_grad=True)
    structure_loss(x, torch.zeros_like(x)).backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as splits:
        train_ids, val_ids = splits['train'], splits['validation']
    assert not set(train_ids) & set(val_ids)
    train(n, args, vgg, hr, train_ids, val_ids)


if __name__ == '__main__':
    main()

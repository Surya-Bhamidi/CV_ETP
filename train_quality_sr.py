"""Fine-tune on mixed input quality to reduce oversmoothing of cleaner faces.

Uses only the established FFHQ training/validation splits. Never modifies the
archived checkpoints or promotes a candidate automatically.
"""
import argparse
import dataclasses
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from investigate_identity import load
from input_quality import InputDegradation
from refine_model import FrozenFace, FrozenFeatures, sha, write_json
from run_project import ROOT, ProgressLoader, runtime
from train_visual_sr import distance, laplacian, structure_loss


class MixedDegradation:
    def __init__(self, n):
        self.standard = n.Degrader()
        self.mild = n.Degrader(4, blur_p=.2, noise=(0, 2), jpeg=(85, 100))

    def __call__(self, hr, rng):
        choice = rng.random()
        if choice < .5: return self.standard(hr, rng)
        if choice < .75: return self.mild(hr, rng)
        from PIL import Image
        return np.asarray(Image.fromarray(hr).resize((28, 28), Image.Resampling.BICUBIC))


@torch.inference_mode()
def validation(n, model, sets, face):
    result = {}
    for profile, data in sets.items():
        values = {'psnr_y': [], 'ssim_y': [], 'arcface_cos': []}
        for low, target in data.batches(32):
            low, target = low.to(n.device), target.to(n.device)
            sr = model(low).clamp(0, 1)
            values['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
            values['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
            values['arcface_cos'].extend((face(sr) * face(target)).sum(1).cpu().tolist())
        result[profile] = {key: float(np.mean(v)) for key, v in values.items()}
    return result


def selection(value, baseline):
    # Guard the existing difficult-input performance; prefer balanced structure
    # across all three profiles. ArcFace is used in training, not selection.
    a, b = value['standard28'], baseline['standard28']
    guards = a['psnr_y'] >= b['psnr_y'] - .2 and a['ssim_y'] >= b['ssim_y'] - .003
    score = sum(v['ssim_y'] + .01 * v['psnr_y'] for v in value.values()) / len(value)
    return guards, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='mixed_quality_v1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-count', type=int, default=256)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.val_count < 1: raise ValueError('Positive epoch and validation counts required')
    out = (ROOT / 'work/identity_investigation' / args.name).resolve()
    out.relative_to((ROOT / 'work/identity_investigation').resolve())
    if out.exists(): raise ValueError('Choose a fresh run name to preserve earlier evidence')
    out.mkdir(parents=True)
    n, cfg = runtime()
    if n.device != 'cuda': raise RuntimeError('CUDA required for this training run')
    base = ROOT / 'work/visual_sr/full_reconstruction/best.pt'
    model = load(n, base).requires_grad_(True)
    face = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    vgg = FrozenFeatures().to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split:
        train, val, test = split['train'], split['validation'][:args.val_count], split['test']
    if args.train_limit: train = train[:args.train_limit]
    assert not set(train) & (set(val) | set(test))
    sets = {profile: n.FixedEvalSet(hr, val, InputDegradation(n, profile), 112, 1000)
            for profile in ['standard28', 'mild28', 'clean28']}
    weights = {'pixel': 1., 'structure': .2, 'perceptual': .03, 'laplacian': .1, 'identity': .01}
    manifest = {'base_checkpoint': base.relative_to(ROOT).as_posix(), 'base_sha256': sha(base),
        'source_sha256': sha(__file__), 'seed': cfg.seed, 'epochs': args.epochs, 'batch': args.batch,
        'learning_rate': 2e-5, 'train_images': len(train), 'validation_images_per_profile': len(val),
        'train_ids_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
        'profiles': {'standard': .5, 'mild': .25, 'clean_bicubic': .25}, 'weights': weights,
        'identity_prior': 'Frozen ArcFace w600k_r50, first 8 samples/batch; no identity labels',
        'selection': 'Mean SSIM+0.01*PSNR over 3 profiles, standard PSNR >= base-0.2 and SSIM >= base-0.003',
        'test_used_for_training_or_selection': False}
    write_json(out / 'manifest.json', manifest)
    (out / 'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline = validation(n, model, sets, face)
    write_json(out / 'baseline_validation.json', baseline)
    best_score, best_epoch, best_metrics = selection(baseline, baseline)[1], 0, baseline
    dataset = n.FaceTrainSet(ROOT / 'work/data/ffhq128_u8.npy', train, MixedDegradation(n), 112, seed=cfg.seed)
    loader = n.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0,
                          pin_memory=True, generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=1e-6)
    history = []
    def save(path, epoch):
        state = {'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'epoch': 35 + epoch,
            'finetune_epoch': 5 + epoch, 'quality_finetune_epoch': epoch, 'manifest': manifest,
            'optimizer': opt.state_dict(), 'scheduler': schedule.state_dict(), 'history': history,
            'best_epoch': best_epoch, 'best_validation': best_metrics,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)
    save(out / 'best.pt', 0)
    print('BASELINE', json.dumps(baseline), flush=True)
    for epoch in range(args.epochs):
        start = time.monotonic(); sums = {}
        model.train(); dataset.set_epoch(60 + epoch); loader.generator.manual_seed(cfg.seed + 60 + epoch)
        for low, target in ProgressLoader(loader):
            low, target = low.to(n.device), target.to(n.device)
            opt.zero_grad(set_to_none=True)
            raw = model(low); sr = raw.clamp(0, 1)
            with torch.no_grad():
                target_features = vgg(target[:16]); target_identity = face(target[:8])
            parts = {'pixel': F.l1_loss(raw, target), 'structure': structure_loss(sr, target),
                     'laplacian': distance(laplacian(sr), laplacian(target)),
                     'perceptual': distance(vgg(sr[:16]), target_features),
                     'identity': 1 - (face(sr[:8]) * target_identity).sum(1).mean()}
            loss = sum(weights[k] * v for k, v in parts.items())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); schedule.step()
            for k, v in parts.items(): sums[k] = sums.get(k, 0) + v.detach().item()
        model.eval()
        value = validation(n, model, sets, face)
        eligible, score = selection(value, baseline)
        history.append({'epoch': epoch + 1, 'seconds': time.monotonic() - start, 'validation': value,
                        'eligible': eligible, 'losses': {k: v / len(loader) for k, v in sums.items()}})
        if eligible and score > best_score:
            best_score, best_epoch, best_metrics = score, epoch + 1, value
            save(out / 'best.pt', epoch + 1)
        save(out / 'last.pt', epoch + 1); write_json(out / 'history.json', history)
        print('QUALITY EPOCH', json.dumps(history[-1]), flush=True)
    write_json(out / 'validation_result.json', {'best_epoch': best_epoch, 'best_validation': best_metrics,
               'baseline': baseline, 'epochs_completed': args.epochs, 'eligible': best_epoch > 0})
    print('COMPLETE best epoch', best_epoch, flush=True)


if __name__ == '__main__':
    main()

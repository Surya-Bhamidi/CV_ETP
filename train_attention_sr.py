"""Controlled attention-only fine-tuning. No test or QMUL images enter training.

All variants use identical FFHQ examples, augmentations, epochs and losses.
Only attention weights train; SSC, head, tail, FFN and upsampler stay frozen.
"""
import argparse
import dataclasses
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import load
from refine_model import FrozenFace, FrozenFeatures, sha, write_json
from run_project import ROOT, ProgressLoader, runtime
from sparse_attention import install_attention
from train_quality_sr import MixedDegradation, selection, validation
from train_visual_sr import distance, laplacian, structure_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['control', 'context_global', 'context_quadrant'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train-limit', type=int, default=8192)
    parser.add_argument('--val-count', type=int, default=256)
    args = parser.parse_args()
    out = (ROOT / 'work/attention_investigation' / args.name).resolve()
    out.relative_to((ROOT / 'work/attention_investigation').resolve())
    if out.exists(): raise ValueError('Use a new run name to preserve previous evidence')
    out.mkdir(parents=True)
    n, cfg = runtime()
    if n.device != 'cuda': raise RuntimeError('CUDA is required for this experiment')
    base = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'
    model = load(n, base)
    spec = {'policy': 'quadrant' if args.variant == 'context_quadrant' else 'global',
            'keep_ratio': .5, 'backend': 'gather', 'spatial_context': args.variant != 'control'}
    install_attention(model, **spec)
    model.requires_grad_(False)
    for block in model.body: block.attn.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    face = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    vgg = FrozenFeatures().to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split:
        train_ids, val_ids = split['train'][:args.train_limit or None], split['validation'][:args.val_count]
        assert not set(train_ids) & (set(val_ids) | set(split['test']))
    sets = {p: n.FixedEvalSet(hr, val_ids, InputDegradation(n, p), 112, 1000) for p in ['standard28', 'mild28', 'clean28']}
    weights = {'pixel': 1., 'structure': .2, 'perceptual': .03, 'laplacian': .1, 'identity': .01}
    manifest = {'variant': args.variant, 'attention_spec': spec, 'base_checkpoint_sha256': sha(base),
                'source_sha256': sha(__file__), 'attention_source_sha256': sha(ROOT / 'sparse_attention.py'),
                'epochs': args.epochs, 'train_images': len(train_ids), 'validation_images_per_profile': len(val_ids),
                'trainable_parameters': sum(p.numel() for p in trainable), 'parameters': n.n_params(model),
                'frozen_components': 'SSC, FFN, head, tail, upsampler and all norms',
                'lr': 3e-5, 'batch': 64, 'seed': 42, 'augmentation_epoch_start': 70, 'loss_weights': weights,
                'test_or_qmul_used_for_training_or_selection': False}
    write_json(out / 'manifest.json', manifest)
    (out / 'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline = validation(n, model, sets, face)
    best_metrics, best_epoch = baseline, 0
    best_score = selection(baseline, baseline)[1]
    dataset = n.FaceTrainSet(ROOT / 'work/data/ffhq128_u8.npy', train_ids, MixedDegradation(n), 112, seed=42)
    loader = n.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
                          generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(trainable, lr=3e-5, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=1e-6)
    history = []
    def save(path, epoch):
        state = {'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'attention_spec': spec,
                 'epoch': 38 + epoch, 'finetune_epoch': 8 + epoch, 'manifest': manifest,
                 'history': history, 'best_epoch': best_epoch, 'best_validation': best_metrics}
        temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)
    save(out / 'best.pt', 0)
    print('ATTENTION BASELINE', args.variant, json.dumps(baseline), flush=True)
    for epoch in range(args.epochs):
        start = time.monotonic(); model.train(); sums = {}
        dataset.set_epoch(70 + epoch); loader.generator.manual_seed(112 + epoch)
        for low, target in ProgressLoader(loader):
            low, target = low.to(n.device), target.to(n.device)
            opt.zero_grad(set_to_none=True)
            raw = model(low); sr = raw.clamp(0, 1)
            with torch.no_grad(): features, identity = vgg(target[:16]), face(target[:8])
            parts = {'pixel': F.l1_loss(raw, target), 'structure': structure_loss(sr, target),
                     'laplacian': distance(laplacian(sr), laplacian(target)),
                     'perceptual': distance(vgg(sr[:16]), features),
                     'identity': 1 - (face(sr[:8]) * identity).sum(1).mean()}
            loss = sum(weights[key] * value for key, value in parts.items())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            opt.step(); schedule.step()
            for key, value in parts.items(): sums[key] = sums.get(key, 0.) + value.detach().item()
        model.eval(); value = validation(n, model, sets, face)
        eligible, score = selection(value, baseline)
        history.append({'epoch': epoch + 1, 'seconds': time.monotonic() - start, 'validation': value,
                        'eligible': eligible, 'losses': {key: value / len(loader) for key, value in sums.items()}})
        if eligible and score > best_score:
            best_score, best_epoch, best_metrics = score, epoch + 1, value; save(out / 'best.pt', epoch + 1)
        save(out / 'last.pt', epoch + 1); write_json(out / 'history.json', history)
        print('ATTENTION EPOCH', args.variant, json.dumps(history[-1]), flush=True)
    write_json(out / 'validation_result.json', {'baseline': baseline, 'best_validation': best_metrics,
                'best_epoch': best_epoch, 'epochs_completed': args.epochs, 'best_score': best_score})
    print('ATTENTION COMPLETE', args.variant, 'best epoch', best_epoch, flush=True)


if __name__ == '__main__': main()

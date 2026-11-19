"""Identity-disjoint QMUL training-domain adaptation and matched removal pilots.

QMUL targets are native LR observations enlarged to 112, NOT clean HR targets.
All test images and identities are excluded. MobileFaceNet is not used for loss
or checkpoint selection. The architecture pilots share retained pretrained
weights, data, schedule and loss; these are fine-tuning, not from-scratch ablations.
"""
import argparse
import dataclasses
import io
import json
import time
import zipfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from input_quality import InputDegradation
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, ProgressLoader, runtime
from train_quality_sr import MixedDegradation
from train_visual_sr import structure_loss

BASE = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'
OUT = ROOT / 'work/qmul_domain'


class CameraSet:
    handles = {}
    def __init__(self, archive, files): self.archive, self.files, self.zipped = archive, files, None
    def pair(self, index):
        if self.zipped is None:
            key = str(self.archive)
            if key not in self.handles: self.handles[key] = zipfile.ZipFile(self.archive)
            self.zipped = self.handles[key]
        with Image.open(io.BytesIO(self.zipped.read(self.files[index]))) as image:
            image = image.convert('RGB')
            return np.array(image.resize((28, 28), Image.Resampling.BICUBIC), copy=True), np.array(image.resize((112, 112), Image.Resampling.BICUBIC), copy=True)


class JointSet(torch.utils.data.Dataset):
    def __init__(self, n, ff_ids, camera, control=False):
        self.ffhq = n.FaceTrainSet(ROOT / 'work/data/ffhq128_u8.npy', ff_ids[:16384], MixedDegradation(n), 112, seed=42)
        self.other = n.FaceTrainSet(ROOT / 'work/data/ffhq128_u8.npy', ff_ids[16384:32768], MixedDegradation(n), 112, seed=42) if control else None
        self.camera, self.epoch = camera, 0
    def __len__(self): return 16384
    def set_epoch(self, epoch):
        self.epoch = epoch; self.ffhq.set_epoch(epoch)
        if self.other is not None: self.other.set_epoch(epoch)
    def __getitem__(self, i):
        low, clean = self.ffhq[i]
        if self.other is not None: return low, clean, *self.other[i]
        camera_low, observed = self.camera.pair(i)
        if np.random.default_rng(np.random.SeedSequence([42, self.epoch, i])).random() < .5:
            camera_low, observed = camera_low[:, ::-1].copy(), observed[:, ::-1].copy()
        tensor = lambda a: torch.from_numpy(a).permute(2, 0, 1).float() / 255
        return low, clean, tensor(camera_low), tensor(observed)


@torch.inference_mode()
def validation(n, model, face, ff_sets, camera):
    model.eval(); result = {'ffhq': {}}
    for profile, data in ff_sets.items():
        psnr, ssim = [], []
        for low, target in data.batches(32):
            target = target.to(n.device); sr = model(low.to(n.device)).clamp(0, 1)
            psnr.extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
            ssim.extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
        result['ffhq'][profile] = {'psnr_y': float(np.mean(psnr)), 'ssim_y': float(np.mean(ssim))}
    embeddings = []
    for offset in range(0, len(camera.files), 32):
        values = [camera.pair(i)[0] for i in range(offset, min(offset + 32, len(camera.files)))]
        low = torch.from_numpy(np.stack(values)).permute(0, 3, 1, 2).float().to(n.device) / 255
        embeddings.append(face(model(low).clamp(0, 1)).cpu().numpy())
    embedding = np.concatenate(embeddings)
    a, b = embedding[0::2], embedding[1::2]
    scores = np.r_[(a * b).sum(1), (a * np.roll(b, -1, axis=0)).sum(1)]
    labels = np.r_[np.ones(len(a)), np.zeros(len(a))]
    result['domain_validation_arcface_auc'] = float(roc_auc_score(labels, scores))
    result['domain_validation_pairs'] = len(labels)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['domain_full', 'domain_ffhq_control', 'domain_no_attention', 'domain_no_ssc'], required=True)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    out = OUT / args.variant
    if out.exists(): raise ValueError('Run already exists; preserve evidence rather than overwrite')
    out.mkdir(parents=True)
    n, cfg = runtime()
    if n.device != 'cuda': raise RuntimeError('CUDA required for this controlled experiment')
    data_manifest = json.loads((OUT / 'data_manifest.json').read_text())
    architecture = {'use_ssc': args.variant != 'domain_no_ssc', 'use_attn': args.variant != 'domain_no_attention'}
    model = n.SRNet(cfg, **architecture).to(n.device)
    original = torch.load(BASE, map_location='cpu', weights_only=True)
    shared = {key: value for key, value in original['model'].items() if key in model.state_dict()}
    model.load_state_dict(shared, strict=True); del original, shared
    model.requires_grad_(True)
    face = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split:
        ff_ids, val_ids = split['train'][:32768], split['validation'][:128]
    ff_sets = {p: n.FixedEvalSet(hr, val_ids, InputDegradation(n, p), 112, 1000) for p in ['standard28', 'clean28']}
    archive = ROOT / data_manifest['source_archive']
    camera = CameraSet(archive, data_manifest['train_files'][:16384])
    camera_val = CameraSet(archive, data_manifest['validation_files'])
    control = args.variant == 'domain_ffhq_control'
    dataset = JointSet(n, ff_ids, camera, control)
    loader = n.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
                          generator=torch.Generator().manual_seed(42))
    manifest = {'variant': args.variant, 'architecture': architecture, 'base_checkpoint_sha256': sha(BASE),
        'source_sha256': sha(__file__), 'data_manifest_sha256': sha(OUT / 'data_manifest.json'),
        'epochs': args.epochs, 'updates_per_epoch': len(loader), 'lr': 2e-5,
        'ffhq_images': 32768 if control else 16384, 'qmul_training_images': 0 if control else 16384,
        'domain_validation_identities': 256, 'domain_validation_images': 512,
        'domain_targets_are_clean_hr': False, 'test_used_for_training_or_selection': False,
        'parameters': n.n_params(model), 'weights': {'first_branch_pixel': 1., 'first_branch_structure': .2, 'second_branch_observation_pixel': .25, 'second_branch_arcface': .02},
        'second_branch_data': 'FFHQ clean targets' if control else 'QMUL native LR observations enlarged to112',
        'selection': 'Max official-training heldout-identity ArcFace AUC; FFHQ standard/clean PSNR >= mixed baseline-0.5 dB, SSIM >= mixed baseline-0.01. If all fail, evaluate final epoch as a failed candidate.',
        'ablation_scope': 'Two-epoch fine-tuning pilots with identical retained pretrained weights; not independent from-scratch training'}
    write_json(out / 'manifest.json', manifest); (out / 'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline = validation(n, model, face, ff_sets, camera_val)
    # The guards reference the unchanged full mixed model, including removal pilots.
    full_reference = json.loads((OUT / 'domain_full/baseline_validation.json').read_text()) if args.variant != 'domain_full' else baseline
    write_json(out / 'baseline_validation.json', baseline)
    def eligible(value):
        return all(value['ffhq'][p]['psnr_y'] >= full_reference['ffhq'][p]['psnr_y'] - .5 and
                   value['ffhq'][p]['ssim_y'] >= full_reference['ffhq'][p]['ssim_y'] - .01 for p in ff_sets)
    best_score = baseline['domain_validation_arcface_auc'] if eligible(baseline) else -float('inf')
    best_epoch, history = 0, []
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=1e-6)
    def save(path, epoch):
        state = {'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'architecture': architecture,
                 'epoch': 38 + epoch, 'manifest': manifest, 'history': history, 'best_epoch': best_epoch}
        temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)
    save(out / 'best.pt', 0)
    print('DOMAIN BASELINE', args.variant, json.dumps(baseline), flush=True)
    for epoch in range(args.epochs):
        start = time.monotonic(); model.train(); sums = {}
        dataset.set_epoch(90 + epoch); loader.generator.manual_seed(132 + epoch)
        for low, target, camera_low, observed in ProgressLoader(loader):
            low, target, camera_low, observed = [value.to(n.device) for value in [low, target, camera_low, observed]]
            opt.zero_grad(set_to_none=True)
            raw = model(torch.cat([low, camera_low])); sr = raw.clamp(0, 1)
            with torch.no_grad(): teacher = face(observed[:16])
            parts = {'pixel': F.l1_loss(raw[:32], target), 'structure': .2 * structure_loss(sr[:32], target),
                     'second_observation': .25 * F.l1_loss(raw[32:], observed),
                     'second_identity': .02 * (1 - (face(sr[32:48]) * teacher).sum(1).mean())}
            loss = sum(parts.values())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite domain loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); scheduler.step()
            for key, value in parts.items(): sums[key] = sums.get(key, 0) + value.detach().item()
        model.eval(); value = validation(n, model, face, ff_sets, camera_val)
        valid = eligible(value)
        history.append({'epoch': epoch + 1, 'validation': value, 'eligible': valid, 'seconds': time.monotonic() - start,
                        'losses': {key: value / len(loader) for key, value in sums.items()}})
        if valid and value['domain_validation_arcface_auc'] > best_score:
            best_score, best_epoch = value['domain_validation_arcface_auc'], epoch + 1; save(out / 'best.pt', epoch + 1)
        save(out / 'last.pt', epoch + 1); write_json(out / 'history.json', history)
        print('DOMAIN EPOCH', args.variant, json.dumps(history[-1]), flush=True)
    if best_score == -float('inf'):
        best_epoch = args.epochs; save(out / 'best.pt', args.epochs)
    write_json(out / 'validation_result.json', {'baseline': baseline, 'history': history, 'best_epoch': best_epoch,
        'passed_ffhq_guards': best_score != -float('inf'), 'selected_using_test': False})
    print('DOMAIN TRAINING COMPLETE', args.variant, 'selected', best_epoch, flush=True)


if __name__ == '__main__': main()

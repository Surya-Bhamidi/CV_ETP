"""Camera adaptation with explicit texture and identity preservation on paired FFHQ.

Separate exploratory refinement of the completed matched ablation study. The
loss differs from that study, so this run is not another matched removal row.
QMUL verification images and MobileFaceNet never enter training or selection.
"""
import dataclasses
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import load
from quality_metric import load_lpips
from refine_model import FrozenFace, FrozenFeatures, sha, write_json
from run_project import ROOT, ProgressLoader, runtime
from train_qmul_domain import CameraSet, JointSet, validation
from train_visual_sr import distance, laplacian, structure_loss

OUT = ROOT / 'work/qmul_quality/balanced_v1'
BASE = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'


@torch.inference_mode()
def assess(n, model, face, sets, camera, perceptual):
    value = validation(n, model, face, sets, camera)
    for profile, data in sets.items():
        errors = []
        for low, target in data.batches(32):
            sr = model(low.to(n.device)).clamp(0, 1)
            errors.extend(perceptual(sr, target.to(n.device), normalize=True).flatten().cpu().tolist())
        value['ffhq'][profile]['lpips'] = float(np.mean(errors))
    return value


def main():
    if OUT.exists(): raise ValueError('Preserve existing run; choose a new name')
    OUT.mkdir(parents=True)
    n, cfg = runtime()
    torch.manual_seed(42)
    model = load(n, BASE).requires_grad_(True)
    face = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    vgg = FrozenFeatures().to(n.device).eval().requires_grad_(False)
    perceptual = load_lpips(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split:
        train, val = split['train'][:32768], split['validation'][:128]
    sets = {p: n.FixedEvalSet(hr, val, InputDegradation(n, p), 112, 1000) for p in ['standard28', 'clean28']}
    data = json.loads((ROOT / 'work/qmul_domain/data_manifest.json').read_text())
    archive = ROOT / data['source_archive']
    camera = CameraSet(archive, data['train_files'][:16384])
    camera_val = CameraSet(archive, data['validation_files'])
    dataset = JointSet(n, train, camera)
    loader = n.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0,
                          pin_memory=True, generator=torch.Generator().manual_seed(42))
    weights = {'pixel': 1., 'structure': .2, 'laplacian': .1, 'vgg': .03,
               'lpips': .05, 'ffhq_identity': .01, 'camera_observation': .1, 'camera_identity': .02}
    manifest = {'base_checkpoint_sha256': sha(BASE), 'source_sha256': sha(__file__),
                'data_manifest_sha256': sha(ROOT / 'work/qmul_domain/data_manifest.json'),
                'seed': 42, 'epochs': 2, 'updates_per_epoch': len(loader), 'learning_rate': 2e-5,
                'ffhq_images': 16384, 'qmul_training_images': 16384, 'weights': weights,
                'parameters': n.n_params(model), 'architecture': {'use_ssc': True, 'use_attn': True},
                'selection': 'Max official-training heldout-identity ArcFace AUC, with BOTH FFHQ validation LPIPS <= mixed baseline, PSNR >= baseline-0.10 dB, SSIM >= baseline-0.0015. Baseline retained if no eligible improvement.',
                'camera_targets_are_clean_hr': False, 'test_used_for_training_or_selection': False,
                'scope': 'Exploratory perceptual refinement; different loss from the matched ablation study'}
    write_json(OUT / 'manifest.json', manifest)
    (OUT / 'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline = assess(n, model, face, sets, camera_val, perceptual)
    write_json(OUT / 'baseline_validation.json', baseline)
    history, best_epoch = [], 0
    best_score = baseline['domain_validation_arcface_auc']
    def save(path, epoch):
        torch.save({'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'epoch': 38 + epoch,
                    'architecture': manifest['architecture'], 'manifest': manifest, 'history': history,
                    'best_epoch': best_epoch}, path)
    save(OUT / 'best.pt', 0)
    print('BALANCED BASELINE', json.dumps(baseline), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2 * len(loader), eta_min=1e-6)
    for epoch in range(2):
        start = time.monotonic(); model.train(); sums = {}
        dataset.set_epoch(90 + epoch); loader.generator.manual_seed(132 + epoch)
        for low, target, camera_low, observed in ProgressLoader(loader):
            low, target, camera_low, observed = [x.to(n.device) for x in [low, target, camera_low, observed]]
            opt.zero_grad(set_to_none=True)
            raw = model(torch.cat([low, camera_low])); sr = raw.clamp(0, 1)
            with torch.no_grad():
                target_vgg = vgg(target[:8]); ff_face = face(target[:8]); cam_face = face(observed[:8])
            parts = {'pixel': F.l1_loss(raw[:32], target), 'structure': structure_loss(sr[:32], target),
                     'laplacian': distance(laplacian(sr[:32]), laplacian(target)),
                     'vgg': distance(vgg(sr[:8]), target_vgg),
                     'lpips': perceptual(sr[:16], target[:16], normalize=True).mean(),
                     'ffhq_identity': 1 - (face(sr[:8]) * ff_face).sum(1).mean(),
                     'camera_observation': F.l1_loss(raw[32:], observed),
                     'camera_identity': 1 - (face(sr[32:40]) * cam_face).sum(1).mean()}
            loss = sum(weights[k] * v for k, v in parts.items())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite balanced loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); scheduler.step()
            for k, v in parts.items(): sums[k] = sums.get(k, 0) + v.detach().item()
        model.eval(); value = assess(n, model, face, sets, camera_val, perceptual)
        eligible = all(value['ffhq'][p]['lpips'] <= baseline['ffhq'][p]['lpips'] and
                       value['ffhq'][p]['psnr_y'] >= baseline['ffhq'][p]['psnr_y'] - .1 and
                       value['ffhq'][p]['ssim_y'] >= baseline['ffhq'][p]['ssim_y'] - .0015 for p in sets)
        history.append({'epoch': epoch + 1, 'seconds': time.monotonic() - start, 'validation': value,
                        'eligible': eligible, 'losses': {k: v / len(loader) for k, v in sums.items()}})
        if eligible and value['domain_validation_arcface_auc'] > best_score:
            best_epoch, best_score = epoch + 1, value['domain_validation_arcface_auc']
            save(OUT / 'best.pt', epoch + 1)
        save(OUT / 'last.pt', epoch + 1); write_json(OUT / 'history.json', history)
        print('BALANCED EPOCH', json.dumps(history[-1]), flush=True)
    write_json(OUT / 'validation_result.json', {'baseline': baseline, 'history': history,
        'selected_epoch': best_epoch, 'selected_new_checkpoint': best_epoch > 0,
        'test_used_for_selection': False})
    print('BALANCED COMPLETE selected', best_epoch, flush=True)


if __name__ == '__main__': main()

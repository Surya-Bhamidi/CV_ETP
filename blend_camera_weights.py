"""Validation-only weight interpolation after the texture pilot's strict guards.

This is a separate, explicitly exploratory selection step. It does not relabel
an ineligible training epoch as passing. It changes no architecture/parameter
count and needs one inference pass. No verification-test images are consulted.
"""
import dataclasses
import json

import numpy as np
import torch

from input_quality import InputDegradation
from quality_metric import load_lpips
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime
from train_balanced_camera import BASE, OUT as TRAIN, assess
from train_qmul_domain import CameraSet

OUT = ROOT / 'work/qmul_quality/balanced_merge_v1'


def main():
    if OUT.exists(): raise ValueError('Preserve recorded selection; use a new run name')
    OUT.mkdir(parents=True)
    n, cfg = runtime()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    base = torch.load(BASE, map_location='cpu', weights_only=True)
    candidate = torch.load(TRAIN / 'last.pt', map_location='cpu', weights_only=True)
    manifest = dict(candidate['manifest'])
    manifest.update(source_sha256=sha(__file__), training_source_sha256=candidate['manifest']['source_sha256'],
        texture_checkpoint_sha256=sha(TRAIN / 'last.pt'), weight_coefficients=[.25, .5, .75, 1.],
        selection='Exploratory validation-only weight interpolation. Max camera training-validation ArcFace AUC with BOTH FFHQ LPIPS <= original mixed baseline, PSNR >= baseline-0.10 dB, SSIM >= baseline-0.0015. Baseline retained if no eligible gain.',
        scope='Weight interpolation after strict per-epoch guards; not a matched architecture ablation')
    write_json(OUT / 'manifest.json', manifest)
    (OUT / 'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    model = n.SRNet(cfg).to(n.device).eval().requires_grad_(False)
    face = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    lpips = load_lpips(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split: ids = split['validation'][:128]
    sets = {p: n.FixedEvalSet(hr, ids, InputDegradation(n, p), 112, 1000) for p in ['standard28', 'clean28']}
    data = json.loads((ROOT / 'work/qmul_domain/data_manifest.json').read_text())
    camera = CameraSet(ROOT / data['source_archive'], data['validation_files'])
    model.load_state_dict(base['model']); baseline = assess(n, model, face, sets, camera, lpips)
    write_json(OUT / 'baseline_validation.json', baseline)
    best_score, best_alpha, history = baseline['domain_validation_arcface_auc'], 0., []
    torch.save(base, OUT / 'best.pt')
    for alpha in [.25, .5, .75, 1.]:
        state = {key: value.lerp(candidate['model'][key], alpha) if value.is_floating_point() else value
                 for key, value in base['model'].items()}
        model.load_state_dict(state, strict=True)
        value = assess(n, model, face, sets, camera, lpips)
        eligible = all(value['ffhq'][p]['lpips'] <= baseline['ffhq'][p]['lpips'] and
            value['ffhq'][p]['psnr_y'] >= baseline['ffhq'][p]['psnr_y'] - .1 and
            value['ffhq'][p]['ssim_y'] >= baseline['ffhq'][p]['ssim_y'] - .0015 for p in sets)
        history.append({'texture_weight': alpha, 'validation': value, 'eligible': eligible})
        if eligible and value['domain_validation_arcface_auc'] > best_score:
            best_score, best_alpha = value['domain_validation_arcface_auc'], alpha
            torch.save({'model': state, 'config': dataclasses.asdict(cfg), 'epoch': candidate['epoch'],
                'architecture': manifest['architecture'], 'manifest': manifest, 'texture_weight': alpha}, OUT / 'best.pt')
        write_json(OUT / 'history.json', history)
        print('WEIGHT VALIDATION', json.dumps(history[-1]), flush=True)
    write_json(OUT / 'validation_result.json', {'baseline': baseline, 'history': history,
        'selected_epoch': candidate['epoch'] - 38, 'selected_texture_weight': best_alpha,
        'selected_new_checkpoint': best_alpha > 0, 'test_used_for_selection': False})
    print('WEIGHT SELECTION COMPLETE', best_alpha, flush=True)


if __name__ == '__main__': main()

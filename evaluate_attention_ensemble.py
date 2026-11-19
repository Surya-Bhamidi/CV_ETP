"""Validation-first test of image-only self-ensemble on the audited attention model."""
import json

import numpy as np
import torch

from audit_attention import BASE, OUT, validation
from input_quality import InputDegradation
from investigate_identity import load, save_grid
from refine_model import FrozenFace, write_json
from run_project import ROOT, runtime
from sparse_attention import SelfEnsemble


@torch.inference_mode()
def main():
    n, _ = runtime(); torch.backends.cudnn.benchmark = False
    model = load(n, BASE)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split:
        val_ids, test_ids = split['validation'][:256], split['test'][:512]
    profiles = ['standard28', 'mild28', 'clean28']
    sets = {p: n.FixedEvalSet(hr, val_ids, InputDegradation(n, p), 112, 1000) for p in profiles}
    methods = {'single': model, 'flip2': SelfEnsemble(model, 2), 'dihedral8': SelfEnsemble(model, 8)}
    results = {'validation_images_per_profile': 256, 'test_images_per_profile': 512, 'validation': {},
               'selection': 'Max mean SSIM across 3 profiles; each PSNR >= single-0.05 dB, each SSIM >= single; selected before test'}
    for name, candidate in methods.items():
        results['validation'][name] = validation(n, candidate, sets)
        print('ENSEMBLE VALIDATION', name, json.dumps(results['validation'][name]), flush=True)
    baseline = results['validation']['single']
    eligible = [name for name, values in results['validation'].items() if all(
        values[p]['ssim_y'] >= baseline[p]['ssim_y'] and values[p]['psnr_y'] >= baseline[p]['psnr_y'] - .05 for p in profiles)]
    selected = max(eligible, key=lambda name: sum(results['validation'][name][p]['ssim_y'] for p in profiles))
    results['selected'] = selected
    write_json(OUT / 'ensemble_results.json', results)
    if selected == 'single':
        print('ENSEMBLE: no candidate passed; no test tuning performed', flush=True); return
    face = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    results['test'] = {}
    for profile in profiles:
        data = n.FixedEvalSet(hr, test_ids, InputDegradation(n, profile), 112, 2000)
        scores, rows = {}, None
        for low, target in data.batches(32):
            low, target = low.to(n.device), target.to(n.device)
            clean = face(target)
            views = {'single': model(low).clamp(0, 1), selected: methods[selected](low)}
            for name, sr in views.items():
                entry = scores.setdefault(name, {'psnr_y': [], 'ssim_y': [], 'mobileface_cos': []})
                entry['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
                entry['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
                entry['mobileface_cos'].extend((face(sr) * clean).sum(1).cpu().tolist())
            if rows is None: rows = {'Clean original': target.cpu(), 'Actual input (28)': low.cpu(), **{k: v.cpu() for k, v in views.items()}}
        results['test'][profile] = {name: {key: float(np.mean(v)) for key, v in values.items()} for name, values in scores.items()}
        write_json(OUT / f'ensemble_test_{profile}_per_image.json', scores)
        save_grid(rows, OUT / f'ensemble_test_{profile}.png', f'TEST | {profile} | selected ensemble versus same model, one pass')
        write_json(OUT / 'ensemble_results.json', results)
        print('ENSEMBLE TEST', profile, json.dumps(results['test'][profile]), flush=True)
    results['heldout_identity_guard'] = all(results['test'][p][selected]['mobileface_cos'] >= results['test'][p]['single']['mobileface_cos'] - .005 for p in profiles)
    write_json(OUT / 'ensemble_results.json', results)


if __name__ == '__main__': main()

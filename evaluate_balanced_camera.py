"""Evaluate the validation-selected balanced run; never select epochs on test."""
import json

import numpy as np
import torch

import benchmark_qmul as benchmark
from input_quality import InputDegradation
from investigate_identity import load, save_grid
from quality_metric import load_lpips
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime
from train_balanced_camera import BASE
from blend_camera_weights import OUT


@torch.inference_mode()
def main():
    selection = json.loads((OUT / 'validation_result.json').read_text())
    if not selection['selected_new_checkpoint']:
        print('No candidate passed validation; retain earlier models.', flush=True)
        return
    n, _ = runtime()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    path = OUT / 'best.pt'
    original_factory = benchmark.factory
    def factory(namespace, name):
        if name != 'balanced_camera': return original_factory(namespace, name)
        model = load(namespace, path)
        meta = {'method': name, 'role': 'project-trained exploratory perceptual refinement',
                'input': 'same cached LR28 RGB pixels', 'output': '112x112 RGB',
                'checkpoint': path.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(path),
                'parameters': namespace.n_params(model),
                'training_manifest': json.loads((OUT / 'manifest.json').read_text()),
                'not_a_matched_architecture_ablation': True}
        return model, meta
    benchmark.factory = factory
    r, manifest = benchmark.prepare()
    faces = {'arcface': FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False),
             'mobileface': FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)}
    benchmark.run(n, r, manifest, 'balanced_camera', faces, 64)
    del faces; torch.cuda.empty_cache()
    models = {'previous_mixed': load(n, BASE), 'camera_raw': load(n, ROOT / 'work/qmul_domain/domain_full/best.pt'),
              'balanced_camera': load(n, path)}
    lpips = load_lpips(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as splits: test = splits['test'][:512]
    result = {'images_per_profile': 512, 'test_used_for_selection': False, 'profiles': {},
              'checkpoint_sha256': sha(path), 'source_sha256': sha(__file__)}
    for profile in ['standard28', 'clean28']:
        data = n.FixedEvalSet(hr, test, InputDegradation(n, profile), 112, 2000)
        scores = {name: {key: [] for key in ['psnr_y', 'ssim_y', 'lpips']} for name in models}
        rows = None
        for low, target in data.batches(32):
            low, target = low.to(n.device), target.to(n.device)
            views = {name: model(low).clamp(0, 1) for name, model in models.items()}
            for name, sr in views.items():
                scores[name]['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
                scores[name]['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
                scores[name]['lpips'].extend(lpips(sr, target, normalize=True).flatten().cpu().tolist())
            if rows is None: rows = {'Clean original': target.cpu(), 'Actual input (28)': low.cpu(), **{k: v.cpu() for k, v in views.items()}}
        result['profiles'][profile] = {name: {key: float(np.mean(values)) for key, values in entries.items()} for name, entries in scores.items()}
        write_json(OUT / (profile + '_per_image.json'), scores)
        save_grid(rows, OUT / (profile + '_comparison.png'), 'HELD-OUT TEST | ' + profile + ' | validation-selected balanced camera model')
        write_json(OUT / 'paired_test.json', result)
        print('BALANCED PAIRED TEST', profile, json.dumps(result['profiles'][profile]), flush=True)
    qmul = json.loads((benchmark.OUT / 'results/balanced_camera.json').read_text())
    before = json.loads((benchmark.OUT / 'results/ours_mixed.json').read_text())
    checks = {'validation_selected_new_epoch': selection['selected_new_checkpoint'],
              'independent_mobileface_auc_improves': qmul['evaluators']['mobileface']['auc'] >= before['evaluators']['mobileface']['auc'] + .003,
              'independent_mobileface_accuracy_not_worse': qmul['evaluators']['mobileface']['acc'] >= before['evaluators']['mobileface']['acc'],
              'test_texture_and_structure_guards': all(v['balanced_camera']['lpips'] <= v['previous_mixed']['lpips'] and
                  v['balanced_camera']['ssim_y'] >= v['previous_mixed']['ssim_y'] - .0015 and
                  v['balanced_camera']['psnr_y'] >= v['previous_mixed']['psnr_y'] - .1 for v in result['profiles'].values())}
    review = {'available_for_comparison': all(checks.values()), 'checks': checks,
              'checkpoint': path.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(path),
              'default_model_unchanged': True, 'experimental': True,
              'warning': 'Exploratory camera adaptation with texture-preservation losses. No clean QMUL HR supervision. Sharper detail and verification scores do not certify identity recovery.'}
    write_json(OUT / 'review.json', review)
    print('BALANCED REVIEW', json.dumps(review), flush=True)


if __name__ == '__main__': main()

"""Validation-first trial of repeated SR, self-ensemble and back-projection."""
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from input_quality import InputDegradation
from investigate_identity import load
from quality_metric import load_lpips_alex
from refine_model import FrozenFace, sha, write_json
from restoration_passes import ensemble_predictions, predict, resize
from run_project import ROOT, runtime

OUT = ROOT / 'work/restoration_passes_v1'
CHECKPOINT = ROOT / 'work/recognition_v4/best.pt'
METHODS = ['single', 'feedback2', 'feedback3', 'feedback5', 'feedback8',
           'ensemble2', 'ensemble4', 'ensemble8', 'backprojection3']
PROFILES = ['standard28', 'mild28', 'clean28']


def png(tensor, path):
    array = tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    Image.fromarray(array).save(path)


@torch.inference_mode()
def score(n, value, target, lpips, face):
    assert value.shape == target.shape and torch.isfinite(value).all()
    # Match the exported UI pixels for perceptual/face-feature diagnostics.
    exported = value.mul(255).round().clamp(0, 255).div(255)
    clean = face(target)
    return {
        'psnr_y': n.Metrics.psnr_y(value, target, 4).cpu().tolist(),
        'ssim_y': n.Metrics.ssim_y(value.cpu(), target.cpu(), 4).tolist(),
        'lpips_alex': lpips(exported, target, normalize=True).flatten().cpu().tolist(),
        'mobileface_cosine': (face(exported) * clean).sum(1).cpu().tolist()}


def means(values):
    return {method: {metric: float(np.mean(v)) for metric, v in entry.items()} for method, entry in values.items()}


def append(bank, key, values):
    if key not in bank:
        bank[key] = {metric: [] for metric in values}
    for metric, items in values.items():
        bank[key][metric].extend(items)


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'selection.json').exists():
        raise ValueError('Preserve completed selection; use a new experiment directory rather than silently retuning.')
    n, _ = runtime()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load(n, CHECKPOINT)
    lpips = load_lpips_alex(n.device)
    face = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as ids:
        val, test = ids['validation'][:128], ids['test'][:512]
    protocol = {
        'source_sha256': sha(__file__), 'inference_source_sha256': sha(ROOT / 'restoration_passes.py'),
        'checkpoint_sha256': sha(CHECKPOINT), 'methods': METHODS,
        'profiles': PROFILES, 'validation_indices': list(range(128)), 'test_indices_if_selected': list(range(512)),
        'literal_cascade_diagnostic_indices': list(range(16)),
        'literal_cascade_dimensions': {'one_pass': [112, 112], 'two_passes': [448, 448], 'three_passes': [1792, 1792]},
        'literal_cascade_metric_policy': 'Actual successive x4 outputs, then antialiased resize to112 for fair comparison with the same clean target. Literal diagnostic uses the same first16 validation inputs per profile.',
        'feedback': 'Each additional pass downsamples the previous output to the original LR dimensions, then runs the same model again. No new observation is introduced.',
        'self_ensemble': 'Same original input in 2,4,8 flipped/rotated views; inverse transforms and average, always x4 output.',
        'backprojection': 'One neural pass, then 3 residual-consistency corrections with fixed0.2 step and bicubic forward assumption; not 3 neural passes.',
        'selection_before_test': 'Only ensembles eligible. Every validation profile must have PSNR >= single-0.03dB, SSIM >= single, LPIPS <= single+0.002 and face cosine >= single-0.003. Select eligible method with lowest mean LPIPS, require at least0.5% relative mean LPIPS reduction. Ties prefer fewer passes. Otherwise retain single. No per-image switching.',
        'no_weight_updates': True, 'no_test_reference_in_prediction': True,
        'limits': 'Historical split reuse; this is an additional diagnostic with selection on validation. Face cosine and SSIM are not restoration percentages. No new full QMUL verification benchmark.'}
    write_json(OUT / 'protocol.json', protocol)
    x = torch.from_numpy(np.asarray(hr[int(val[0]), 8:120, 8:120]).copy()).permute(2, 0, 1).float().unsqueeze(0).to(n.device) / 255
    low = resize(x, (28, 28))
    a = model(low).clamp(0, 1)
    identical_error = max(float((model(low).clamp(0, 1) - a).abs().max()) for _ in range(2))
    result = {'validation': {}, 'literal_cascade_validation16': {}, 'same_input_repeat_max_abs_difference': identical_error}
    for profile in PROFILES:
        data = n.FixedEvalSet(hr, val, InputDegradation(n, profile), 112, 1000)
        bank = {}
        for start, (low, target) in enumerate(data.batches(16)):
            low, target = low.to(n.device), target.to(n.device)
            single = model(low).clamp(0, 1)
            outputs = {'single': single}
            value = single
            for iteration in range(2, 9):
                value = model(resize(value, low.shape[-2:]).contiguous()).clamp(0, 1)
                if iteration in {2, 3, 5, 8}:
                    outputs[f'feedback{iteration}'] = value
            for count, value in ensemble_predictions(model, low):
                outputs[f'ensemble{count}'] = value
            outputs['backprojection3'] = predict(model, low, 'backprojection3')
            for method, value in outputs.items():
                append(bank, method, score(n, value, target, lpips, face))
                if start == 0:
                    folder = OUT / 'validation_examples' / profile
                    folder.mkdir(parents=True, exist_ok=True)
                    for i in range(min(6, len(value))):
                        png(value[i], folder / f'{i:02}_{method}.png')
            if start == 0:
                for i in range(6):
                    png(low[i], folder / f'{i:02}_input.png'); png(target[i], folder / f'{i:02}_reference.png')
            print(f'VALIDATION {profile}: {(start+1)*16}/128 all fixed-size methods', flush=True)
        result['validation'][profile] = means(bank)
        write_json(OUT / f'validation_{profile}_per_image.json', bank)
        # Literal escalating-resolution trial is batch1 to bound memory.
        literal = {}
        for i in range(16):
            low_np = InputDegradation(n, profile)(np.asarray(hr[int(val[i]),8:120,8:120]), np.random.default_rng(1000+i))
            low = torch.from_numpy(low_np.copy()).permute(2,0,1).float().unsqueeze(0).to(n.device)/255
            target = torch.from_numpy(np.asarray(hr[int(val[i]),8:120,8:120]).copy()).permute(2,0,1).float().unsqueeze(0).to(n.device)/255
            value = low
            for count in range(1, 4):
                value = model(value.contiguous()).clamp(0,1)
                normalized = resize(value, (112,112))
                append(literal, f'cascade{count}', score(n, normalized, target, lpips, face))
                if i < 6:
                    png(normalized[0], OUT/'validation_examples'/profile/f'{i:02}_cascade{count}.png')
            if (i+1)%4 == 0:
                print(f'LITERAL {profile}: {i+1}/16 through 3 x4 passes', flush=True)
        result['literal_cascade_validation16'][profile] = means(literal)
        write_json(OUT / f'literal_{profile}_per_image.json', literal)
        write_json(OUT / 'evaluation.json', result)
    baseline = {p: result['validation'][p]['single'] for p in PROFILES}
    eligible = []
    for method in ['ensemble2','ensemble4','ensemble8']:
        if all((v := result['validation'][p][method])['psnr_y'] >= baseline[p]['psnr_y']-.03 and
               v['ssim_y'] >= baseline[p]['ssim_y'] and v['lpips_alex'] <= baseline[p]['lpips_alex']+.002 and
               v['mobileface_cosine'] >= baseline[p]['mobileface_cosine']-.003 for p in PROFILES):
            eligible.append(method)
    selected = min(eligible, key=lambda m: sum(result['validation'][p][m]['lpips_alex'] for p in PROFILES)) if eligible else 'single'
    base_lpips = np.mean([baseline[p]['lpips_alex'] for p in PROFILES])
    selected_lpips = np.mean([result['validation'][p][selected]['lpips_alex'] for p in PROFILES])
    if 1-selected_lpips/base_lpips < .005:
        selected = 'single'
    selection = {'selected': selected, 'eligible': eligible, 'selected_before_test': True,
                 'checkpoint_sha256': sha(CHECKPOINT), 'validation_evaluation_sha256': sha(OUT / 'evaluation.json'),
                 'test_may_change_selection': False, 'default_ui_remains_single': True}
    write_json(OUT / 'selection.json', selection)
    print('FROZEN SELECTION', json.dumps(selection), flush=True)
    if selected != 'single':
        result['test'] = {}
        for profile in PROFILES:
            bank = {}
            data = n.FixedEvalSet(hr, test, InputDegradation(n, profile), 112, 2000)
            for j,(low,target) in enumerate(data.batches(16)):
                low,target=low.to(n.device),target.to(n.device)
                for method in ['single',selected]:
                    value=predict(model,low,method)
                    append(bank,method,score(n,value,target,lpips,face))
                    if j==0:
                        folder=OUT/'test_examples'/profile;folder.mkdir(parents=True,exist_ok=True)
                        for i in range(6):png(value[i],folder/f'{i:02}_{method}.png')
                if j==0:
                    for i in range(6):png(low[i],folder/f'{i:02}_input.png');png(target[i],folder/f'{i:02}_reference.png')
                if (j+1)%8==0:print(f'TEST {profile}: {(j+1)*16}/512',flush=True)
            result['test'][profile]=means(bank)
            write_json(OUT/f'test_{profile}_per_image.json',bank)
            write_json(OUT/'evaluation.json',result)
    # Repeated warm batch1 crop timings, not camera FPS.
    timings={}
    x=torch.zeros(1,3,28,28,device=n.device)
    for method in METHODS+['cascade2','cascade3']:
        predict(model,x,method)
        values=[]
        for _ in range(10 if method.startswith('cascade') else 30):
            if n.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();predict(model,x,method)
            if n.device=='cuda':torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
        timings[method]={'median_ms':statistics.median(values),'samples_ms':values}
    write_json(OUT/'runtime.json',{'device':torch.cuda.get_device_name(0) if n.device=='cuda' else 'CPU','shape':[1,3,28,28],'timings':timings})
    result['selected']=selected
    write_json(OUT/'evaluation.json',result)
    print('COMPLETE',selected,flush=True)


if __name__ == '__main__':
    main()

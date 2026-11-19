"""Paired reconstruction, native-surveillance verification and presentation exports.

The QMUL test benchmark is never used for tuning loss weights or selecting epochs.
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import OUT, evaluate, load, save_grid
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, notebook_definitions, runtime


@torch.inference_mode()
def surveillance(n, current, candidate, faces, folder):
    r = notebook_definitions('sr_part2a_recognition.ipynb', [1, 6, 7, 8, 9, 11], 'identity_surveillance')
    r.device = n.device
    qmul = r.QMULData(str(ROOT / 'work/recognition'))
    if qmul.missing: raise RuntimeError('QMUL protocol has missing images')
    files = sorted({name for pair in qmul.pairs for name in pair})
    dataset = r.ImageSet([qmul.paths[name] for name in files], 28)
    loader = n.DataLoader(dataset, batch_size=64, num_workers=0)
    bank = {net: {key: [] for key in ['native_bicubic', 'lr_bicubic', 'current', 'mixed_quality']} for net in faces}
    count, last = 0, time.monotonic()
    for native, low in loader:
        low, native = low.to(n.device), native.to(n.device)
        views = {'native_bicubic': native,
                 'lr_bicubic': F.interpolate(low, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1),
                 'current': current(low).clamp(0, 1), 'mixed_quality': candidate(low).clamp(0, 1)}
        for net, face in faces.items():
            for name, image in views.items(): bank[net][name].append(face(image).cpu())
        count += len(low)
        if time.monotonic() - last > 25:
            print('QMUL embedded', count, '/', len(files), flush=True); last = time.monotonic()
    index = {name: i for i, name in enumerate(files)}
    ia = np.array([index[a] for a, b in qmul.pairs]); ib = np.array([index[b] for a, b in qmul.pairs])
    result = {'images': len(files), 'pairs': len(qmul.pair_labels),
              'recognizers': {'arcface': 'Frozen ArcFace w600k_r50 (also a training prior)',
                              'mobileface': 'Frozen MobileFaceNet (not used in new loss or selection)'},
              'clean_reference_available': False, 'test_used_to_select_epoch': False, 'conditions': {}, 'mobileface_conditions': {}}
    for net, conditions in bank.items():
        for name, chunks in conditions.items():
            embedding = torch.cat(chunks).numpy()
            key = 'conditions' if net == 'arcface' else 'mobileface_conditions'
            result[key][name] = r.VerificationEval()(embedding[ia], embedding[ib], qmul.pair_labels)
    write_json(folder / 'qmul_verification.json', result)
    # Fixed, uniformly spaced examples; no selection by successful restoration.
    chosen = np.linspace(0, len(files) - 1, 6, dtype=int)
    lows, natives, sizes = [], [], []
    for i in chosen:
        im = Image.open(qmul.paths[files[i]]).convert('RGB'); sizes.append(list(im.size))
        native, low = dataset[int(i)]; natives.append(native); lows.append(low)
    low = torch.stack(lows).to(n.device)
    save_grid({'Native crop (enlarged)': torch.stack(natives), 'Protocol input (28)': low.cpu(),
               'Bicubic': F.interpolate(low, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1).cpu(),
               'Current custom': current(low).clamp(0, 1).cpu(),
               'Mixed-quality candidate': candidate(low).clamp(0, 1).cpu()},
              folder / 'qmul_real_examples.png', 'REAL QMUL | No paired clean original exists for these crops')
    write_json(folder / 'qmul_examples.json', {'files': [files[i] for i in chosen], 'native_sizes': sizes,
               'selection': 'Uniformly spaced indices of sorted verification filenames, not quality-selected'})
    print('QMUL RESULT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='mixed_quality_v1')
    parser.add_argument('--test-count', type=int, default=2000)
    parser.add_argument('--severity-count', type=int, default=512)
    parser.add_argument('--skip-qmul', action='store_true')
    parser.add_argument('--specialist', action='store_true', help='Evaluate final epoch as a clean-input specialist if the general validation guard fails')
    args = parser.parse_args()
    folder = OUT / args.run
    selected = json.loads((folder / 'validation_result.json').read_text())
    if not selected['eligible'] and not args.specialist:
        raise RuntimeError('No epoch passed general validation guards. --specialist evaluates the final epoch without claiming a general upgrade.')
    candidate_path = folder / ('last.pt' if args.specialist else 'best.pt')
    n, _ = runtime()
    original = load(n, ROOT / 'work/full/ckpt/ours_ssc_attn_best.pt')
    current = load(n, ROOT / 'work/visual_sr/full_reconstruction/best.pt')
    candidate = load(n, candidate_path)
    mobile = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as splits:
        test_ids, val_ids, train_ids = splits['test'], splits['validation'], splits['train']
    result = {'checkpoint_sha256': sha(candidate_path), 'checkpoint': candidate_path.relative_to(ROOT).as_posix(),
              'selection': 'Final epoch, clean-input specialist; general validation guard not waived' if args.specialist else 'Validation-selected general candidate',
              'general_validation_guard_passed': selected['eligible'],
              'standard_test_images': min(args.test_count, len(test_ids)),
              'severity_test_images_per_profile': min(args.severity_count, len(test_ids)),
              'independent_metric': 'MobileFaceNet not in this training objective; cosine is not a recovery percentage',
              'profiles': {}}
    for profile in ['standard28', 'mild28', 'clean28', 'clean56']:
        ids = test_ids[:args.test_count if profile == 'standard28' else args.severity_count]
        data = n.FixedEvalSet(hr, ids, InputDegradation(n, profile), 112, 2000)
        means, scores, rows = evaluate(n, data, original, current, mobile,
                                      selected=['bicubic', 'original', 'current'], extra=candidate)
        result['profiles'][profile] = means
        result.setdefault('ssim_threshold_counts', {})[profile] = {
            name: {'ssim_at_least_0_85': int(np.sum(np.asarray(v['ssim_y']) >= .85)),
                   'ssim_at_least_0_90': int(np.sum(np.asarray(v['ssim_y']) >= .9)), 'images': len(ids)}
            for name, v in scores.items()}
        write_json(folder / f'test_{profile}_per_image.json', scores)
        save_grid(rows, folder / f'test_{profile}.png', f'TEST | {profile} | {len(ids)} images | Fixed first 6 shown')
        write_json(folder / 'heldout_reconstruction.json', result)
        print('HELDOUT', profile, json.dumps(means), flush=True)
    # Explicitly labelled train/validation diagnostics, never reported as test scores.
    for split, ids, seed in [('train', train_ids[:128], 3000), ('validation', val_ids[:256], 1000)]:
        data = n.FixedEvalSet(hr, ids, InputDegradation(n, 'standard28'), 112, seed)
        means, scores, rows = evaluate(n, data, original, current, mobile,
                                      selected=['bicubic', 'current'], extra=candidate)
        write_json(folder / f'{split}_diagnostic.json', {'images': len(ids), 'profile': 'standard28', 'metrics': means,
                   'note': 'Training diagnostic, not generalization evidence' if split == 'train' else 'Used for model selection, not an independent test'})
        save_grid(rows, folder / f'{split}_comparison.png', f'{split.upper()} DIAGNOSTIC | standard28 | Fixed first 6 shown')
    # The existing uploaded LR image is passed unchanged; reference used only to display.
    from PIL import Image as PILImage
    upload = ROOT / 'work/ui/uploaded_input.png'
    if upload.exists():
        image = PILImage.open(upload).convert('RGB')
        low = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float().to(n.device) / 255
        with torch.inference_mode():
            sr = candidate(low).clamp(0, 1).cpu()
            rows = {'Actual supplied input': low.cpu(), 'Current custom': current(low).clamp(0, 1).cpu(), 'Mixed-quality candidate': sr}
            ref = ROOT / 'work/ui/uploaded_original.png'
            if ref.exists():
                rows['Clean reference (display only)'] = torch.from_numpy(np.array(PILImage.open(ref).convert('RGB'), copy=True)).permute(2, 0, 1)[None].float() / 255
            save_grid(rows, folder / 'supplied_input_comparison.png', 'EXISTING UPLOADED IMAGE | Same LR pixels', columns=1)
            array = (sr[0].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            PILImage.fromarray(array).save(folder / 'supplied_input_reconstructed.png')
    if not args.skip_qmul:
        arc = FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
        surveillance(n, current, candidate, {'arcface': arc, 'mobileface': mobile}, folder)
    print('EVALUATION COMPLETE', folder, flush=True)


if __name__ == '__main__':
    main()

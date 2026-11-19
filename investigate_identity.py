"""Reproducible input-quality and fidelity investigation; never trains on test data.

All outputs are estimates. Cosine similarity and SSIM are not recovery percentages.
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime
from input_quality import InputDegradation

OUT = ROOT / 'work/identity_investigation'


def load(n, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = n.SRNet(n.cfg).to(n.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


def variants(low, original, detail):
    bic = F.interpolate(low, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
    old = original(low).clamp(0, 1)
    new = detail(low).clamp(0, 1)
    flip = .5 * (new + detail(low.flip(-1)).flip(-1).clamp(0, 1))
    return {'bicubic': bic, 'original': old, 'current': new, 'flip_ensemble': flip,
            'conservative75': .75 * new + .25 * bic,
            'conservative50': .5 * new + .5 * bic}


def save_grid(rows, path, title, columns=6):
    # Nearest-neighbour display magnification adds no apparent texture.
    tile, label, header = 176, 208, 62
    canvas = Image.new('RGB', (label + columns * tile, header + len(rows) * (tile + 25)), '#f3f3ef')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill='#151c19')
    draw.text((12, 32), 'Display enlarged. Estimated detail is not verified identity recovery.', fill='#414b44')
    for i, (name, images) in enumerate(rows.items()):
        top = header + i * (tile + 25)
        draw.text((10, top + 62), name, fill='#151c19')
        for j, value in enumerate(images[:columns]):
            value = value.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            image = Image.fromarray((value * 255).round().astype('uint8'))
            canvas.paste(image.resize((tile, tile), Image.Resampling.NEAREST), (label + j * tile, top))
            draw.text((label + j * tile + 4, top + tile + 3), f'Example {j + 1}', fill='#414b44')
    canvas.save(path)


@torch.inference_mode()
def evaluate(n, data, original, detail, face, selected=None, extra=None):
    scores, last, count = {}, time.monotonic(), 0
    examples = None
    for low, target in data.batches(32):
        low, target = low.to(n.device), target.to(n.device)
        views = variants(low, original, detail)
        if selected: views = {key: views[key] for key in selected}
        if extra: views['mixed_quality'] = extra(low).clamp(0, 1)
        clean_embedding = face(target)
        for name, value in views.items():
            # clean56 is effective 2x: the x4 network's 224 output is reduced
            # to the native 112 target. The reference is never upsampled.
            if value.shape[-2:] != target.shape[-2:]:
                source = low if name == 'bicubic' else value
                value = F.interpolate(source, size=target.shape[-2:], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            entry = scores.setdefault(name, {'psnr_y': [], 'ssim_y': [], 'mobileface_cos': []})
            entry['psnr_y'].extend(n.Metrics.psnr_y(value, target, 4).cpu().tolist())
            entry['ssim_y'].extend(n.Metrics.ssim_y(value.cpu(), target.cpu(), 4).tolist())
            entry['mobileface_cos'].extend((face(value) * clean_embedding).sum(1).cpu().tolist())
            views[name] = value
        if examples is None:
            examples = {'Clean reference (112)': target.cpu(), f'Actual input ({low.shape[-1]})': low.cpu(),
                        **{name: value.cpu() for name, value in views.items()}}
        count += len(low)
        if time.monotonic() - last > 25:
            print('EVALUATED', count, '/', len(data), flush=True); last = time.monotonic()
    averages = {name: {key: float(np.mean(value)) for key, value in values.items()} for name, values in scores.items()}
    return averages, scores, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--extra', type=str)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n, _ = runtime()
    paths = {'original': ROOT / 'work/full/ckpt/ours_ssc_attn_best.pt',
             'current': ROOT / 'work/visual_sr/full_reconstruction/best.pt'}
    original, detail = [load(n, path) for path in paths.values()]
    face = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    ids = np.load(ROOT / 'work/full/split_ids.npz')[args.split][:args.count]
    result = {'split': args.split, 'images': len(ids), 'checkpoint_sha256': {k: sha(v) for k, v in paths.items()},
              'identity_metric': 'MobileFaceNet cosine, diagnostic only; not a percentage or verification accuracy',
              'clean56_protocol': '56 input -> 224 network output -> 112 evaluation; effective 2x, same native 112 reference',
              'profiles': {}}
    extra = load(n, ROOT / args.extra) if args.extra else None
    for profile in ['standard28', 'mild28', 'clean28', 'clean56']:
        data = n.FixedEvalSet(hr, ids, InputDegradation(n, profile), 112, 1000 if args.split == 'validation' else 2000)
        averages, scores, rows = evaluate(n, data, original, detail, face, extra=extra)
        result['profiles'][profile] = averages
        stem = f'{args.split}_{profile}' + ('_mixed' if extra else '')
        write_json(OUT / f'{stem}_per_image.json', scores)
        save_grid(rows, OUT / f'{stem}.png', f'{args.split.upper()} | {profile} | same input to every method')
        print(profile, json.dumps(averages), flush=True)
        write_json(OUT / (args.split + ('_mixed' if extra else '') + '_quality.json'), result)


if __name__ == '__main__':
    main()

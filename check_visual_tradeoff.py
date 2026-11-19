"""Post-hoc diagnostic: distortion, perceptual distance and face features differ.

Uses the same first 128 validation inputs as the existing UI comparison.
No training, weight selection, test-set gallery replacement or baseline tuning.
"""
import base64
import hashlib
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from compare_three_models import MODELS, ROOT, URL, post, write
from quality_metric import load_lpips_alex
from refine_model import FrozenFace, sha

OUT = ROOT / 'work/visual_tradeoff'


def tensor(url):
    with Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))) as im:
        return torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    protocol = {
        'recorded_before_run_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'split': 'validation', 'indices': list(range(128)),
        'profile': 'standard28', 'models': MODELS, 'settings': 'Existing current UI; unchanged v4, GFPGAN 1.4 full output, CodeFormer w=0.5 full output.',
        'additional_metrics': {'lpips_alex': 'Official LPIPS Alex v0.1, lower is closer in learned perceptual features; not a human-recognition score.',
                               'mobileface_cosine': 'Frozen MobileFace cosine to clean reference, higher is closer in face-feature space; not verification accuracy or human likeness.'},
        'pixels': 'Additional metrics use actual exported 8-bit UI PNGs and clean reference PNGs; original PSNR/SSIM are app float metrics.',
        'training_changes': False, 'test_inputs_used': False,
        'case_selection': 'Explicit post-hoc validation illustrations: up to two largest v4 LPIPS advantages over both baselines and two largest v4 LPIPS disadvantages. Both kinds are displayed. Selection is not a new unbiased estimate.',
        'limits': 'Same previously examined validation cohort. External FFHQ pretraining overlap is not excluded; training splits/budgets differ. Metric agreement does not establish human identity recovery.'}
    write(OUT / 'protocol.json', protocol)
    lpips = load_lpips_alex('cpu')
    face = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').eval().requires_grad_(False)
    rows, images = [], []
    for index in protocol['indices']:
        outputs = {model: post({'sample_id': index, 'sample_split': 'validation', 'profile': 'standard28', 'model': model}) for model in MODELS}
        a = outputs[MODELS[0]]
        assert all(r['input'] == a['input'] and r['reference'] == a['reference'] for r in outputs.values())
        target = tensor(a['reference'])
        batch = torch.cat([tensor(outputs[m]['sr']) for m in MODELS] + [tensor(a['bicubic'])])
        targets = target.expand(len(batch), -1, -1, -1)
        distances = lpips(batch, targets, normalize=True).flatten().tolist()
        cosines = (face(batch) * face(target)).sum(1).tolist()
        row = {'index': index, 'input_png_sha256': hashlib.sha256(base64.b64decode(a['input'].split(',')[1])).hexdigest(), 'models': {}}
        for j, model in enumerate(MODELS + ['bicubic']):
            metric = a['metrics']['bicubic'] if model == 'bicubic' else outputs[model]['metrics']['sr']
            row['models'][model] = {**metric, 'lpips_alex': distances[j], 'mobileface_cosine': cosines[j]}
        rows.append(row)
        images.append({'input': a['input'], 'reference': a['reference'], 'bicubic': a['bicubic'], **{m: outputs[m]['sr'] for m in MODELS}})
        if (index + 1) % 16 == 0:
            print(f'Visual diagnostic: {index + 1}/128 validation images', flush=True)
    write(OUT / 'per_image.json', rows)
    metrics = ['psnr', 'ssim', 'lpips_alex', 'mobileface_cosine']
    summary = {'cohort': '128 previously examined FFHQ validation images, standard28',
               'mean_metrics': {m: {metric: float(np.mean([r['models'][m][metric] for r in rows])) for metric in metrics} for m in MODELS + ['bicubic']},
               'ours_better_than_both_baselines_image_counts': {}}
    for metric in metrics:
        better = (lambda a, b: a < b) if metric == 'lpips_alex' else (lambda a, b: a > b)
        summary['ours_better_than_both_baselines_image_counts'][metric] = sum(all(better(r['models']['identity_v4'][metric], r['models'][other][metric]) for other in MODELS[1:]) for r in rows)
    gaps = [min(r['models'][m]['lpips_alex'] for m in MODELS[1:]) - r['models']['identity_v4']['lpips_alex'] for r in rows]
    win = [i for i in np.argsort(gaps)[::-1].tolist() if gaps[i] > 0][:2]
    loss = [i for i in np.argsort(gaps).tolist() if gaps[i] < 0][:2]
    selected = [{'index': i, 'role': 'v4 lower LPIPS than both baselines', 'selection': 'post-hoc validation illustration'} for i in win]
    selected += [{'index': i, 'role': 'one or both heavy baselines lower LPIPS than v4', 'selection': 'post-hoc validation counterexample'} for i in loss]
    write(OUT / 'summary.json', summary)
    write(OUT / 'selected_cases.json', selected)
    model_names = {'identity_v4': 'Ours v4', 'restoration': 'GFPGAN 1.4', 'codeformer': 'CodeFormer w=0.5', 'bicubic': 'Bicubic'}
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Why sharper and closer differ</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#16261b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border:1px solid #ccc}.row{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;break-inside:avoid;margin:20px 0}.row img{width:100%;image-rendering:auto;aspect-ratio:1;object-fit:contain}.row small{display:block;min-height:35px}section{break-inside:avoid}aside{padding:15px;background:#fff1cb}@media(max-width:700px){.row{grid-template-columns:repeat(3,1fr)}}@media print{body{font-size:12px;margin:0}.case{break-before:page}}</style>',
            '<h1>Sharper-looking faces and smaller reconstruction error are different goals</h1>',
            '<p>A smoother output can score better on PSNR/SSIM while a generative model produces a sharper-looking face. Neither metric alone proves the original identity was recovered. Larger models are not guaranteed to win a different objective or a particular degradation.</p>',
            '<aside>These are the same 128 validation images already examined. No new training or accuracy improvement occurred. The illustrations below were selected after measuring LPIPS and include strengths and counterexamples. They are not random test examples or evidence of universal superiority.</aside>',
            '<h2>All 128 validation images, benchmark blur/noise/JPEG</h2><table><tr><th>Model</th><th>PSNR-Y ↑</th><th>SSIM-Y ↑</th><th>LPIPS-Alex ↓</th><th>MobileFace cosine ↑</th></tr>']
    for m, values in summary['mean_metrics'].items():
        body.append('<tr><td>' + model_names[m] + '</td>' + ''.join(f'<td>{values[k]:.4f}</td>' for k in metrics) + '</tr>')
    body.append('</table><p>LPIPS is learned perceptual distance, not human identity recognition. MobileFace cosine is a diagnostic similarity to the clean reference, not classification accuracy or a percentage restored. Extra metrics use the actual exported 8-bit PNGs. FFHQ pretraining overlap and different training procedures remain limitations.</p>')
    for item in selected:
        i = item['index']; view = images[i]
        body += [f'<section class="case"><h2>Validation face {i + 1:02}: {html.escape(item["role"])}</h2>',
                 '<p>Selected after LPIPS measurement; no training-image substitution. Compare with the clean original yourself.</p><div class="row">']
        for key, label in [('input', 'Exact degraded input'), ('bicubic', 'Bicubic'), ('identity_v4', 'Ours v4'), ('restoration', 'GFPGAN 1.4'), ('codeformer', 'CodeFormer w=0.5'), ('reference', 'Clean original: comparison only')]:
            body.append(f'<div><small>{label}</small><img src="{view[key]}" alt="{label}"></div>')
        body.append('</div><table><tr><th>Model</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>Face-feature cosine</th></tr>')
        for m, values in rows[i]['models'].items():
            body.append('<tr><td>' + model_names[m] + '</td>' + ''.join(f'<td>{values[k]:.4f}</td>' for k in metrics) + '</tr>')
        body.append('</table></section>')
    body.append('<h2>What to say in the presentation</h2><p>Our model is smaller and faster in the measured crop pipeline, and achieved higher PSNR/SSIM on the declared synthetic degradation. Heavy generative models may look sharper and may be better choices when perceptual quality is the priority. We have not demonstrated universally better identity recovery. Training images can demonstrate fitting, but cannot be used as unseen test evidence.</p>')
    body.append('<p>References: <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html">Perception–distortion tradeoff</a>; <a href="https://github.com/richzhang/PerceptualSimilarity">LPIPS</a>; <a href="https://github.com/sczhou/CodeFormer#testing">CodeFormer quality/fidelity control</a>.</p></html>')
    (OUT / 'comparison.html').write_text('\n'.join(body), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('Selected validation illustrations:', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()

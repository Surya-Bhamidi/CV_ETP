"""Fixed, same-input diagnostic comparison of the three presentation models.

Uses the running local app. This is not a new independent FFHQ benchmark:
the official external models may have pretrained on the project FFHQ split.
"""
import base64
import hashlib
import html
import json
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'work/three_model_comparison'
URL = 'http://127.0.0.1:8765'
MODELS = ['identity_v4', 'restoration', 'codeformer']


def post(payload):
    request = urllib.request.Request(URL + '/api/compare', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request, timeout=180))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = json.load(urllib.request.urlopen(URL + '/api/status'))
    assert status['state'] == 'ready' and status['default_model'] == 'identity_v4'
    assert [m['id'] for m in status['models']] == MODELS
    protocol = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'models': status['models'],
        'validation': {'indices': list(range(128)), 'profile': 'standard28'},
        'test_gallery': {'indices': list(range(12)), 'profiles': ['standard28', 'clean28']},
        'qmul_gallery': {'indices': list(range(6)), 'profile': 'native'},
        'settings_selection': 'No parameter tuning or checkpoint selection in this run. CodeFormer w=0.5 is the official cropped-face example setting.',
        'shared_input': 'The same supplied crop for every model; no per-model detector/alignment; externals internally resize to 512x512. All outputs x4 at original aspect ratio.',
        'quality_metrics': 'Mean per-image PSNR-Y/SSIM-Y, 4-pixel border; app float predictions. Larger is closer to the supplied reference, not a restoration percentage.',
        'limits': [
            'External FFHQ pretraining overlap is not excluded. FFHQ comparisons are descriptive diagnostics, not an independent generalization ranking.',
            'Custom training includes official QMUL training identities; official external training is different. Exact training splits and budgets are not matched.',
            'Historical test reuse exists. No test images or clean references are used to alter the outputs in this run.',
            'QMUL has no paired clean reference. No PSNR, SSIM, or single-image restoration percentage can be measured there.',
            'Native QMUL crops can be off-angle or unaligned; these are crop-pipeline comparisons, not full official detector/alignment pipelines.',
            'No human identity-recognition study. Eye/glasses detail and severe blur remain limitations.',
            'Existing QMUL study CodeFormer w=1 scores do not describe the new w=0.5 UI setting. No new full QMUL verification accuracy is claimed.'
        ]}
    # Written before any evaluation. Selection is by fixed index, never output quality.
    write(OUT / 'protocol.json', protocol)
    rows = []
    galleries = []
    cohorts = [('validation', 'standard28', list(range(128)), False),
               ('test', 'standard28', list(range(12)), True),
               ('test', 'clean28', list(range(12)), True),
               ('qmul', 'native', list(range(6)), True)]
    for split, profile, indices, save_images in cohorts:
        for index in indices:
            image_row = {'split': split, 'profile': profile, 'index': index, 'models': {}}
            input_png = reference_png = None
            for model in MODELS:
                payload = {'sample_id': index, 'sample_split': split, 'model': model}
                if split != 'qmul':
                    payload['profile'] = profile
                r = post(payload)
                if input_png is None:
                    input_png, reference_png = r['input'], r['reference']
                assert r['input'] == input_png and r['reference'] == reference_png
                assert r['output_size'] == [4 * n for n in r['input_size']]
                if split == 'qmul':
                    assert r['metrics'] is None and r['reference'] is None
                rows.append({'split': split, 'profile': profile, 'index': index, 'model': model,
                             'input_png_sha256': hashlib.sha256(base64.b64decode(r['input'].split(',')[1])).hexdigest(),
                             'metrics': r['metrics'], 'selected_model_ms': r['processing_ms'],
                             'input_size': r['input_size'], 'output_size': r['output_size']})
                if save_images:
                    image_row['models'][model] = r['sr']
                    image_row.update(input=r['input'], reference=r['reference'], bicubic=r['bicubic'], name=r['name'])
            if save_images:
                galleries.append(image_row)
            if (index + 1) % 16 == 0 or index == indices[-1]:
                print(f'{split}/{profile}: {index + 1}/{len(indices)} inputs, all 3 models', flush=True)
        write(OUT / 'measurements.json', rows)
    summary = {}
    for split, profile, _, _ in cohorts:
        key = split + '/' + profile
        summary[key] = {}
        for model in MODELS:
            subset = [r for r in rows if (r['split'], r['profile'], r['model']) == (split, profile, model)]
            summary[key][model] = {'images': len(subset), 'median_selected_model_ms': statistics.median(r['selected_model_ms'] for r in subset)}
            if split != 'qmul':
                summary[key][model].update(
                    psnr_y=statistics.mean(r['metrics']['sr']['psnr'] for r in subset),
                    ssim_y=statistics.mean(r['metrics']['sr']['ssim'] for r in subset))
        if split != 'qmul':
            subset = [r for r in rows if (r['split'], r['profile'], r['model']) == (split, profile, MODELS[0])]
            summary[key]['bicubic_reference'] = {
                'psnr_y': statistics.mean(r['metrics']['bicubic']['psnr'] for r in subset),
                'ssim_y': statistics.mean(r['metrics']['bicubic']['ssim'] for r in subset)}
    write(OUT / 'summary.json', summary)
    model_labels = {m['id']: m['label'] for m in status['models']}
    document = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-model comparison</title>',
                '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#17221a}h1{font-size:30px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;break-inside:avoid;margin:18px 0 30px}.tile img{width:100%;image-rendering:auto;background:#eee;aspect-ratio:1;object-fit:contain}.tile small{display:block;margin-bottom:8px}section{break-inside:avoid}nav a{margin-right:20px}.warning{background:#fff5d7;padding:14px} @media(max-width:700px){.grid{grid-template-columns:repeat(3,1fr)}}@media print{body{font-size:11px;margin:0;padding:0}.grid{gap:8px;margin:10px 0 18px}h2{break-before:page}.tile small{font-size:10px}}</style>',
                '<h1>One project model, two official pretrained baselines</h1>',
                '<p>Our selected v4: 479,109 parameters. GFPGAN: 86,444,252. CodeFormer: 94,112,707. Historical experiments remain archived; no new model training was performed for this comparison.</p>',
                '<p class="warning">FFHQ comparisons are diagnostic: pretrained overlap is not excluded. All models receive identical input pixels. These results do not establish universal superiority or a percentage of identity recovered.</p>',
                '<nav><a href="#test-standard28">Blur/noise/JPEG</a><a href="#test-clean28">Downsampling only</a><a href="#qmul-native">Real QMUL</a></nav>',
                '<h2>Measured paired diagnostics</h2><p>Indices 0–127 of validation, and 0–11 of test, fixed before this run. No best-looking output selection. Timing covers only selected-model inference, including its internal resizing; not capture, detection, encoding or browser rendering. PSNR-Y / SSIM-Y compare to clean originals.</p>']
    for key, methods in summary.items():
        document.append('<h3>' + html.escape(key) + '</h3><table><tr><th>Method</th><th>PSNR-Y (dB)</th><th>SSIM-Y</th><th>Median model ms</th></tr>')
        for model, scores in methods.items():
            document.append('<tr><td>' + html.escape(model_labels.get(model, 'Bicubic reference')) + '</td><td>' +
                            (f'{scores["psnr_y"]:.3f}' if 'psnr_y' in scores else 'No HR target') + '</td><td>' +
                            (f'{scores["ssim_y"]:.4f}' if 'ssim_y' in scores else 'No HR target') + '</td><td>' +
                            (f'{scores["median_selected_model_ms"]:.1f}' if 'median_selected_model_ms' in scores else '—') + '</td></tr>')
        document.append('</table>')
    document.append('<h2>What can be presented</h2><p>Show the clean original separately from the degraded model input. Explain which degradation was used; changing that setting changes the experiment. Compare eye shape, spectacle frames, nose and mouth on the same crop. The clean reference is never an inference input.</p>')
    document.append('<ul>' + ''.join('<li>' + html.escape(v) + '</li>' for v in protocol['limits']) + '</ul>')
    for split, profile, _, save_images in cohorts:
        if not save_images:
            continue
        document.append(f'<h2 id="{split}-{profile}">{html.escape(split + " / " + profile)}</h2>')
        for row in [r for r in galleries if (r['split'], r['profile']) == (split, profile)]:
            document.append('<section><h3>' + html.escape(row['name']) + '</h3><div class="grid">')
            images = [('Input (display enlarged)', row['input']), ('Bicubic', row['bicubic'])]
            images += [(model_labels[m], row['models'][m]) for m in MODELS]
            images += [('Clean original (comparison only)' if row['reference'] else 'Clean original unavailable', row['reference'])]
            for label, url in images:
                document.append('<div class="tile"><small>' + html.escape(label) + '</small>' + (f'<img src="{url}" alt="{html.escape(label)}">' if url else '<p>No paired HR target</p>') + '</div>')
            document.append('</div></section>')
    document.append('</html>')
    (OUT / 'comparison.html').write_text('\n'.join(document), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

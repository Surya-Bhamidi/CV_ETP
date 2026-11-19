"""Make a portable, measured improvement report including unsuccessful checks."""
import base64
import html
import json
import re

import numpy as np

from refine_model import write_json
from run_project import ROOT
from write_qmul_study import grid, markdown_table

OUT = ROOT / 'work/qmul_quality'
STUDY = ROOT / 'work/qmul_study'


def main():
    folder = OUT / 'balanced_merge_v1'
    selection = json.loads((folder / 'validation_result.json').read_text())
    quality = json.loads((OUT / 'quality_results.json').read_text())
    names = ['bicubic', 'ours_reconstruction', 'ours_mixed', 'domain_full']
    balanced = selection['selected_new_checkpoint']
    if balanced: names.append('balanced_camera')
    labels = {'bicubic': 'Bicubic', 'ours_reconstruction': 'Previous app default',
              'ours_mixed': 'Previous mixed-quality SR', 'domain_full': 'Camera adaptation',
              'balanced_camera': 'Texture-preserving camera adaptation'}
    results = {name: json.loads((STUDY / 'results' / (name + '.json')).read_text()) for name in names}
    rows = [[labels[name], f'{v["evaluators"]["arcface"]["acc"]*100:.2f}',
             f'{v["evaluators"]["mobileface"]["acc"]*100:.2f}',
             f'{v["evaluators"]["mobileface"]["auc"]:.5f}'] for name, v in results.items()]
    lines = ['# Measured camera reconstruction improvements', '',
        'The custom network was adapted using actual official QMUL training crops and paired FFHQ training images. '
        'It retains 479,109 parameters, learned sparse sampling convolution, adaptive top-50% spatial key selection and normalized kernel linear attention. '
        'The key locations vary with content; the retained count is fixed. Dense masked operations remain in the original attention implementation, so token sparsity does not establish proportional compute savings.', '',
        '## Recognition on identical QMUL inputs', '',
        'Every row processes all 9,913 verification images and all 10,640 labeled pairs with identical cached RGB28 inputs, 112 outputs, recognizers and fixed folds. '
        'Thresholds use the other nine folds. These are pair-verification accuracies, not percentages of facial detail restored.', '']
    lines += markdown_table(['Method', 'ArcFace accuracy (%)', 'MobileFace accuracy (%)', 'MobileFace AUC'], rows)
    lines += ['', 'The raw camera candidate improves recognition relative to the previous mixed model and the app default. '
        'It does not consistently beat bicubic across both recognizers, and tiny differences should not be overstated. '
        'The full same-data external-model comparison and architecture ablations are in [STUDY.md](../qmul_study/STUDY.md).', '',
        '## Texture and structure: real paired targets', '',
        'The original camera objective improved SSIM slightly but regressed on LPIPS. '
        'A validation-selected strength-0.30 edge-contrast filter partly reduced that regression; it is not promoted and cannot recover missing information. '
        'The additional balanced training run restores paired perceptual, edge and facial-feature losses and reduces the camera observation-loss weight. '
        'It starts from the previous mixed-quality weights, uses 16,384 paired FFHQ images plus 16,384 official-training camera images for two epochs, and keeps the same architecture.', '',
        'Checkpoint selection uses 128 FFHQ validation images per condition and 512 camera validation crops from 256 held-out official-training identities. '
        'It maximizes camera-validation ArcFace AUC while requiring LPIPS no worse than the mixed baseline in both conditions, PSNR within 0.10 dB and SSIM within 0.0015. '
        'MobileFaceNet and QMUL verification data are excluded from training and checkpoint selection. '
        'This is an exploratory loss refinement, separate from the matched architecture ablation study.', '']
    paired = json.loads((folder / 'paired_test.json').read_text()) if balanced else None
    source = paired['profiles'] if balanced else quality['test']
    quality_rows = []
    for profile, methods in source.items():
        for name, values in methods.items():
            quality_rows.append([profile, name, f'{values["psnr_y"]:.4f}', f'{values["ssim_y"]:.5f}', f'{values["lpips"]:.5f}'])
    lines += markdown_table(['FFHQ test condition', 'Method', 'PSNR-Y', 'SSIM-Y', 'LPIPS (lower better)'], quality_rows)
    lines += ['', 'Each condition uses the same 512 held-out FFHQ images across methods. These paired results are not QMUL reconstruction metrics.', '',
              '## Release decision', '']
    if balanced:
        review = json.loads((folder / 'review.json').read_text())
        lines += [f'Validation selected a {selection["selected_texture_weight"]:.0%} parameter update toward extra epoch {selection["selected_epoch"]}. '
                  + ('The balanced model passes the recorded release checks and is available as a separate experimental app option.' if review['available_for_comparison'] else 'The balanced model fails at least one release check and is not exposed in the app.'), '']
        lines += markdown_table(['Check', 'Pass'], [[k, v] for k, v in review['checks'].items()])
    else:
        lines += ['Neither new training epoch passed the predeclared validation selection criteria. '
                  'The mixed checkpoint is retained for the balanced run; no new quality improvement is claimed and no balanced UI option is exposed. '
                  'Raw camera adaptation remains available as an explicitly disclosed recognition/structure tradeoff.']
    lines += ['', 'The unblended texture-training epochs and their eligibility are recorded separately. '
              'Weight interpolation is an additional exploratory validation selection, not a relabeling of an ineligible epoch. '
              'It produces one 479,109-parameter checkpoint with one inference pass, without averaging model outputs. '
              'The selected weight fraction is not an accuracy or restoration percentage.', '',
              'The existing app default is unchanged. All prior checkpoints and the failed candidates remain archived.', '',
        '## What the evidence supports', '',
        '- The QMUL camera targets are native low-resolution observations resized larger, not clean HR originals. No QMUL PSNR, SSIM, LPIPS or 90% restoration claim is justified.',
        '- Verification measures correct same/different pair decisions using identity labels; it does not require a paired clean HR image. Human recognition and exact identity recovery have not been established.',
        '- A generative face prior can invent a plausible but different person. The measured heavy restorers did not improve verification under the shared crop protocol. Do not present invented details as recovered evidence.',
        '- Keep usable native input dimensions. The app preserves aspect ratio and does not shrink QMUL gallery images to 28; the offline benchmark intentionally forces identical 28-pixel inputs for every method.',
        '- Reliable additional visual information requires better observations: less compression, larger face crops, or tracked and aligned original video frames. A single very blurred crop cannot determine details that it never recorded.',
        '- Repeated-test comparisons, shared identities across pairs, single training seeds, unknown pretraining overlap and camera-domain shift limit generalization claims. The results do not establish 90% recovery.', '',
        '## Artifacts', '',
        '- [Full same-QMUL study](../qmul_study/STUDY.md)',
        '- [All main-study metrics](../qmul_study/comparison.csv)',
        '- [Validation and sharpening experiment](quality_results.json)',
        '- [Balanced training selection](balanced_v1/validation_result.json)',
        '- [Exploratory weight selection](balanced_merge_v1/validation_result.json)',
        '- [Live camera comparison](http://127.0.0.1:8765/?split=qmul&model=' + ('balanced' if balanced and review['available_for_comparison'] else 'domain') + ')', '']
    figures = [('Raw camera adaptation: paired clean-input tradeoff', OUT / 'clean28_quality.png')]
    if balanced:
        from write_qmul_study import LABELS
        LABELS['balanced_camera'] = labels['balanced_camera']
        manifest = json.loads((STUDY / 'cache/manifest.json').read_text())
        grid(['bicubic', 'ours_mixed', 'domain_full', 'balanced_camera'], OUT / 'balanced_camera_examples.png',
             'Same fixed QMUL crops | no paired clean original', manifest)
        figures = [('QMUL: actual crops and reconstructions', OUT / 'balanced_camera_examples.png'),
                   ('FFHQ: downsampling-only input', folder / 'clean28_comparison.png'),
                   ('FFHQ: blur, noise and compression', folder / 'standard28_comparison.png')]
    (OUT / 'QUALITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8"><title>Measured camera reconstruction improvements</title>',
             '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18271e}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bac5b7;padding:7px;text-align:left}img{width:100%;height:auto}h2{margin-top:2em}@media print{img{break-inside:avoid}h2{break-after:avoid}}</style>']
    # Render this writer's small controlled Markdown subset without a new dependency.
    inline = lambda value: re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html.escape(value, quote=True))
    in_table = False
    for line in lines:
        if line.startswith('| '):
            if not in_table: parts.append('<table>'); in_table = True
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if all(cell == '---' for cell in cells): continue
            parts.append('<tr>' + ''.join('<td>' + inline(cell) + '</td>' for cell in cells) + '</tr>')
        else:
            if in_table: parts.append('</table>'); in_table = False
            if line.startswith('# '): parts.append('<h1>' + inline(line[2:]) + '</h1>')
            elif line.startswith('## '): parts.append('<h2>' + inline(line[3:]) + '</h2>')
            elif line: parts.append('<p>' + inline(line) + '</p>')
    if in_table: parts.append('</table>')
    for title, path in figures:
        parts.append('<h2>' + html.escape(title) + '</h2><img alt="' + html.escape(title) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">')
    (OUT / 'quality_presentation.html').write_text('\n'.join(parts), encoding='utf-8')
    # Independently reproduce the new row's pair decisions and metrics.
    if balanced:
        from sklearn.metrics import roc_auc_score
        with np.load(STUDY / 'emb/balanced_camera_pairs.npz') as data:
            for net in ['arcface', 'mobileface']:
                values = results['balanced_camera']['evaluators'][net]
                correct = (data[net + '_scores'] >= np.array(values['thresholds'])[data['folds']]) == data['labels']
                assert np.array_equal(correct, data[net + '_correct'])
                assert abs(correct.mean() - values['acc']) < 1e-12
                assert abs(roc_auc_score(data['labels'], data[net + '_scores']) - values['auc']) < 1e-12
        with np.load(STUDY / 'examples/balanced_camera.npz') as data:
            assert np.array_equal(data['indices'], np.linspace(0, 9912, 6, dtype=int))
    write_json(OUT / 'report_checks.json', {'validation_selection_recorded': True, 'raw_new_pair_decisions_reproduced': balanced,
               'paired_targets_only_claimed_for_ffhq': True, 'no_restoration_percentage_claimed': True})
    print('QUALITY REPORT WRITTEN', flush=True)


if __name__ == '__main__': main()

"""Write the full same-cohort comparison, intervention and adaptation-ablation study."""
import base64
import csv
import html
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_qmul import DOMAIN_METHODS, OUT
from refine_model import sha, write_json
from run_project import ROOT

MAIN = ['bicubic', 'conv_baseline', 'ours_original', 'ours_reconstruction', 'ours_mixed', 'ours_flip2',
        'ours_fidelity50', 'realesrgan', 'realesrgan_x4plus', 'gfpgan_amp', 'codeformer_tf32']
INTERVENTIONS = ['ours_mixed', 'ours_no_attention', 'ours_fixed_sampling', 'ours_dense_keys', 'ours_fidelity50']
LABELS = {'native_bicubic': 'Native crop -> bicubic', 'bicubic': 'Shared LR28 -> bicubic',
          'conv_baseline': 'Trained convolution baseline', 'ours_original': 'Ours: original pixel model',
          'ours_reconstruction': 'Ours: reconstruction refinement', 'ours_mixed': 'Ours: mixed quality',
          'ours_flip2': 'Ours: mixed + flip averaging', 'ours_fidelity50': 'Ours: 50% fidelity blend',
          'ours_no_attention': 'Attention bypassed (no retraining)', 'ours_fixed_sampling': 'Sampling offsets disabled (no retraining)',
          'ours_dense_keys': 'All attention keys (no retraining)', 'realesrgan': 'Real-ESRGAN compact',
          'realesrgan_x4plus': 'Real-ESRGAN x4plus (23 RRDB)', 'gfpgan_amp': 'GFPGAN 1.4', 'codeformer_tf32': 'CodeFormer, fidelity 1',
          'domain_full': 'Ours: camera adaptation', 'domain_ffhq_control': 'Matched extra FFHQ-only control',
          'domain_no_attention': 'Camera adaptation, no attention', 'domain_no_ssc': 'Camera adaptation, regular convolution'}


def markdown_table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]


def paired_interval(a, b):
    delta = a.astype(float) - b.astype(float); rng = np.random.default_rng(42); samples = []
    for _ in range(16):
        indices = rng.integers(0, len(delta), size=(125, len(delta)))
        samples.extend(delta[indices].mean(1).tolist())
    return {'difference_percentage_points': float(delta.mean() * 100),
            'paired_pair_bootstrap_95_interval_pp': (np.percentile(samples, [2.5, 97.5]) * 100).tolist(),
            'resamples': 2000, 'scope': 'Fixed-threshold pair resampling; not identity-cluster, training-seed or recalibration uncertainty'}


def grid(methods, path, title, manifest):
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 15)
    tile, label, height = 150, 280, 176
    rows = [('Input28 (same pixels)', None)] + [(LABELS[m], m) for m in methods]
    canvas = Image.new('RGB', (label + 6 * tile, 70 + height * len(rows)), '#f1f2ec'); draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), title, font=font, fill='#18271e')
    draw.text((15, 35), 'Actual QMUL crops; no clean HR target. Fixed uniformly spaced filenames; display enlarged.', font=font, fill='#455149')
    low = np.load(OUT / 'cache/lr28.npy', mmap_mode='r')
    indices = np.linspace(0, len(low) - 1, 6, dtype=int)
    for row, (name, method) in enumerate(rows):
        top = 70 + row * height; draw.text((10, top + 55), name, font=font, fill='#18271e')
        if method is None: images = low[indices]
        else:
            with np.load(OUT / 'examples' / (method + '.npz')) as saved:
                assert np.array_equal(saved['indices'], indices); images = saved['images']
        for col, image in enumerate(images):
            canvas.paste(Image.fromarray(image).resize((tile, tile), Image.Resampling.NEAREST), (label + col * tile, top))
            draw.text((label + col * tile + 3, top + tile + 2), manifest['files'][indices[col]][:19], font=font, fill='#455149')
    canvas.save(path)


def main():
    names = list(dict.fromkeys(['native_bicubic'] + MAIN + INTERVENTIONS + DOMAIN_METHODS))
    missing = [name for name in names if not (OUT / 'results' / (name + '.json')).exists()]
    if missing: raise RuntimeError('Full study is not complete: ' + ', '.join(missing))
    results = {name: json.loads((OUT / 'results' / (name + '.json')).read_text()) for name in names}
    manifest = json.loads((OUT / 'cache/manifest.json').read_text())
    ff = json.loads((OUT / 'domain_ffhq_test.json').read_text())
    checks = {'all_methods_full_9913_images': all(v['images'] == 9913 for v in results.values()),
              'all_methods_full_10640_pairs': all(v['pairs'] == 10640 for v in results.values()),
              'one_identical_pair_cohort_and_fold_assignment': len({v['protocol_sha256'] for v in results.values()}) == 1,
              'same_lr_input_pixels_for_every_lr28_method': len({v['metadata']['lr28_sha256'] for v in results.values()}) == 1,
              'same_two_frozen_recognizers': len({json.dumps(v['metadata']['recognizer_sha256'], sort_keys=True) for v in results.values()}) == 1,
              'no_clean_hr_qmul_metrics_claimed': manifest['clean_hr_reference_available'] is False}
    assert all(checks.values()), checks
    write_json(OUT / 'study_checks.json', checks)
    differences = {}
    with np.load(OUT / 'emb/ours_mixed_pairs.npz') as saved:
        base = {net: saved[net + '_correct'] for net in ['arcface', 'mobileface']}
    for name in ['bicubic', 'ours_fidelity50', 'realesrgan_x4plus', 'gfpgan_amp', 'codeformer_tf32', *DOMAIN_METHODS]:
        with np.load(OUT / 'emb' / (name + '_pairs.npz')) as saved:
            differences[name] = {net: paired_interval(saved[net + '_correct'], base[net]) for net in base}
    write_json(OUT / 'paired_accuracy_differences.json', differences)
    fields = ['method', 'label', 'role', 'parameters', 'precision', 'images', 'pairs', 'arcface_acc_percent', 'mobileface_acc_percent', 'arcface_auc', 'mobileface_auc', 'arcface_tar_at_far1_percent', 'mobileface_tar_at_far1_percent', 'amortized_reconstruction_ms']
    csv_rows = []
    for name, result in results.items():
        meta = result['metadata']; a, m = result['evaluators']['arcface'], result['evaluators']['mobileface']
        csv_rows.append(dict(zip(fields, [name, LABELS[name], meta['role'], meta.get('parameters', 0),
            'AMP FP16 restoration / FP32 recognition' if name.endswith('_amp') else 'FP32/TF32 restoration / FP32 recognition' if name == 'codeformer_tf32' else 'FP32', result['images'], result['pairs'],
            a['acc']*100, m['acc']*100, a['auc'], m['auc'], a['tar_far1']*100, m['tar_far1']*100, meta['mean_amortized_reconstruction_ms']])))
    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    def rows(methods):
        return [[LABELS[name], f'{results[name]["metadata"].get("parameters", 0)/1e6:.3f}',
                 'AMP' if name.endswith('_amp') else 'FP32/TF32' if name == 'codeformer_tf32' else 'FP32',
                 f'{results[name]["evaluators"]["arcface"]["acc"]*100:.2f}',
                 f'{results[name]["evaluators"]["mobileface"]["acc"]*100:.2f}',
                 f'{results[name]["evaluators"]["mobileface"]["auc"]:.5f}'] for name in methods]
    headers = ['Processing', 'Parameters (M)', 'Restoration precision', 'ArcFace accuracy (%)', 'MobileFace accuracy (%)', 'MobileFace AUC']
    data_checks = json.loads((ROOT / 'work/qmul_domain/data_checks.json').read_text())
    selected = json.loads((ROOT / 'work/qmul_domain/domain_full/validation_result.json').read_text())
    old, new = results['ours_mixed']['evaluators'], results['domain_full']['evaluators']
    gate = {'validation_ffhq_guards': selected['passed_ffhq_guards'],
            'selected_an_actual_extra_training_epoch': selected['best_epoch'] > 0,
            'independent_mobileface_auc_improves': new['mobileface']['auc'] >= old['mobileface']['auc'] + .003,
            'independent_mobileface_accuracy_not_worse': new['mobileface']['acc'] >= old['mobileface']['acc'],
            'paired_ffhq_test_guards': all(ff['quality_guards'].values())}
    available = all(gate.values())
    review = {'available_for_comparison': available, 'checkpoint': 'work/qmul_domain/domain_full/best.pt',
              'checkpoint_sha256': sha(ROOT / 'work/qmul_domain/domain_full/best.pt'), 'checks': gate,
              'default_model_unchanged': True, 'experimental': True,
              'warning': 'Experimental adaptation using official QMUL training identities. Recognition scores improve over mixed SR, but paired perceptual detail is worse. No clean camera HR targets were used. Verification metrics do not certify restored identity.'}
    write_json(ROOT / 'work/qmul_domain/review.json', review)
    lines = ['# Same-QMUL model comparison and ablation study', '',
        'Every row in this study was freshly evaluated on the same 9,913 images and all 10,640 verification pairs. '
        'No model is described as better merely because its output looks sharper. This study reports identity verification, not a percentage of an image restored.', '',
        '## Controlled protocol', '',
        '- All SR methods receive identical cached 28 x 28 RGB pixels from the same recorded camera crops, and produce 112 x 112 RGB outputs.',
        '- Both frozen evaluators use the same weights for every method: ArcFace R50 and MobileFaceNet. ArcFace is a training prior in some custom models; MobileFaceNet is excluded from the new adaptation loss and selection.',
        '- Ten stratified folds are fixed with seed 0. Each threshold is calibrated on the other nine folds. All 5,320 positive and 5,320 negative pairs are retained.',
        '- Face restorers receive the same crop, internally resized to 512. No method-specific detector or landmark alignment is added. This can limit face-prior methods on nonfrontal or misaligned camera crops.',
        '- GFPGAN uses documented CUDA AMP FP16 restoration. CodeFormer uses FP32 tensors with TF32-enabled restoration contractions after its AMP attempt failed finite-output checks. All recognizers stay FP32 with TF32 disabled. Eight-image numeric diagnostics are saved; these do not establish global equivalence. Other SR rows use FP32.',
        '- Native bicubic is shown separately: it uses the same images/pairs but bypasses the forced 28-pixel resize. It is a different input pipeline.',
        '- No clean HR reference exists for these QMUL crops. No QMUL PSNR, SSIM, LPIPS or restoration percentage is claimed.', '',
        '## Model comparison (not an architecture ablation)', '']
    lines += markdown_table(headers, rows(MAIN))
    lines += ['', 'Real-ESRGAN compact and the full 23-RRDB x4plus are different models. GFPGAN is version 1.4 at full strength; CodeFormer uses fidelity 1 and AdaIN. '
        'All external checkpoints are official, separately pretrained models; their training data/budgets differ from the custom network. '
        'These results establish performance under this shared input pipeline, not the maximum possible performance of each method under other preprocessing.', '',
        'Native-input context:', ''] + markdown_table(headers, rows(['native_bicubic']))
    lines += ['', '## Fully trained joint architecture ablation', '',
        'The original convolution baseline and original SSC+attention model were both trained from fresh initialization on the same 67,000 FFHQ training images for 30 epochs (31,380 optimizer updates), with the same degradation, L1 objective and validation-PSNR selection protocol. '
        'Their fresh QMUL rows above therefore also provide a fully trained joint ablation: ordinary convolution with both SSC/attention absent versus the original full model. '
        'Removing both mechanisms changes capacity (371,715 versus 479,109 parameters); this comparison does not isolate each mechanism individually. '
        'It is separate from the later perceptual, mixed-quality and camera-adaptation objectives.', '',
        'Training-completion evidence is retained in work/full/training_completion.json. The joint ablation does not show a surveillance-recognition benefit for the original full architecture, even though its paired FFHQ reconstruction scores were higher.', '',
        '## Inference interventions (no retraining)', '',
        'These rows use the same mixed-quality checkpoint and intervene on one component or inference setting. Removing a component can disrupt co-adapted weights, so this table alone is not a from-scratch architecture ablation.', '']
    lines += markdown_table(headers, rows(INTERVENTIONS))
    lines += ['', 'The fixed 50% fidelity blend averages the custom reconstruction with bicubic. Its coefficient was fixed before QMUL evaluation, not chosen by sweeping this test set.', '',
        '## Matched camera-adaptation fine-tuning ablations', '',
        'Four runs start from identical retained weights of the 38-epoch FFHQ mixed-quality checkpoint. Each receives two epochs and 512 optimizer updates per epoch, batch size 64, identical shuffling/augmentation seeds and learning-rate schedule. '
        'The component-removal runs are re-fine-tuned, rather than only disabled at inference. They remain short warm-start ablations, not independently trained 30-epoch architectures.', '',
        'The full camera run and removal runs use 16,384 FFHQ images plus 16,384 actual official QMUL training crops. The FFHQ-only control replaces the camera branch with another 16,384 FFHQ images while retaining identical branch losses, weights, updates and batch size. '
        'The joint objective uses FFHQ pixel/structure loss, a camera observation-preservation loss and small frozen ArcFace feature distillation from the native camera observation. '
        'The enlarged native camera target is still LR data, not clean HR ground truth. This trains fidelity to observed camera information; it does not supervise the true missing facial texture.', '',
        f'The camera subset covers {data_checks["training_identities"]:,} training identities. Validation uses {data_checks["validation_identities"]} separate official-training identities and 512 crops. '
        'Training/validation identities have zero overlap with the verification identities. The archive contains 220,888 training JPEGs across 5,319 identities; the official site lists 220,890, so the exact available count is reported rather than silently claiming that total.', '',
        'Checkpoint selection maximizes held-out-training-identity ArcFace AUC with paired FFHQ validation quality guards. The QMUL test pairs and MobileFaceNet never select a checkpoint. '
        'If a removal candidate never passes those guards, its predeclared final-epoch fallback is evaluated as a failed candidate and marked below.', '']
    lines += markdown_table(headers, rows(DOMAIN_METHODS))
    lines += ['', 'Selection and quality safeguards:', '']
    selection_rows = []
    for name in DOMAIN_METHODS:
        record = json.loads((ROOT / 'work/qmul_domain' / name / 'validation_result.json').read_text())
        selection_rows.append([LABELS[name], record['best_epoch'], 'PASS' if record['passed_ffhq_guards'] else 'FAIL / final-epoch fallback'])
    lines += markdown_table(['Run', 'Selected extra epoch', 'FFHQ validation guard'], selection_rows)
    lines += ['', '## Paired reconstruction tradeoff on FFHQ test images', '',
        'The preselected full camera candidate was also compared on 512 identical FFHQ test images per condition. These are actual paired targets; this table does not describe QMUL restoration fidelity.', '']
    ff_rows = []
    for profile, methods in ff['profiles'].items():
        for method, values in methods.items(): ff_rows.append([profile, LABELS[method], f'{values["psnr_y"]:.4f}', f'{values["ssim_y"]:.5f}'])
    lines += markdown_table(['FFHQ condition', 'Model', 'PSNR-Y', 'SSIM-Y'], ff_rows)
    quality_path = ROOT / 'work/qmul_quality/quality_results.json'
    quality_rows = []
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        for profile, methods in quality['test'].items():
            for method, values in methods.items():
                quality_rows.append([profile, method, f'{values["ssim_y"]:.5f}', f'{values["lpips"]:.5f}'])
        lines += ['', '## Additional perceptual-quality check', '',
            'A later LPIPS check on the same 512 paired FFHQ test images per condition reveals a regression missed by the original PSNR/SSIM guards. '
            'Raw camera adaptation smooths detail: LPIPS is worse than the previous mixed model on both conditions (lower LPIPS is better). '
            'A bounded edge-contrast filter selected on validation partly reduces that regression, but does not recover the earlier perceptual score. '
            'The filter is not promoted to the app. Camera adaptation is therefore an experimental recognition/structure tradeoff, not an across-the-board quality improvement.', '']
        lines += markdown_table(['Paired FFHQ condition', 'Method', 'SSIM-Y', 'LPIPS (lower better)'], quality_rows)
        lines += ['', 'A separate texture-preserving training experiment and its release decision are documented in [the quality improvement report](../qmul_quality/QUALITY_REPORT.md). '
                  'That exploratory loss refinement is separate from the matched architecture ablations above.']
    lines += ['', '## Uncertainty and reproducibility', '',
        'Per-pair scores, decisions, calibrated thresholds, image-order hashes, input-pixel hashes, evaluator hashes, model hashes, exact settings and source hashes are saved. '
        'comparison.csv contains all metrics, including both AUCs and TAR at FAR 1%. paired_accuracy_differences.json reports paired differences from the mixed-quality model.', '',
        'The paired intervals resample fixed pair decisions; they do not account for shared identities/images across pairs, threshold recalibration, training seeds or dataset shift. '
        'Small gains should be treated cautiously. Near-tied key selection and CUDA arithmetic can change thresholds slightly between runtime configurations. '
        'All numbers in these tables come from the same recorded benchmark configuration, rather than mixing earlier runs.', '',
        'This fixed QMUL test cohort has already been evaluated repeatedly. Training identities are disjoint, but these comparisons are not a newly untouched benchmark. '
        'Recognizer pretraining overlap with benchmark identities and external model pretraining overlap are not established. Human recognition has not been measured.', '',
        '## What can improve this project?', '',
        '1. Use the measured camera-domain candidate only if its independent recognition result and paired reconstruction guards support it. The automatic review is in work/qmul_domain/review.json; no default is changed by this study.',
        '2. Preserve usable native crop detail at inference. Avoid shrinking a larger camera crop to 28 pixels operationally; retain the forced resize only when reproducing this controlled benchmark.',
        '3. Train on matched camera conditions using the official training partition, with held-out identities for validation. The new run tests this directly instead of tuning against the test images.',
        '4. If recognition is the goal, investigate a recognizer trained for native surveillance imagery. Better-looking SR is not automatically a better recognition input.',
        '5. For genuine visual-detail recovery, acquire cleaner paired observations or original video with tracked, aligned frames. Arbitrary still crops sharing a test identity must not be fused using ground-truth labels.',
        '6. Do not increase a generative restoration strength solely to make a face sharper; its invented details can change the person. Choose model settings using held-out training validation, then evaluate once under a frozen protocol.', '',
        '## Figures and artifacts', '',
        '- [External model visual comparison](external_comparison.png)',
        '- [Inference interventions](inference_ablations.png)',
        '- [Camera-domain training ablations](domain_ablations.png)',
        '- [FFHQ heavy-input tradeoff](domain_ffhq_standard28.png)',
        '- [FFHQ clean-downsampling tradeoff](domain_ffhq_clean28.png)',
        '- [Full metrics CSV](comparison.csv)',
        '- [Portable presentation](study_presentation.html)', '',
        'Sources: [QMUL official protocol](https://qmul-survface.github.io/protocols.html), '
        '[GFPGAN official implementation](https://github.com/TencentARC/GFPGAN), '
        '[CodeFormer official implementation](https://github.com/sczhou/CodeFormer), '
        '[Real-ESRGAN official implementation](https://github.com/xinntao/Real-ESRGAN).', '']
    grid(['bicubic', 'ours_mixed', 'realesrgan_x4plus', 'gfpgan_amp', 'codeformer_tf32'], OUT / 'external_comparison.png', 'Same camera inputs: external and custom restoration', manifest)
    grid(INTERVENTIONS, OUT / 'inference_ablations.png', 'Same checkpoint: inference interventions (no retraining)', manifest)
    grid(['ours_mixed', *DOMAIN_METHODS], OUT / 'domain_ablations.png', 'Matched two-epoch adaptation and component-removal pilots', manifest)
    (OUT / 'STUDY.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(14, 6), layout='constrained'); x = np.arange(len(MAIN))
    for net, delta in [('arcface', -.2), ('mobileface', .2)]:
        axis.bar(x + delta, [results[name]['evaluators'][net]['acc'] * 100 for name in MAIN], width=.4, label=net)
    axis.set_xticks(x, [LABELS[name] for name in MAIN], rotation=35, ha='right'); axis.set_ylim(0, 100)
    axis.set_ylabel('Verification accuracy (%)'); axis.set_title('Same 10,640 QMUL pairs | same LR28 input | reconstruction is not an accuracy percentage')
    axis.legend(); axis.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'verification_comparison.png', dpi=150); fig.savefig(OUT / 'verification_comparison.pdf'); plt.close(fig)
    parts = ['<!doctype html><meta charset="utf-8"><title>QMUL comparison and ablation study</title>',
        '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f2f3ed;color:#18271e}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border:1px solid #bac5b7;text-align:left}img{width:100%;height:auto}h2{margin-top:2em}@media print{body{background:white}img{break-inside:avoid}h2{break-after:avoid}}</style>',
        '<h1>Same-QMUL comparison and ablation study</h1><p>9,913 identical verification images, all 10,640 pairs, fixed folds and two frozen recognizers. Verification accuracy is not the percentage of an image restored. No paired clean HR target exists for these camera crops.</p>']
    for title, group in [('Model comparison', MAIN), ('Inference interventions (no retraining)', INTERVENTIONS), ('Matched two-epoch camera-adaptation ablations', DOMAIN_METHODS)]:
        parts.append('<h2>' + title + '</h2><table><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr>')
        for row in rows(group): parts.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in row) + '</tr>')
        parts.append('</table>')
        if group == MAIN:
            parts.append('<p>The original convolution baseline and original full SSC+attention model also form a fully trained joint architecture ablation: both trained from scratch on 67,000 FFHQ images for 30 epochs and 31,380 updates with the same L1 objective. Capacity differs, and this does not isolate the two mechanisms individually.</p>')
    parts.append('<h2>Native-input context (different preprocessing)</h2><table><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr>')
    for row in rows(['native_bicubic']): parts.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in row) + '</tr>')
    parts.append('</table>')
    parts += ['<p>AMP means FP16 restoration with FP32 weights/evaluators. CodeFormer uses FP32 tensors with TF32 restoration contractions after AMP failed finite-output checks; evaluation stays FP32/TF32-disabled. External pretraining and model budgets differ. No per-model landmark alignment was applied. Native bicubic uses a different native-input pipeline and is documented separately in STUDY.md.</p>',
              '<p>The adaptation pilots start from shared retained FFHQ-pretrained weights; these are short fine-tuning ablations, not training from scratch. Camera training/validation identities are disjoint from verification identities. Native enlarged camera observations are not clean HR targets.</p>']
    if quality_rows:
        parts.append('<h2>Perceptual quality tradeoff</h2><p>The additional paired FFHQ LPIPS check reveals detail loss despite the small SSIM gains. Lower LPIPS is better. Validation-selected edge contrast partly reduces this regression and is not promoted. Raw camera adaptation remains an experimental recognition/structure tradeoff.</p><table><tr><th>Condition</th><th>Method</th><th>SSIM-Y</th><th>LPIPS</th></tr>')
        for row in quality_rows: parts.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in row) + '</tr>')
        parts.append('</table>')
    figures = ['verification_comparison', 'external_comparison', 'inference_ablations', 'domain_ablations', 'domain_ffhq_standard28', 'domain_ffhq_clean28']
    for name in figures:
        data = base64.b64encode((OUT / (name + '.png')).read_bytes()).decode('ascii')
        parts.append(f'<h2>{html.escape(name.replace("_", " ").title())}</h2><img alt="{html.escape(name)}" src="data:image/png;base64,{data}">')
    parts.append('<p>Detailed settings, paired differences, quality safeguards and limitations are in STUDY.md and comparison.csv. No default model is changed. Repeated-test comparisons do not establish a fresh generalization benchmark or 90% identity recovery.</p>')
    (OUT / 'study_presentation.html').write_text('\n'.join(parts), encoding='utf-8')
    print('STUDY WRITTEN', OUT, 'camera candidate available:', available, flush=True)


if __name__ == '__main__': main()

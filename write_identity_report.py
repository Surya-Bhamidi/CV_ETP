"""Write a presentation report from measured experiment files, without rerunning models."""
import argparse
import html
import json

from refine_model import write_json
from run_project import ROOT


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='mixed_quality_v1')
    args = parser.parse_args()
    folder = ROOT / 'work/identity_investigation' / args.run
    metrics = json.loads((folder / 'heldout_reconstruction.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    validation = json.loads((folder / 'validation_result.json').read_text())
    qmul = json.loads((folder / 'qmul_verification.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for profile in ['standard28', 'mild28', 'clean28']:
        series = [validation['baseline'][profile]] + [row['validation'][profile] for row in history]
        for axis, key, label in zip(axes, ['psnr_y', 'ssim_y'], ['Validation PSNR-Y (dB)', 'Validation SSIM-Y']):
            axis.plot(range(len(series)), [point[key] for point in series], marker='o', label=profile)
            axis.set_xlabel('Additional fine-tune epoch (0 = current checkpoint)')
            axis.set_ylabel(label); axis.grid(alpha=.2); axis.legend()
    fig.savefig(folder / 'training_curves.png', dpi=160); plt.close(fig)
    profiles = metrics['profiles']
    current, candidate = profiles['standard28']['current'], profiles['standard28']['mixed_quality']
    checks = {
        'clean_input_structure_improves': profiles['clean28']['mixed_quality']['ssim_y'] > profiles['clean28']['current']['ssim_y'],
        'clean_input_independent_feature_similarity_improves': profiles['clean28']['mixed_quality']['mobileface_cos'] > profiles['clean28']['current']['mobileface_cos'],
        'standard_test_psnr_guard': candidate['psnr_y'] >= current['psnr_y'] - .2,
        'standard_test_ssim_guard': candidate['ssim_y'] >= current['ssim_y'] - .003,
        'general_validation_guard': metrics['general_validation_guard_passed'],
        'qmul_arcface_auc_not_worse_than_current': qmul['conditions']['mixed_quality']['auc'] >= qmul['conditions']['current']['auc'],
        'qmul_arcface_auc_not_worse_than_bicubic': qmul['conditions']['mixed_quality']['auc'] >= qmul['conditions']['lr_bicubic']['auc'],
        'qmul_mobileface_auc_not_worse_than_current': qmul['mobileface_conditions']['mixed_quality']['auc'] >= qmul['mobileface_conditions']['current']['auc'],
        'qmul_mobileface_auc_not_worse_than_bicubic': qmul['mobileface_conditions']['mixed_quality']['auc'] >= qmul['mobileface_conditions']['lr_bicubic']['auc'],
    }
    warning = ('Experimental mixed-quality FFHQ model. Cleaner-input reconstruction improved. '
               + ('Real QMUL verification still trails bicubic enlargement. ' if not checks['qmul_mobileface_auc_not_worse_than_bicubic'] else '')
               + ('' if checks['general_validation_guard'] else 'The general validation guard failed; use only as a cleaner-input comparison. ')
               + 'Facial details can still be wrong. Compare with Original; recognizable identity is not guaranteed.')
    review = {'checkpoint': metrics['checkpoint'], 'checkpoint_sha256': metrics['checkpoint_sha256'],
              'available_for_comparison': checks['clean_input_structure_improves'] and checks['clean_input_independent_feature_similarity_improves'],
              'default_model': 'detail', 'experimental': True, 'checks': checks, 'warning': warning,
              'scope': 'Paired synthetic FFHQ reconstruction; real QMUL verification diagnostic, no identity certification'}
    write_json(ROOT / 'work/identity_investigation/candidate_review.json', review)
    lines = ['# Face reconstruction and identity: measured results', '',
        'The mixed-quality candidate was trained and evaluated to investigate changed faces and missing detail. '
        'It is an experimental selectable model. The existing reconstruction model remains the default.', '',
        '## Which datasets are actually used?', '',
        '- FFHQ128: 67,000 training, 1,000 validation and 2,000 test images. Native cached images are 128 x 128; clean targets are 112 x 112.',
        '- The original and previous refinement checkpoints trained on FFHQ, not QMUL. This new run also trains only on the same FFHQ training split.',
        '- QMUL-SurvFace: real surveillance evaluation only. The local cache contains the evaluation folders, not a QMUL training set.',
        '- QMUL native surveillance crops have no paired clean high-resolution originals in this protocol. Paired PSNR, SSIM or a percentage of exact detail restored cannot be measured for them.', '',
        '## What was changed?', '',
        f'{manifest["epochs"]} additional epochs on {manifest["train_images"]:,} FFHQ training images, starting from the existing 35-epoch reconstruction checkpoint. '
        f'Additional training took {sum(row["seconds"] for row in history) / 60:.1f} minutes including validation. The same 479,109-parameter x4 architecture is retained.', '',
        'Training inputs mix 50% existing heavy degradation, 25% mild degradation and 25% bicubic downsampling only. '
        'Losses combine pixel L1, structural similarity, multiscale detail, VGG and a small frozen ArcFace feature loss. '
        'The independent FFHQ feature diagnostic uses MobileFaceNet, which was not used in this new objective or checkpoint selection. '
        'Both recognizers have WebFace600K pretraining; independent architecture does not establish independent pretraining data.', '',
        f'Validation: {manifest["validation_images_per_profile"]} images per input-quality profile. '
        'The preset general gate requires heavy-input PSNR within 0.2 dB and SSIM within 0.003 of the current model. '
        f'General validation gate: **{"PASS" if metrics["general_validation_guard_passed"] else "FAIL"}**. '
        f'Candidate choice: {metrics["selection"]}. No test or supplied image was used for training or epoch selection.', '',
        '## Same-input held-out reconstruction comparisons', '',
        'Higher PSNR/SSIM is better. Cosine similarity is a diagnostic feature distance, not a probability, identity verification accuracy, or recovery percentage. '
        'All methods within each condition receive identical LR pixels. The clean reference is only a training target or evaluation reference.', '']
    labels = {'standard28': 'Heavy benchmark, 28 -> 112 (x4)', 'mild28': 'Mild, 28 -> 112 (x4)',
              'clean28': 'Downsampling only, 28 -> 112 (x4)', 'clean56': 'Downsampling only, 56 -> 112 (effective x2)'}
    rows = []
    for profile, methods in profiles.items():
        count = metrics['standard_test_images'] if profile == 'standard28' else metrics['severity_test_images_per_profile']
        for method in ['bicubic', 'current', 'mixed_quality']:
            v = methods[method]
            rows.append([labels[profile], count, method, f'{v["psnr_y"]:.3f}', f'{v["ssim_y"]:.4f}', f'{v["mobileface_cos"]:.4f}'])
    lines += table(['Condition', 'Test images', 'Method', 'PSNR-Y (dB)', 'SSIM-Y', 'MobileFace cosine'], rows)
    lines += ['', 'The 56-pixel experiment is an easier, different task: it retains four times as many input pixels. '
        'For the x4 network, 56 -> 224 inference is antialiased down to the native 112 target for scoring. '
        'The clean reference is never upsampled for this evaluation. Its numbers must not be reported as x4 restoration of 28-pixel surveillance inputs.', '',
        '## Can we claim 85-90% restored?', '',
        '**No universal restoration percentage has been established.** SSIM 0.85 means structural similarity 0.85 under the stated metric; '
        'it does not mean 85% of the true facial details or identity were recovered. Face verification accuracy measures a different task.', '']
    rows = []
    for profile, methods in metrics['ssim_threshold_counts'].items():
        v = methods['mixed_quality']
        rows.append([labels[profile], v['images'], v['ssim_at_least_0_85'], v['ssim_at_least_0_90']])
    lines += table(['Candidate input condition', 'Images', 'SSIM >= 0.85', 'SSIM >= 0.90'], rows)
    lines += ['', 'These are counts over the stated test subsets, not cherry-picked examples and not identity recovery rates.', '',
        '## How good is real QMUL performance?', '',
        f'Fresh evaluation: {qmul["pairs"]:,} verification pairs using {qmul["images"]:,} unique images. '
        'ArcFace is a frozen evaluator and also a training prior for the new candidate, so gains on it alone do not prove general identity preservation. '
        'MobileFaceNet provides an additional evaluator excluded from this training objective and selection. '
        'Verification uses cross-fold threshold calibration from the existing protocol. No new identification Rank-1 claim is made.', '']
    lines += ['ArcFace evaluator:', '']
    lines += table(['Processing', 'Verification accuracy (%)', 'AUC', 'TAR at FAR 1% (%)'], [
        [name, f'{v["acc"] * 100:.2f}', f'{v["auc"]:.5f}', f'{v["tar_far1"] * 100:.2f}'] for name, v in qmul['conditions'].items()])
    lines += ['', 'Independent MobileFaceNet evaluator:', '']
    lines += table(['Processing', 'Verification accuracy (%)', 'AUC', 'TAR at FAR 1% (%)'], [
        [name, f'{v["acc"] * 100:.2f}', f'{v["auc"]:.5f}', f'{v["tar_far1"] * 100:.2f}'] for name, v in qmul['mobileface_conditions'].items()])
    lines += ['', 'Native bicubic enlarges the native crop directly. Other conditions share the protocol-resized 28 x 28 crop. '
        'They are different input pipelines; protocol resizing is a controlled experiment, not a requirement for real uploads.', '',
        'For context, the archived original pixel model achieved 89.18% LFW verification and 68.04% QMUL verification; '
        'bicubic achieved 94.07% and 69.46%, respectively. The original QMUL closed-set Rank-1 was only 10.58%. '
        'A verification percentage should never be described as the fraction of an image restored.', '',
        '## Why does the person change?', '',
        'Face outline is low-frequency structure. Eye shape, eyelids, mouth and other distinguishing details occupy fewer pixels and can be destroyed by downsampling, '
        'blur and compression. A model can preserve outline while estimating the wrong distinguishing features. '
        'The validation screen additionally found that the heavy-degradation-trained model oversmoothed cleaner inputs: '
        'the cleaner-input MobileFace feature similarity was lower than bicubic. This is a measurable training/input mismatch, not merely a display-size issue.', '',
        'GFPGAN and CodeFormer synthesize details from learned facial priors; sharper results can change identity. '
        'Their official documentation describes identity/fidelity tradeoffs. Previously recorded CodeFormer, alignment and compact Real-ESRGAN trials did not solve this project\'s problem. '
        'Adding sharpening, larger output dimensions or more generative strength cannot establish the truth of missing features.', '',
        '## Practical ways to improve recognizable detail', '',
        '1. Preserve the largest native face crop available. Do not shrink a usable 56/112-pixel face to 28 pixels before restoring it. Upsizing an already tiny image adds no observations.',
        '2. If video exists, use several original frames: select sharper frames, align motion and test a multi-frame SR method. Different frames can contain additional real observations.',
        '3. Use degradation matched to the input camera. The mixed-quality experiment addresses one mismatch; real camera blur, motion, noise and compression still need matched training data.',
        '4. For further domain training, obtain the official QMUL training split or genuinely paired surveillance/clean data. Do not fine-tune on this QMUL test set.',
        '5. Keep pixel/structure constraints when adding identity and perceptual losses. Check an independent recognizer plus blind human same-person assessments; feature similarity alone is insufficient.',
        '6. Treat strong GAN/diffusion output as estimated appearance. Reducing strength may reduce changed features, but it cannot guarantee identity. Validate alignment rather than assuming landmarks from tiny faces are reliable.', '',
        '## Figures to show', '',
        '- [Training images: diagnostic comparison](train_comparison.png)',
        '- [Validation training curves](training_curves.png)',
        '- [Validation images: comparison](validation_comparison.png)',
        '- [Heavy-input test comparison](test_standard28.png)',
        '- [Mild-input test comparison](test_mild28.png)',
        '- [Downsampling-only x4 test comparison](test_clean28.png)',
        '- [Higher-information effective x2 comparison](test_clean56.png)',
        '- [Actual QMUL crops and outputs](qmul_real_examples.png)',
        '- [Existing supplied input and reconstruction](supplied_input_comparison.png)',
        '- [Reconstructed PNG of supplied input](supplied_input_reconstructed.png)', '',
        'All sheets show the fixed first six examples, except QMUL which uses uniformly spaced filenames. No success-based image selection was performed.', '',
        '## How to demonstrate it', '',
        'Start `Open_Comparison_UI.cmd`. Select **Mixed-quality SR (experimental)**. Choose **FFHQ validation** or **FFHQ test**, '
        'then compare **Benchmark**, **Mild** and **Downsampling only** input settings. Use **Original** and **Side by side**. '
        'Say which degradation setting is shown. For an already low-resolution upload, use **Reconstruct this image x4** so it is not degraded again.', '',
        '## Evidence and limits', '',
        '- Training/validation/test image indices are disjoint. Identity-disjointness, near-duplicate exclusion and recognizer pretraining overlap are not established.',
        '- The fixed FFHQ test split has been evaluated by previous experiments. These are repeated diagnostic comparisons, not a new untouched benchmark.',
        '- Human recognizability and exact identity recovery have not been measured. The candidate stays experimental and does not replace the default.',
        '- Source/checkpoint hashes, histories, per-image metrics, split usage, selection and protocol metadata are preserved alongside this report.', '',
        'Sources: [QMUL official dataset](https://qmul-survface.github.io/), '
        '[GFPGAN official implementation](https://github.com/TencentARC/GFPGAN), '
        '[CodeFormer official implementation](https://github.com/sczhou/CodeFormer), '
        '[identity-preserving face SR research](https://openaccess.thecvf.com/content_ECCV_2018/html/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.html), '
        '[BasicVSR video SR research](https://ckkelvinchan.github.io/projects/BasicVSR/).', '']
    (folder / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    # A portable local HTML handout with embedded figures for presentation.
    import base64
    parts = ['<!doctype html><meta charset="utf-8"><title>Face reconstruction results</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f5f5f0;color:#17251c}img{width:100%;height:auto}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:left}h2{margin-top:2em}</style>',
        '<h1>Face reconstruction: measured comparisons</h1><p>Experimental mixed-quality refinement. No claim of 85–90% identity recovery.</p>',
        '<p>Custom models train on FFHQ (67,000 training, 1,000 validation, 2,000 test images). QMUL is used only for real surveillance evaluation; it has no paired clean HR reference in this protocol.</p>',
        '<p>The full text and numerical protocol are in RESULTS.md in the same folder.</p>']
    parts.append('<table><thead><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in ['Input condition', 'N', 'Method', 'PSNR', 'SSIM', 'Feature cosine']) + '</tr></thead><tbody>')
    for profile, methods in profiles.items():
        count = metrics['standard_test_images'] if profile == 'standard28' else metrics['severity_test_images_per_profile']
        for method in ['bicubic', 'current', 'mixed_quality']:
            v = methods[method]
            cells = [labels[profile], str(count), method, f'{v["psnr_y"]:.3f}', f'{v["ssim_y"]:.4f}', f'{v["mobileface_cos"]:.4f}']
            parts.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    parts.append('</tbody></table><p>SSIM and feature cosine are not recovery percentages. The clean56 condition is effective 2x; other conditions are 4x.</p>')
    parts.append('<h2>Real QMUL verification</h2><p>10,640 verification pairs. The new model improves over the current reconstruction, but remains below bicubic with both recognizers. This does not establish solved identity recovery.</p>')
    parts.append('<table><tr><th>Evaluator</th><th>Processing</th><th>Verification accuracy (%)</th><th>AUC</th></tr>')
    for net, conditions in [('ArcFace (training prior)', qmul['conditions']), ('MobileFaceNet (excluded from loss)', qmul['mobileface_conditions'])]:
        for method, value in conditions.items():
            parts.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in [net, method, f'{value["acc"]*100:.2f}', f'{value["auc"]:.5f}']) + '</tr>')
    parts.append('</table>')
    figure_titles = {'training_curves': 'Validation curves across extra training epochs',
                    'train_comparison': 'Training image diagnostic (not test evidence)',
                    'validation_comparison': 'Validation comparison (used for model selection)',
                    'test_standard28': 'Heavy degradation: 28 to 112, 4x',
                    'test_mild28': 'Mild degradation: 28 to 112, 4x',
                    'test_clean28': 'Downsampling only: 28 to 112, 4x',
                    'test_clean56': 'Higher-information input: 56 to 112, effective 2x',
                    'qmul_real_examples': 'Actual QMUL surveillance crops (no paired clean original)',
                    'supplied_input_comparison': 'The existing supplied image, reconstructed again'}
    for name in ['training_curves', 'train_comparison', 'validation_comparison', 'test_standard28', 'test_mild28', 'test_clean28', 'test_clean56', 'qmul_real_examples', 'supplied_input_comparison']:
        path = folder / (name + '.png')
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            parts.append(f'<h2>{html.escape(figure_titles[name])}</h2><img alt="{html.escape(name)}" src="data:image/png;base64,{encoded}">')
    (folder / 'presentation.html').write_text('\n'.join(parts), encoding='utf-8')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()

"""Build a teacher-facing audit and surveillance explanation from measured files."""
import base64
import html
import json

import torch

from audit_attention import BASE, OUT
from refine_model import write_json


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]


def main():
    audit = json.loads((OUT / 'audit.json').read_text())
    interventions = json.loads((OUT / 'validation_attention.json').read_text())
    ensemble = json.loads((OUT / 'ensemble_results.json').read_text())
    qmul = json.loads((OUT / 'qmul_ensemble_verification.json').read_text())
    example = json.loads((OUT / 'qmul_example.json').read_text())
    base = torch.load(BASE, map_location='cpu', weights_only=True)['model']
    pilot_rows, pilot_checks = [], {}
    for name in ['control', 'context_global', 'context_quadrant']:
        folder = OUT / ('pilot_' + name)
        manifest = json.loads((folder / 'manifest.json').read_text())
        result = json.loads((folder / 'validation_result.json').read_text())
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)['model']
        unchanged = all(torch.equal(value, state[key]) for key, value in base.items() if '.attn.' not in key)
        changed = sum(not torch.equal(value, state[key]) for key, value in base.items() if '.attn.' in key)
        assert unchanged and changed > 0
        pilot_checks[name] = {'all_non_attention_weights_unchanged': unchanged, 'changed_attention_tensors': changed,
                             'epochs_completed': result['epochs_completed'], 'best_epoch': result['best_epoch']}
        metrics = result['best_validation']
        pilot_rows.append([name, manifest['parameters'], manifest['trainable_parameters'], result['best_epoch'],
                           *[f'{metrics[p]["ssim_y"]:.6f}' for p in ['standard28', 'mild28', 'clean28']]])
    write_json(OUT / 'controlled_training_checks.json', pilot_checks)
    selected = ensemble['selected']
    improved = selected == 'flip2' and ensemble['heldout_identity_guard'] and all(
        ensemble['test'][p]['flip2']['ssim_y'] > ensemble['test'][p]['single']['ssim_y'] for p in ensemble['test'])
    checks = {'selected_on_validation': selected == 'flip2', 'heldout_ffhq_structure_improves': improved,
              'heldout_independent_feature_guard': ensemble['heldout_identity_guard'],
              'qmul_arcface_auc_not_worse_than_single': qmul['conditions']['flip2']['auc'] >= qmul['conditions']['single_pass']['auc'],
              'qmul_mobileface_auc_not_worse_than_single': qmul['mobileface_conditions']['flip2']['auc'] >= qmul['mobileface_conditions']['single_pass']['auc'],
              'qmul_arcface_accuracy_not_worse_than_single': qmul['conditions']['flip2']['acc'] >= qmul['conditions']['single_pass']['acc'],
              'qmul_mobileface_accuracy_not_worse_than_single': qmul['mobileface_conditions']['flip2']['acc'] >= qmul['mobileface_conditions']['single_pass']['acc']}
    warning = ('Experimental two-pass averaging of the same mixed-quality attention model; extra inference time, no extra model parameters. '
               'FFHQ structure improved slightly. QMUL verification accuracy was mixed across recognizers and remains below bicubic; facial identity is not guaranteed.')
    review = {'available_for_comparison': improved, 'checkpoint_sha256': audit['checkpoint_sha256'],
              'inference_passes': 2, 'additional_parameters': 0, 'checks': checks, 'warning': warning,
              'default_model_unchanged': True}
    write_json(OUT / 'ensemble_review.json', review)
    params = audit['parameters']
    lines = ['# Attention verification, reconstruction limits and real QMUL examples', '',
        'The trained custom model contains six deformable-style sparse sampling convolution modules and six adaptive spatial-key-sparse linear attention modules. '
        'The mechanism is present and mathematically tested. This is a custom implementation of these ideas; exact reproduction of a specific published architecture is not established.', '',
        '## What attention is actually implemented?', '',
        '- Each pixel receives an importance score from a learned 1x1 convolution/GELU/1x1 convolution scorer.',
        '- Top-k spatial selection keeps 50% of locations as keys: 392 of 784 at 28 x 28 input. All 784 locations remain queries.',
        '- Which positions survive depends on image content. The 50% count is fixed, not an input-adaptive computational budget.',
        '- Queries and keys use the positive feature map ELU(x)+1. A normalized kernel dot product aggregates values.',
        '- The associative contraction forms a 48 x 48 key/value summary and avoids a 784 x 784 pairwise attention matrix. Its contraction cost is O(N*C*C), linear in spatial N at fixed channel width C. Scoring, top-k selection, projections and sampling have their own costs.',
        '- Hard selection uses a straight-through sigmoid surrogate during training. Scorer and offset gradients were checked.',
        '- Spatial sparsity means selected spatial key positions. The original implementation still multiplies dense tensors after masking; it does not automatically skip those positions computationally.', '',
        'A separate backward check on the actual trained checkpoint found finite, nonzero importance-scorer and sampling-offset gradients in all six blocks. '
        'Some scorer outputs are nearly saturated and their gradient magnitudes are small. The binary key maps show rankings, not evidence that attention is semantically focused on the correct identity features. '
        'See trained_gradient_checks.json and qmul_attention_statistics.json for the measurements. No optimizer step was performed in that check.', '',
        '```text\nLR RGB -> 3x3 feature convolution\n       -> [GroupNorm -> sparse sampling conv -> residual add\n           GroupNorm -> adaptive sparse linear attention -> residual add\n           GroupNorm -> pointwise FFN -> residual add] x 6\n       -> feature convolution + long skip\n       -> PixelShuffle x2 -> PixelShuffle x2 -> RGB\n       -> add bicubic input enlargement -> reconstructed image\n```', '',
        '## Is sparse sampling convolution really present?', '',
        'Yes. Each block predicts 18 offsets for 9 sampling points and calls torchvision deform_conv2d. Offsets are tanh-bounded to +/-2 around a 3x3 grid, '
        'giving support inside a 7x7 region of the feature map. Zero-offset output was checked against a normal 3x3 convolution, and offset gradients are nonzero.', '',
        'This is deformable-style learned sampling. The repository also contains an ASCConv variant that selects feature-correlated positions, but the active trained checkpoints do not use that variant. '
        'Do not describe the trained model as an exact ASC-SRN reproduction. GroupNorm(1,C), hard top-k selection and the stated crop sizes are implementation choices in this project.', '',
        '## Is it lightweight?', '']
    lines += table(['Measured property', 'Value'], [
        ['Total trained SR parameters', f'{params:,}'], ['Attention parameters (all six blocks)', f'{audit["attention_parameters"]:,}'],
        ['Sparse-sampling parameters (all six blocks)', f'{audit["ssc_parameters"]:,}'],
        ['Raw FP32 weight storage', f'{params * 4 / 1e6:.3f} MB (not total runtime memory)'],
        ['Feature width / blocks / scale', '48 / 6 / x4'],
        ['Warm batch-1 CUDA median, 28 -> 112', f'{audit["latency"]["28"]["original_model"]["median_ms"]:.2f} ms'],
        ['Measured GPU', audit['hardware']]])
    lines += ['', 'Small weight count supports a lightweight parameter claim. It does not prove phone/CPU real-time speed, low total application RAM, or a complete FLOP count. '
        'Timing uses 10 warm-up calls and 40 synchronized timed calls, excludes decoding/UI/recognition, and is specific to this GPU. '
        'The comparison app can also load optional GFPGAN; that external model and PyTorch runtime memory are separate from the small custom SR network. ArcFace/VGG are training/evaluation tools, not required for custom SR inference.', '',
        'A gathered-key implementation was tested. It matches explicit mathematics in float64; floating-point reduction order and later hard top-k boundaries can introduce small complete-network differences. '
        f'The measured maximum 8-bit output difference on two validation images was {audit["checks"]["whole_trained_network_png_max_difference_levels"]:.0f} level. '
        'It was slower at 28-pixel input on this GPU, so the current inference path was retained. Sparse mathematical support is not a measured speedup by itself.', '',
        '## Did changing attention improve reconstruction?', '',
        'First, seven inference interventions were evaluated on the same 256 validation images per quality profile. These are interventions on one trained checkpoint, '
        'not separately trained architecture ablations. More keys or disabling attention did not improve overall quality.', '']
    lines += table(['Inference intervention', 'Heavy SSIM', 'Mild SSIM', 'Clean-downsample SSIM'], [
        [name, *[f'{values[p]["ssim_y"]:.6f}' for p in ['standard28', 'mild28', 'clean28']]]
        for name, values in interventions['policies'].items()])
    lines += ['', 'Then three controlled two-epoch pilots used identical 8,192 FFHQ training images, augmentations, losses and learning rates. '
        'Only attention parameters trained. The context variants add a zero-initialized depthwise 3x3 spatial scorer residual; the quadrant variant also ensures regional key coverage. '
        'All non-attention weights were verified unchanged. Model selection used validation, never QMUL or FFHQ test images.', '']
    lines += table(['Pilot', 'Parameters', 'Trainable', 'Chosen extra epoch', 'Heavy SSIM', 'Mild SSIM', 'Clean SSIM'], pilot_rows)
    lines += ['', 'The attention extensions did not demonstrate a meaningful balanced improvement over the control. They were not promoted. '
        'These limited pilots do not prove that attention is globally optimal or that all architecture changes would fail.', '',
        '## The small improvement that did pass', '',
        'Horizontal-flip averaging was selected on validation and then tested on 512 held-out images per profile. '
        'It uses the same weights twice, reverses the flip of the second prediction and averages the two outputs. '
        'No clean reference is supplied during either prediction. Eight rotation/reflection passes performed worse and were rejected.', '']
    rows = []
    for profile, methods in ensemble['test'].items():
        for name, values in methods.items():
            rows.append([profile, name, f'{values["psnr_y"]:.4f}', f'{values["ssim_y"]:.6f}', f'{values["mobileface_cos"]:.6f}'])
    lines += table(['Input', 'Method', 'PSNR-Y', 'SSIM-Y', 'Independent MobileFace cosine'], rows)
    lines += ['', 'The heavy-input figures here use the same first 512 images for both methods. Do not compare those subset averages directly with a previous 2,000-image mean. '
        'Two-pass inference adds computation, not parameters. Feature cosine and SSIM are not identity recovery percentages.', '',
        '## How does QMUL accuracy work without a clean original?', '',
        'QMUL supplies native surveillance face crops and identity/pair annotations. It does not supply matching clean HR targets for these crops. '
        'The benchmark has 5,320 labelled same-person pairs and 5,320 labelled different-person pairs in this project\'s verification protocol.', '',
        '```text\nRecorded face A -> SR -> frozen recognizer -> embedding A\nRecorded face B -> SR -> frozen recognizer -> embedding B\nCosine similarity(A,B) -> compare with threshold -> same or different\nCompare that decision with the dataset pair label -> correct or incorrect\n```', '',
        'A threshold is calibrated using nine folds and evaluated on the remaining fold, repeated across ten folds. The mean decision accuracy is reported. '
        'No clean image is needed to know whether a labelled pair is the same person. This is recognition verification, not image reconstruction accuracy. '
        'The project adds cross-fold verification accuracy; do not confuse it with the official open-set identification protocol or its Rank-1/TPIR results.', '',
        f'Fresh full check: {qmul["pairs"]:,} pairs, {qmul["images"]:,} unique images. The original single-pass mixed-quality result in this run is '
        f'{qmul["conditions"]["single_pass"]["acc"]*100:.2f}%, or approximately {round(qmul["conditions"]["single_pass"]["acc"]*qmul["pairs"]):,} correct pair decisions out of {qmul["pairs"]:,}. '
        'It does not mean that percentage of each face was recovered.', '']
    for title, key in [('ArcFace (also a training prior)', 'conditions'), ('MobileFaceNet (excluded from the loss)', 'mobileface_conditions')]:
        lines += [title, '']
        lines += table(['Processing', 'Verification accuracy (%)', 'AUC', 'TAR at FAR 1% (%)'], [
            [name, f'{v["acc"]*100:.2f}', f'{v["auc"]:.5f}', f'{v["tar_far1"]*100:.2f}'] for name, v in qmul[key].items()])
        lines += ['']
    lines += ['The verification pipeline resizes each input to 28 x 28 for controlled comparisons. '
        'The new interactive native QMUL gallery keeps the crop\'s original size and aspect ratio; its displayed input pipeline is different. '
        'Neither view has a clean reference, and no per-image PSNR/SSIM or restoration percentage is reported.', '',
        'Two-pass QMUL accuracy is mixed: ArcFace accuracy decreased slightly while MobileFaceNet accuracy increased slightly; both remain below bicubic. '
        'AUC improved slightly for both. The single-pass rerun also differs slightly from earlier recorded accuracy (with nearly unchanged AUC); '
        'CUDA reductions, near-tied hard key selection and calibration-threshold sensitivity prevent assuming bitwise repeatability across runtime settings. '
        'The side-by-side numbers in this table come from the same run. No statistically established QMUL accuracy gain or 90% result is claimed.', '',
        '## Real camera example', '',
        f'File: `{example["file"]}`. Recorded crop: **27 x 32 pixels**. Native x4 output: **108 x 128**. '
        'This is a camera-derived face crop, not a full camera frame. It was selected as the first example in a previously fixed, uniformly spaced filename list, not for successful reconstruction. '
        'The exported native JPEG is a byte-for-byte copy of the dataset source; hashes are in qmul_example.json.', '',
        '![Real QMUL crop and reconstruction](qmul_one_example.png)', '',
        '![Actual learned key-selection masks on that crop](qmul_attention_masks.png)', '',
        'White mask pixels mark selected keys, not eye/face landmarks, explanations of identity, or verified recovered details.', '',
        '![Labelled same-person pair](qmul_same_person_pair.png)', '',
        '![Labelled different-person pair](qmul_different_people_pair.png)', '',
        'The pair labels come from the dataset annotations. Displayed cosine values are feature similarity, not probabilities and not calibrated single-pair accuracy.', '',
        '## Can this reach 90%?', '',
        '**90% true detail or identity recovery has not been demonstrated.** If the target is SSIM >= 0.90, it needs an explicit paired test and input-quality condition. '
        'If the target is >=90% QMUL verification, it needs correct decisions on at least 90% of the labelled pairs; that is a different objective.', '',
        'The two-pass clean 28 -> 112 test mean is about 0.8545, not 0.90. Previous clean 56 -> 112 effective x2 experiments exceeded 0.95 SSIM, '
        'but those inputs retained more real information and were a different, easier task. A larger blurry native CCTV crop is also not equivalent to that clean synthetic test.', '',
        'The next substantial opportunities are better native capture or multiple aligned video frames, training on genuinely matched camera degradations and using the official surveillance training identities for domain work. '
        'Reusing QMUL test images for training would invalidate the benchmark. Stronger generative detail can look sharper while changing the person; attention changes cannot establish the truth of detail absent from the observations.', '',
        '## Demonstrate it', '',
        '- In the local app, select **QMUL real camera crops (no HR target)** to view six actual native crops. Select crop 02 to see a larger 59 x 80 example.',
        '- Select **Mixed-quality SR + flip averaging** for the measured two-pass option. The default model is unchanged.',
        '- Original and PSNR/SSIM stay unavailable on QMUL. FFHQ validation/test examples still provide paired originals.',
        '- Open **attention_presentation.html** for a portable handout with the figures embedded.', '',
        '## Evidence and limits', '',
        'The FFHQ train/validation/test image indices are disjoint; identity-disjointness and recognizer pretraining overlap are not established. '
        'Both recognizers share WebFace600K pretraining. These fixed test splits have been evaluated repeatedly; they are diagnostic comparisons, not fresh untouched benchmarks. '
        'Human recognizability has not been measured. Small parameter count and correct attention mathematics do not establish identity fidelity.', '',
        'Sources: [linear attention paper](https://proceedings.mlr.press/v119/katharopoulos20a.html), '
        '[torchvision deformable convolution](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.deform_conv2d.html), '
        '[QMUL official dataset and protocol links](https://qmul-survface.github.io/).', '']
    (OUT / 'ATTENTION_AND_QMUL.md').write_text('\n'.join(lines), encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8"><title>Attention and real QMUL results</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f3f3ee;color:#18291d}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:7px;text-align:left}pre{white-space:pre-wrap;background:#e6e9df;padding:15px}h2{margin-top:2em}</style>',
        '<h1>Verified attention and real QMUL camera examples</h1>',
        '<p>479,109-parameter custom network. Six sparse-sampling blocks and six adaptive spatial-key-sparse linear attention blocks. Reconstruction estimates missing detail; 90% identity recovery is not established.</p>',
        '<h2>What the attention does</h2><p>It scores spatial positions, keeps 392 of 784 as keys, and mixes information using a normalized ELU+1 kernel. A 48 x 48 summary avoids the full 784 x 784 attention matrix. The positions adapt; the 50% count is fixed. Sparse sampling uses nine learned deformable locations within a 7 x 7 region.</p>',
        f'<p>Raw FP32 weights: {params*4/1e6:.3f} MB, excluding runtime memory. Warm batch-1 inference on {html.escape(audit["hardware"])}: {audit["latency"]["28"]["original_model"]["median_ms"]:.2f} ms for 28 to 112. This is not a phone or full application benchmark.</p>',
        '<h2>Attention-focused experiments</h2><p>Seven inference interventions and three controlled, two-epoch attention-only training pilots did not show a meaningful balanced gain. The existing attention was retained. Two-pass horizontal-flip averaging improved the clean-input test SSIM from 0.8518 to 0.8545; eight rotation/reflection passes were rejected.</p>',
        '<h2>How accuracy is measured without HR</h2><pre>Face A -> SR -> embedding A\nFace B -> SR -> embedding B\nCosine + calibrated threshold -> same/different decision\nCompare with dataset annotation -> correct/incorrect</pre>',
        '<p>Accuracy is the percentage of correct pair decisions over 10,640 labelled pairs. It is not a percentage of an image restored. Thresholds are calibrated on other folds. No clean target is needed for those pair labels.</p>',
        '<table><tr><th>Evaluator</th><th>Processing</th><th>Verification accuracy (%)</th><th>AUC</th></tr>']
    for net, key in [('ArcFace (training prior)', 'conditions'), ('MobileFaceNet (excluded from loss)', 'mobileface_conditions')]:
        for method, value in qmul[key].items():
            parts.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in [net, method, f'{value["acc"]*100:.2f}', f'{value["auc"]:.5f}']) + '</tr>')
    parts += ['</table>', '<p>The benchmark uses a shared resized 28-pixel input. The interactive QMUL gallery preserves native crop dimensions and aspect ratio. No matching clean reference exists for these examples.</p>']
    figures = {'qmul_one_example': 'Actual QMUL camera crop and its reconstructions',
               'qmul_attention_masks': 'Actual learned key-selection maps (not identity evidence)',
               'qmul_same_person_pair': 'Dataset-labelled same-person pair',
               'qmul_different_people_pair': 'Dataset-labelled different-person pair',
               'ensemble_test_clean28': 'Held-out clean-downsampling comparison: one versus two passes',
               'ensemble_test_standard28': 'Held-out heavy-degradation comparison: one versus two passes'}
    for name, title in figures.items():
        data = base64.b64encode((OUT / (name + '.png')).read_bytes()).decode('ascii')
        parts.append(f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{data}">')
    parts.append('<p>Full protocol, validation/pilot tables, limitations and sources are in ATTENTION_AND_QMUL.md. No claim of 90% restoration is made.</p>')
    (OUT / 'attention_presentation.html').write_text('\n'.join(parts), encoding='utf-8')
    print(json.dumps({'ensemble_review': review, 'controlled_training_checks': pilot_checks}, indent=2))


if __name__ == '__main__': main()

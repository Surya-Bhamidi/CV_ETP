"""Report the 70% target without substituting fusion or recalibration for it."""
import base64
import csv
import html
import json
import re

from prepare_recognition_v4 import OUT
from refine_model import write_json
from write_qmul_study import markdown_table


def main():
    result=json.loads((OUT/'evaluation.json').read_text());selected=json.loads((OUT/'selection.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text());checks=json.loads((OUT/'verification_checks.json').read_text())
    lines=['# Attempt to exceed 70% QMUL verification','','## Primary result','',
        ('The validation-selected v4 checkpoint exceeded 70% MobileFace pair-verification accuracy under the unchanged primary calibration.'
         if result['primary_target_over70'] else 'The attempt did not exceed 70% MobileFace pair-verification accuracy under the unchanged primary calibration. '
         'Any fusion or expanded-calibration score is a separate recognition pipeline and does not change this primary result.'),'',
        'The main cohort remains all 9,913 images and all 10,640 original verification pairs. Every model receives identical cached 28 x 28 RGB pixels and produces 112 x 112 outputs. '
        'Primary thresholds use the original 512 validation crops from 256 official-training identities, with the original 256 matched and 256 unmatched validation pairs. '
        'No threshold is fit on a test fold. Recognition percentages do not measure the proportion of facial detail restored.','']
    def rows(dataset,calibration):
        return [[name,key,f'{v["accuracy"]*100:.2f}',f'{v["auc"]:.5f}',f'{v["threshold_from_training_validation"]:.6f}']
            for name,datasets in result['recognition'].items() for key,v in datasets[dataset][calibration].items()]
    headers=['SR checkpoint','Recognition pipeline','Accuracy (%)','AUC','Validation threshold']
    lines+=markdown_table(headers,rows('prior_9913_images','primary'))
    lines+=['','The fusion row uses a fixed equal average of ArcFace and MobileFace scores, standardized with negative validation-pair statistics. '
        'It uses two recognizers and changes no reconstructed pixels. It is a heavier recognition pipeline; the SR core itself still has 479,109 parameters. '
        'ArcFace is a training prior; MobileFace is excluded from training and checkpoint selection. Fusion is not an independent-evaluator result.','',
        '## Additional harder identity-disjoint diagnostic cohort','',
        'This uses the same 1,423 images across 456 additional official-test identities and 3,416 balanced pairs as v3. These identities are absent from the main pair cohort, training and validation. '
        'The cohort has already been evaluated in v3, and historical pixel-model identification covered a larger dataset; it is not a new untouched benchmark.','']
    lines+=markdown_table(headers,rows('additional_identity_disjoint','primary'))
    lines+=['','## Expanded validation calibration: separate pipeline','',
        f'The expanded pool has {protocol["validation_images"]:,} crops from the same 256 held-out official-training identities. '
        f'Its {protocol["expanded_pairs"]:,} balanced calibration pairs were frozen before training and test scoring. '
        'Exact decoded-image hashes were checked against the audited 32,768 training images and both evaluation cohorts, with zero matches. '
        'The extra validation crops never enter gradient training. Because thresholds differ, these rows must not be substituted for the primary 512-calibrated result.','',
        'Main 10,640-pair cohort:','']
    lines+=markdown_table(headers,rows('prior_9913_images','expanded'))
    lines+=['','Additional diagnostic cohort:','']+markdown_table(headers,rows('additional_identity_disjoint','expanded'))
    lines+=['','## What changed in training','',
        'Two further epochs start from the selected v3 checkpoint. Each has 1,024 updates, 16 paired FFHQ examples and 32 camera crops per batch. '
        'Two native camera shots from the same official-training identity are sampled together. A small supervised contrastive loss on frozen ArcFace features '
        'encourages same-person consistency and separates different training identities, alongside pixel, structure, perceptual, feature-distillation and LR-observation constraints. '
        'Only paired FFHQ training images supply clean HR texture targets. Camera identity labels are used only from the official training split.','',
        f'Validation selected extra epoch {selected["selected_epoch"]}; a new candidate was selected: {selected["selected_new_checkpoint"]}. '
        'Selection maximizes original 512-validation ArcFace AUC, with mean FFHQ validation LPIPS at most base+0.005, PSNR at least base-0.15 dB and SSIM at least base-0.003. '
        'The baseline is retained if no epoch passes and improves validation AUC. No test score changes checkpoint selection or app availability.','',
        '## Paired reconstruction quality','',
        'Both input conditions use the same 512 FFHQ test images across methods. Lower LPIPS is better. LPIPS-Squeeze is optimized in training; '
        'LPIPS-Alex is excluded from training and selection. These are FFHQ paired metrics, not QMUL restoration percentages.','']
    quality=[]
    for profile,methods in result['paired_ffhq'].items():
        for name,v in methods.items():quality.append([profile,name,f'{v["psnr_y"]:.4f}',f'{v["ssim_y"]:.5f}',f'{v["lpips_squeeze"]:.5f}',f'{v["lpips_alex"]:.5f}'])
    lines+=markdown_table(['Condition','SR checkpoint','PSNR-Y','SSIM-Y','LPIPS-Squeeze','LPIPS-Alex'],quality)
    lines+=['','## Integrity and interpretation','',
        '- Original primary calibration indices, thresholds, standardized fusion scores, raw pair decisions, AUC and 512-image quality means were independently reproduced.',
        '- Protocol, cached pixels, selected checkpoint and calibration statistics are hashed and frozen before current test scoring.',
        '- The earlier scoped audit found no direct split/identity/duplicate leakage. Expanded validation also has zero exact decoded-image matches with the audited training and evaluation sets.',
        '- Unknown upstream pretraining overlap and shared unlabelled people within FFHQ cannot be ruled out. Repeated-test-informed research choices remain a limitation even when epochs and thresholds use validation.',
        '- A point estimate above 70% does not establish that true deployment accuracy exceeds 70%; there is one training seed, correlated pairs and camera-domain shift.',
        '- No human recognition or preference study establishes a noticeable improvement. Inspect the saved fixed examples. Very small blurry camera crops can remain unrecognizable.',
        '- There is no paired clean QMUL HR original. Sharper or more verification-friendly outputs do not certify exact identity recovery.','',
        '## Artifacts and source','',
        '- [Full metrics](evaluation.json)',
        '- [All recognition rows CSV](recognition.csv)',
        '- [Validation selection](selection.json)',
        '- [Calibration statistics](calibration.json)',
        '- [Integrity checks](verification_checks.json)',
        '- [Supervised contrastive learning paper](https://arxiv.org/abs/2004.11362)',
        '- [Live comparison](http://127.0.0.1:8765/?split=qmul&model='+('identity_v4' if selected['selected_new_checkpoint'] else 'restoration_v3')+'&compare=previous)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fields=['sr','cohort','calibration','recognizer','accuracy_percent','auc','threshold','pairs']
    with (OUT/'recognition.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        for name,datasets in result['recognition'].items():
            for dataset,calibrations in datasets.items():
                for calibration,pipelines in calibrations.items():
                    for key,v in pipelines.items():writer.writerow([name,dataset,calibration,key,v['accuracy']*100,v['auc'],v['threshold_from_training_validation'],v['pairs']])
    inline=lambda text:re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',html.escape(text,quote=True))
    parts=['<!doctype html><meta charset="utf-8"><title>QMUL70% attempt</title>',
        '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18271e}table{border-collapse:collapse;width:100%;font-size:12px}td{border:1px solid #bac5b7;padding:6px}img{width:100%;height:auto}@media print{img{break-inside:avoid}h2{break-after:avoid}}</style>']
    table=False
    for line in lines:
        if line.startswith('| '):
            if not table:parts.append('<table>');table=True
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(c=='---' for c in cells):continue
            parts.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
        else:
            if table:parts.append('</table>');table=False
            if line.startswith('# '):parts.append('<h1>'+inline(line[2:])+'</h1>')
            elif line.startswith('## '):parts.append('<h2>'+inline(line[3:])+'</h2>')
            elif line:parts.append('<p>'+inline(line)+'</p>')
    if table:parts.append('</table>')
    for name in ['qmul_comparison','standard28_comparison','clean28_comparison']:
        parts.append('<h2>'+name.replace('_',' ')+'</h2><img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">')
    (OUT/'presentation.html').write_text('\n'.join(parts),encoding='utf-8')
    write_json(OUT/'report_status.json',{'verified':bool(checks),'primary_over70':result['primary_target_over70'],'separate_primary_calibrated_fusion_over70':result['fusion_over70']})
    print('V4 REPORT WRITTEN',flush=True)


if __name__=='__main__':main()

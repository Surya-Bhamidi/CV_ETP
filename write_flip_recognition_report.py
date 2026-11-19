"""Verify and report the fixed recognition-only follow-up without hiding failures."""
import base64
import html
import json
import re

import numpy as np
from playwright.sync_api import sync_playwright
from sklearn.metrics import roc_auc_score

from evaluate_flip_recognition import OUT,V4
from evaluate_recognition_v4 import fit_calibration,fuse
from refine_model import sha,write_json
from run_project import ROOT,notebook_definitions
from write_qmul_study import markdown_table


def main():
    result=json.loads((OUT/'evaluation.json').read_text());calibration=json.loads((OUT/'calibration.json').read_text())
    v4=json.loads((V4/'evaluation.json').read_text())
    assert result['v4_original_evaluation_sha256']==sha(V4/'evaluation.json')
    assert result['protocol']['v4_selection_sha256']==sha(V4/'selection.json')
    assert result['calibration_sha256']==sha(OUT/'calibration.json')
    assert result['protocol']['image_output_changed'] is False
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'flip_verify')
    with np.load(OUT/'emb/validation.npz') as data:
        scores={key:data[key+'_scores'] for key in ['arcface','mobileface']}
        assert fit_calibration(r,scores,data['labels'])==calibration
    for dataset,pipelines in result['recognition'].items():
        with np.load(OUT/'emb'/f'{dataset}.npz') as data:
            scores={key:data[key+'_scores'] for key in ['arcface','mobileface']}
            assert np.array_equal(fuse(scores,calibration['fusion_parameters']),data['fusion_scores'])
            for key,record in pipelines.items():
                values=data[key+'_scores'];correct=(values>=calibration['thresholds'][key])==data['labels']
                assert abs(correct.mean()-record['accuracy'])<1e-12
                assert abs(roc_auc_score(data['labels'],values)-record['auc'])<1e-12
                assert int(correct.sum())==record['correct_pairs']
    goal={key:values['accuracy']>.70 for key,values in result['recognition']['prior_9913_images'].items()}
    with np.load(ROOT/'work/qmul_study/cache/pairs.npz') as pair_data:anchors=pair_data['ia']
    cohort=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    anchor_ids=np.array([cohort['files'][i].split('_')[0] for i in anchors])
    _,inverse=np.unique(anchor_ids,return_inverse=True);counts=np.bincount(inverse)
    with np.load(OUT/'emb/prior_9913_images.npz') as data,np.load(V4/'emb/identity_v4_prior_9913_images.npz') as old:
        correct=((data['mobileface_scores']>=calibration['thresholds']['mobileface'])==data['labels']).astype(float)
        old_threshold=v4['recognition']['identity_v4']['prior_9913_images']['primary']['mobileface']['threshold_from_training_validation']
        previous=((old['mobileface_scores']>=old_threshold)==old['labels']).astype(float)
        correct_sums=np.bincount(inverse,weights=correct);difference_sums=np.bincount(inverse,weights=correct-previous)
        rng=np.random.default_rng(42);indices=rng.integers(0,len(counts),(2000,len(counts)));denominator=counts[indices].sum(1)
        interval=np.percentile(correct_sums[indices].sum(1)/denominator*100,[2.5,97.5]).tolist()
        delta_interval=np.percentile(difference_sums[indices].sum(1)/denominator*100,[2.5,97.5]).tolist()
    uncertainty={'seed':42,'resamples':2000,'main_mobileface_accuracy_95_anchor_cluster_interval_percent':interval,
        'main_mobileface_change_95_anchor_cluster_interval_pp':delta_interval,
        'scope':'Fixed-model/calibration anchor-ID resampling; shared negative identities, threshold fitting, training seeds, historical test reuse and upstream overlap are not fully accounted for.'}
    write_json(OUT/'uncertainty.json',uncertainty)
    lines=['# Further recognition attempt toward70%','',
        'The SR checkpoint and output pixels are unchanged from identity-pair v4. This fixed follow-up averages normalized recognition features '
        'from the original and horizontally mirrored SR output. Each recognizer therefore runs twice. ArcFace/MobileFace fusion remains an equal '
        'average of negative-validation-standardized scores; it uses two recognizers.','',
        'All thresholds use the unchanged original512 validation crops from256 official-training identities and the same512 balanced validation pairs. '
        'There is no expanded calibration or test-label threshold fitting in this follow-up. The settings were frozen before the current test scores were computed. '
        'The cohorts have been evaluated repeatedly, so this is exploratory research rather than a fresh generalization benchmark.','',
        '## Main cohort: all9,913 images and10,640 pairs','']
    rows=[]
    for method,values in [('v4 single view',v4['recognition']['identity_v4']['prior_9913_images']['primary']),('v4 mirrored-feature average',result['recognition']['prior_9913_images'])]:
        for key,record in values.items():rows.append([method,key,f'{record["accuracy"]*100:.2f}',f'{record["auc"]:.5f}'])
    lines+=markdown_table(['Recognition processing','Pipeline','Accuracy (%)','AUC'],rows)
    lines+=['',f'Mirrored MobileFace main-cohort anchor-cluster bootstrap interval: {interval[0]:.2f}% to {interval[1]:.2f}%. '
        f'Paired change interval: {delta_interval[0]:+.2f} to {delta_interval[1]:+.2f} percentage points. '
        'These diagnostics condition on fixed models and calibration and do not fully cover shared negative identities, threshold fitting, training seeds, repeated-test bias or pretraining overlap. '
        'An observed point estimate just above70% does not establish true deployment accuracy above70%.','']
    lines+=['','Observed point estimates above70%: '+', '.join(key for key,value in goal.items() if value) if any(goal.values()) else '','No point estimate in the mirrored-feature experiment exceeds70%.' if not any(goal.values()) else '',
            'A recognition score is not an image-restoration percentage. Mirrored recognition features add no new facial detail to the output image.','',
            '## Additional harder cohort: same1,423 images,456 identities and3,416 pairs','']
    rows=[]
    for method,values in [('v4 single view',v4['recognition']['identity_v4']['additional_identity_disjoint']['primary']),('v4 mirrored-feature average',result['recognition']['additional_identity_disjoint'])]:
        for key,record in values.items():rows.append([method,key,f'{record["accuracy"]*100:.2f}',f'{record["auc"]:.5f}'])
    lines+=markdown_table(['Recognition processing','Pipeline','Accuracy (%)','AUC'],rows)
    lines+=['','## What remains limited','',
        '- The original standalone MobileFace primary result from v4 is68.10%. Any higher mirrored-feature or fusion value uses different recognition processing and must be labeled accordingly.',
        '- The separate v4 report also shows an expanded-validation v3 fusion result of70.05%. That changes calibration and recognition processing; it is not the primary result or a v4 training gain.',
        '- Very small gains can fall within sampling/calibration variation. There is one training seed, correlated pairs and camera-domain shift; a point estimate does not certify true deployment performance above70%.',
        '- The scoped audit detected no direct split/identity/duplicate leakage in the checked data, but unknown upstream pretraining overlap, unlabelled FFHQ person overlap and historical repeated-test-informed choices remain unresolved.',
        '- Paired image-quality measurements and fixed before/after examples are those of v4. There are no paired clean HR QMUL originals or human recognition/preference results.','',
        '## Artifacts','',
        '- [Full follow-up metrics](evaluation.json)',
        '- [Frozen follow-up settings](protocol.json)',
        '- [Validation calibration](calibration.json)',
        '- [V4 training, quality and calibration comparison](../recognition_v4/REPORT.md)',
        '- [Live v4 image comparison](http://127.0.0.1:8765/?split=qmul&model=identity_v4&compare=previous)','']
    # Add spaces in prose without changing model identifiers such as v4.
    replacements={'toward70':'toward 70','above70':'above 70','exceeds70':'exceeds 70','original512':'original 512','from256':'from 256','same512':'same 512','all9,913':'all 9,913','and10,640':'and 10,640','same1,423':'same 1,423','images,456':'images, 456','and3,416':'and 3,416','is68.10':'is 68.10','of70.05':'of 70.05'}
    lines=[__import__('functools').reduce(lambda text,item:text.replace(*item),replacements.items(),line) for line in lines]
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>Further recognition attempt</title>',
        '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18271e}table{border-collapse:collapse;width:100%}td{border:1px solid #bac5b7;padding:6px}img{width:100%;height:auto}@media print{img{break-inside:avoid}}</style>']
    inline=lambda text:re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',html.escape(text,quote=True))
    table=False
    for line in lines:
        if line.startswith('| '):
            if not table:parts.append('<table>');table=True
            cells=[cell.strip() for cell in line.strip('|').split('|')]
            if all(cell=='---' for cell in cells):continue
            parts.append('<tr>'+''.join('<td>'+inline(cell)+'</td>' for cell in cells)+'</tr>')
        else:
            if table:parts.append('</table>');table=False
            if line.startswith('# '):parts.append('<h1>'+inline(line[2:])+'</h1>')
            elif line.startswith('## '):parts.append('<h2>'+inline(line[3:])+'</h2>')
            elif line:parts.append('<p>'+inline(line)+'</p>')
    if table:parts.append('</table>')
    for name in ['qmul_comparison','standard28_comparison','clean28_comparison']:
        parts.append('<h2>'+name.replace('_',' ')+'</h2><img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((V4/(name+'.png')).read_bytes()).decode()+'">')
    (OUT/'presentation.html').write_text('\n'.join(parts),encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu']);page=browser.new_page()
        page.goto((OUT/'presentation.html').as_uri());assert page.locator('img').evaluate_all('(images)=>images.every(image=>image.complete&&image.naturalWidth>0)')
        page.emulate_media(media='print');page.pdf(path=str(OUT/'report.pdf'),format='A4',landscape=True,print_background=True,
            margin={'top':'12mm','bottom':'12mm','left':'10mm','right':'10mm'});browser.close()
    write_json(OUT/'verification_checks.json',{'raw_pair_scores_accuracy_auc_reproduced':True,'validation_only_thresholds_reproduced':True,
        'fixed_fusion_reproduced':True,'sr_pixels_unchanged':True,'point_estimates_above70':goal,'portable_figures_checked_and_pdf_exported':True,'report_source_sha256':sha(__file__)})
    print('FLIP REPORT VERIFIED',json.dumps(goal),flush=True)


if __name__=='__main__':main()

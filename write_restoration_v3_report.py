"""Export direct leakage evidence, fixed-calibration scores and visual tradeoffs."""
import base64
import html
import json
import re

from refine_model import sha, write_json
from run_project import ROOT
from train_restoration_v3 import OUT
from write_qmul_study import markdown_table


def main():
    audit=json.loads((ROOT/'work/leakage_review_v2/audit.json').read_text())
    evaluation=json.loads((OUT/'evaluation.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    review=json.loads((OUT/'review.json').read_text())
    lines=['# Leakage audit and further restoration experiment','','## Direct data-leakage checks','',
        'The audit checks the actual 70,000 FFHQ cached images, 32,768 official QMUL training crops, 512 validation crops, '
        '9,913 prior verification crops and 1,423 additional diagnostic crops. No direct overlap was detected under the following checks. '
        'This is a scoped result, not an absolute guarantee about unlabelled identities or upstream pretraining.','']
    lines+=markdown_table(['Potential direct leakage','Detected'],[[k,v] for k,v in audit['blocking_findings'].items()])
    lines+=['','The strict near-duplicate rule combines perceptual-hash Hamming distance <=4 with mean RGB difference <=3/255 after resizing to32. '
        'Ten hash candidates exceeded the RGB difference cutoff; they remain listed for review. FFHQ image IDs, raw pixel hashes and center-crop hashes were also checked. '
        'The exact cross-dataset check compares center-cropped FFHQ28 with QMUL28; it does not prove that the same people never occur in both datasets.','',
        '## Limits that must remain in the presentation','',
        '- FFHQ has no person-identity labels in this cache. Its split is image-disjoint; person-disjointness is not established.',
        '- Unknown identity/image overlap in ArcFace, MobileFaceNet, VGG, LPIPS or external-restorer pretraining cannot be certified absent.',
        '- Earlier test scores were repeatedly examined and some app release gates used test results. Historical scores carry test-informed selection bias even though training images were disjoint.',
        '- Previous pair-fold calibration can share identities and images across folds. The new comparison uses thresholds calibrated only on official-training validation identities.',
        '- The extra cohort contains 456 official-test identities absent from the recent 9,913-image pair comparisons. The original pixel-model identification run previously covered a larger test set, so this is an additional diagnostic cohort, not a claim of universally untouched data.','',
        '## Further training and frozen selection','',
        'The same 479,109-parameter SSC plus adaptive sparse kernel-linear-attention network was trained for two further epochs on '
        '32,768 FFHQ training images and 32,768 official QMUL training crops. Camera-like synthetic degradation adds variable native sampling and anisotropic blur. '
        'The camera observation constraint now compares the downsampled SR output with observed LR pixels, avoiding a full-size target that forces blurred texture. '
        'Only paired FFHQ training targets supply clean detail. A frozen ArcFace prior anchors observed facial features; MobileFaceNet and LPIPS-Alex are excluded from loss and checkpoint selection.','',
        'The validation criteria were saved before training. Full epoch failures remain in training_selection.json. '
        'If no full epoch passes, an explicitly recorded validation-only comparison of 50% and 75% parameter updates uses the same numerical limits. '
        'This exploratory selection is completed before test evaluation; no test score enables or disables the new option.','',
        f'Selected extra epoch: {selection["selected_epoch"]}; parameter-update fraction: {selection["selected_parameter_update_fraction"]:.2f}; '
        f'new candidate available: {review["available_for_comparison"]}. The fraction is not restoration accuracy.','',
        '## Recognition with validation-only threshold calibration','',
        'For every method, thresholds are calibrated using 512 crops from 256 official-training validation identities. '
        'Those identities do not occur in training or either evaluation cohort. Test labels only score the frozen outputs. '
        'These accuracies use different calibration from the historical 10-fold scores and must not be compared directly with those numbers. '
        'AUC is also reported because it does not depend on the chosen operating threshold.','']
    rows=[]
    for name,cohorts in evaluation['qmul'].items():
        for cohort,nets in cohorts.items():
            for net,values in nets.items():rows.append([name,cohort,net,f'{values["accuracy"]*100:.2f}',f'{values["auc"]:.5f}',values['pairs']])
    lines+=markdown_table(['Method','Cohort','Recognizer','Accuracy (%)','AUC','Pairs'],rows)
    lines+=['',f'The additional cohort has {evaluation["additional_images"]} fixed random crops across {evaluation["additional_identities"]} identities and '
        f'{evaluation["additional_pairs"]} balanced positive/negative pairs. Identical positive-image pairs excluded: {evaluation["additional_identical_positive_pairs_excluded"]}. '
        'It is not the official pair list. Shared identities across pairs and one training seed limit uncertainty/generalization claims.','',
        '## Paired FFHQ reconstruction','',
        'Each condition uses the same 512 held-out FFHQ test images. These test images have been evaluated in earlier experiments; they do not constitute a newly untouched dataset. '
        'LPIPS-Squeeze participates in training. LPIPS-Alex uses a separately downloaded official backbone excluded from training and selection. '
        'Lower perceptual distance is better; neither metric is a percentage of the face restored.','']
    rows=[]
    for profile,methods in evaluation['paired_ffhq'].items():
        for name,values in methods.items():rows.append([profile,name,f'{values["psnr_y"]:.4f}',f'{values["ssim_y"]:.5f}',f'{values["lpips_squeeze"]:.5f}',f'{values["lpips_alex"]:.5f}'])
    lines+=markdown_table(['Condition','Method','PSNR-Y','SSIM-Y','LPIPS-Squeeze','LPIPS-Alex'],rows)
    lines+=['','## Interpretation','']
    if 'restoration_v3' in evaluation['qmul']:
        for cohort in ['prior_9913_images','additional_identity_disjoint']:
            before=evaluation['qmul']['balanced_previous'][cohort];after=evaluation['qmul']['restoration_v3'][cohort]
            lines.append(f'- {cohort}: MobileFace accuracy change {(after["mobileface"]["accuracy"]-before["mobileface"]["accuracy"])*100:+.2f} percentage points; '
                         f'ArcFace change {(after["arcface"]["accuracy"]-before["arcface"]["accuracy"])*100:+.2f} percentage points. Mixed or negative results must remain visible.')
        for profile,methods in evaluation['paired_ffhq'].items():
            before,after=methods['balanced_previous'],methods['restoration_v3']
            lines.append(f'- {profile}: independent LPIPS-Alex change {(after["lpips_alex"]/before["lpips_alex"]-1)*100:+.1f}%; '
                         f'PSNR change {after["psnr_y"]-before["psnr_y"]:+.3f} dB; SSIM change {after["ssim_y"]-before["ssim_y"]:+.5f}.')
    lines+=['','Visible gains must be assessed in the saved images; no human preference or recognition study has been run. '
        'Some severely blurred QMUL crops remain unrecognizable. No clean camera HR original exists, so exact face/identity recovery and a 90% restoration figure cannot be verified. '
        'Original video with tracked, aligned frames or a larger, less compressed face crop would supply additional real evidence that a single blurred still lacks.','',
        'Recognition differences on the additional cohort are small and mixed across recognizers. The bootstrap diagnostics in [uncertainty.json](uncertainty.json) '
        'resample fixed pair decisions by anchor identity and paired FFHQ images; they do not account fully for shared negative identities, calibration fitting, training seeds, upstream overlap or historical test reuse. '
        'An interval covering zero does not establish a reliable recognition gain.','',
        '## Artifacts and sources','',
        '- [Scoped audit JSON](../leakage_review_v2/audit.json)',
        '- [Frozen additional cohort](../leakage_review_v2/additional_cohort.json)',
        '- [Final validation selection](selection.json)',
        '- [Full measured results](evaluation.json)',
        '- [Live native camera comparison](http://127.0.0.1:8765/?split=qmul&model=restoration_v3&compare=previous)',
        '- [QMUL official protocol](https://qmul-survface.github.io/protocols.html)',
        '- [Real-ESRGAN degradation modeling paper](https://arxiv.org/abs/2107.10833)',
        '- [Official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>Leakage audit and further restoration</title>',
           '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18271e}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #bac5b7;padding:6px}img{width:100%;height:auto}@media print{img{break-inside:avoid}h2{break-after:avoid}}</style>']
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
    for name in ['qmul_additional_comparison','standard28_comparison','clean28_comparison']:
        parts.append('<h2>'+html.escape(name.replace('_',' '))+'</h2><img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">')
    (OUT/'presentation.html').write_text('\n'.join(parts),encoding='utf-8')
    write_json(OUT/'report_provenance.json',{'source_sha256':sha(__file__),'evaluation_sha256':sha(OUT/'evaluation.json')})
    print('V3 REPORT WRITTEN',flush=True)


if __name__=='__main__':main()

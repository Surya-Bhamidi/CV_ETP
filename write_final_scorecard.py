"""Consolidate existing measured results without rerunning or selecting models."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'work/final_scores'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows] + ['']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        'v4': 'work/recognition_v4/evaluation.json',
        'mirrored_recognition': 'work/recognition_flip_v5/evaluation.json',
        'mirrored_uncertainty': 'work/recognition_flip_v5/uncertainty.json',
        'native': 'work/field_check_v2/native_results.json',
        'native_uncertainty': 'work/field_check_v2/uncertainty.json',
        'native_runtime': 'work/field_check_v2/runtime_spotcheck.json',
        'native_latency_sweep': 'work/field_check_v2/latency.json',
        'three_model_quality': 'work/visual_tradeoff/summary.json',
        'three_model_conditions': 'work/three_model_comparison/summary.json',
        'passes': 'work/restoration_passes_v1/evaluation.json',
        'pass_runtime': 'work/restoration_passes_v1/runtime.json',
        'clean_correction': 'work/restoration_passes_v1/clean_only_followup/evaluation.json',
        'review': 'work/recognition_v4/review.json'}
    data = {name: read(path) for name, path in paths.items()}
    snapshot = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Current v4 and recorded inference variations; no new evaluation or training. Old checkpoints are not pooled into current scores.',
                'model': {'name': 'Identity-pair SR v4', 'parameters': 479109, 'scale': 4,
                          'raw_fp32_weight_bytes': 479109 * 4, 'checkpoint_sha256': data['review']['checkpoint_sha256']},
                'sources': {name: {'path': path, 'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for name, path in paths.items()},
                'results': data}
    (OUT / 'SCORES.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8', newline='\n')
    lines = ['# Current model: complete measured scorecard', '',
             'Model: identity-pair v4, 479,109 parameters, x4 output. Raw FP32 parameter storage is1,916,436 bytes (1.916MB), excluding buffers, runtime activations and the heavy comparison models. No new training or evaluation was performed to assemble this scorecard.', '',
             '**Honest assessment:** a compact, fast research prototype with useful distortion-score results. It is not demonstrated to restore90% of a person, achieve universally better visual quality than heavy models, or reliably recover identity from severely degraded crops. Eye/spectacle fidelity remains a known weakness. There is room to improve.', '',
             '## What each score means', '']
    lines += table(['Metric', 'Direction', 'Meaning and limit'], [
        ['PSNR-Y (dB)', 'Higher', 'Pixel-error measure on luminance versus a clean reference; not visual realism or identity accuracy. A3.01dB increase approximately halves mean squared luminance error under the same protocol.'],
        ['SSIM-Y', 'Higher', 'Local luminance, contrast and structure similarity to the reference; values near1 are closer.0.85 does not mean85% identity/detail restored.'],
        ['LPIPS-Alex / LPIPS-Squeeze', 'Lower', 'Learned perceptual-feature distance to a reference. The backbone matters; Alex and Squeeze scores must not be compared numerically as one metric. No universal pass/fail threshold.'],
        ['MobileFace cosine', 'Higher', 'Normalized face-feature similarity to the reference, generally between-1 and1; a diagnostic proxy, not a probability, human-recognition score or restoration percentage.'],
        ['Verification accuracy', 'Higher', 'Percent of labeled same/different-person pairs classified correctly at a specified threshold. Original controlled pair sets are balanced, giving a50% constant-class baseline; rejecting unsupported pairs can change class balance. Not open-set person-identification accuracy.'],
        ['ROC AUC', 'Higher', 'How well similarity scores rank same-person above different-person pairs over thresholds.0.5 is chance ranking,1 is perfect ranking. Not a percent restored.'],
        ['Pair coverage', 'Higher', 'Fraction of pairs supported by the app input-size policy. Unsupported pairs must remain visible.'],
        ['All-pair success', 'Higher', 'Correct supported decisions divided by every original pair; unsupported pairs count as unresolved/unsuccessful. Not ordinary forced-decision accuracy.'],
        ['Latency (ms)', 'Lower', 'Time for the stated processing path on the stated hardware. Median is typical; p95 means95% of recorded requests were no slower. Crop inference is not camera FPS.'],
        ['Parameters', 'Smaller for compactness', 'Learned tensor count; neither an accuracy score nor the full app memory footprint.']])
    lines += ['## Image quality:512 FFHQ test images per condition', '',
              'Original v4 evaluation on112x112 clean targets reconstructed from28x28 inputs. Four-pixel-border Y-channel PSNR/SSIM. These are synthetic input degradations, not QMUL real-camera targets.', '']
    rows = []
    for p, models in data['v4']['paired_ffhq'].items():
        v = models['identity_v4']
        rows.append([p, f"{v['psnr_y']:.4f}", f"{v['ssim_y']:.5f}", f"{v['lpips_alex']:.5f}", f"{v['lpips_squeeze']:.5f}"])
    lines += table(['Condition', 'PSNR-Y', 'SSIM-Y', 'LPIPS-Alex', 'LPIPS-Squeeze'], rows)
    lines += ['## Latest optional clean-only correction:512 FFHQ test images', '',
              'One neural pass plus three residual consistency corrections. Clean-bicubic scope was selected on validation before this test. Single pass remains the default; this mode is disabled for native uploads, QMUL, noisy/mild inputs and external models.', '']
    rows = []
    for m, v in data['clean_correction']['test_clean28'].items():
        rows.append([m, *[f'{v[k]:.5f}' for k in ['psnr_y', 'ssim_y', 'lpips_alex', 'mobileface_cosine']]])
    lines += table(['Mode', 'PSNR-Y', 'SSIM-Y', 'LPIPS-Alex', 'MobileFace cosine'], rows)
    a, b = [data['clean_correction']['test_clean28'][m] for m in ['single', 'backprojection3']]
    lines += [f"PSNR change: +{b['psnr_y']-a['psnr_y']:.4f}dB; SSIM change: +{b['ssim_y']-a['ssim_y']:.5f}; relative LPIPS reduction: {100*(1-b['lpips_alex']/a['lpips_alex']):.2f}%; cosine change: +{b['mobileface_cosine']-a['mobileface_cosine']:.5f}.", '',
              'The original quality report computes LPIPS on float predictions; the latest pass study uses actual8-bit export pixels for LPIPS and face diagnostics. Hence the older clean baseline LPIPS0.17172 and the latest0.17141 are different measurement pipelines, not a new model improvement. These test images have been examined historically; no claim of a new untouched test cohort is made.', '',
              '## Current three-model comparison:128 FFHQ validation images, standard28', '',
              'The exact same degraded pixels go into every model. External models resize internally to512 and return112. Published training sources share FFHQ but exact splits, training budgets and camera adaptation differ. External FFHQ pretraining overlap is not excluded. Descriptive diagnostics, not an independent matched-training benchmark.', '']
    means = data['three_model_quality']['mean_metrics']
    timing = data['three_model_conditions']['validation/standard28']
    rows = []
    for m, count in [('identity_v4',479109),('restoration',86444252),('codeformer',94112707),('bicubic',0)]:
        v = means[m]
        rows.append([m, f'{count:,}', *[f'{v[k]:.5f}' for k in ['psnr','ssim','lpips_alex','mobileface_cosine']], f"{timing[m]['median_selected_model_ms']:.2f}" if m in timing else 'Not timed'])
    lines += table(['Method', 'Parameters', 'PSNR-Y', 'SSIM-Y', 'LPIPS-Alex', 'Face cosine', 'Median model ms'], rows)
    counts = data['three_model_quality']['ours_better_than_both_baselines_image_counts']
    lines += ['v4 better than both heavy baselines on: ' + ', '.join(f'{k}: {v}/128' for k,v in counts.items()) + '.', '',
              'The heavy models have better mean perceptual distance. GFPGAN also has higher mean face-feature cosine in this comparison. v4 has higher PSNR/SSIM and lower measured crop-model latency. No human likeness/preference study has been completed.', '',
              '## Small fixed UI test gallery:12 faces, all three methods', '',
              'These small diagnostic gallery means are not the512-image headline. Clean and standard inputs are different experiments.', '']
    rows = []
    for cohort in ['test/standard28','test/clean28']:
        for m,v in data['three_model_conditions'][cohort].items():
            rows.append([cohort,m,f"{v['psnr_y']:.5f}",f"{v['ssim_y']:.5f}",f"{v['median_selected_model_ms']:.2f}" if 'median_selected_model_ms' in v else 'Not timed'])
    lines += table(['Condition', 'Method', 'PSNR-Y', 'SSIM-Y', 'Median model ms'], rows)
    lines += ['## Controlled QMUL verification: forced28x28 inputs', '',
              'Main:9,913 images,10,640 labeled pairs. Harder additional cohort:1,423 images from456 identities,3,416 pairs. Original512-crop official-training-validation calibration is primary. The SR image is unchanged when recognition features are mirrored or recognizer scores are fused.', '']
    rows = []
    for cohort in ['prior_9913_images','additional_identity_disjoint']:
        for method,v in data['v4']['recognition']['identity_v4'][cohort]['primary'].items():
            rows.append([cohort,'Single-view '+method,f"{v['accuracy']*100:.4f}%",f"{v['auc']:.6f}",f"{v['correct_pairs']}/{v['pairs']}"])
        for method,v in data['mirrored_recognition']['recognition'][cohort].items():
            rows.append([cohort,'Mirrored-feature '+method,f"{v['accuracy']*100:.4f}%",f"{v['auc']:.6f}",f"{v['correct_pairs']}/{v['pairs']}"])
    lines += table(['Cohort', 'Recognizer pipeline', 'Accuracy', 'AUC', 'Correct / total'], rows)
    lo,hi=data['mirrored_uncertainty']['main_mobileface_accuracy_95_anchor_cluster_interval_percent']
    lines += [f'The70.0282% mirrored-MobileFace result is7451/10640 correct pairs, only3 above exactly70%. Approximate fixed-model anchor-cluster95% interval: {lo:.2f}–{hi:.2f}%. It excludes some uncertainty from shared negative identities, threshold fitting, seeds, historical test reuse and unknown upstream pretraining. It is not reliable evidence of a general above70% guarantee.', '',
              '## Secondary expanded validation calibration', '',
              'Different threshold-calibration pool, not new SR weights or the same primary benchmark. Included for completeness; do not silently substitute its scores for the primary calibration.', '']
    rows=[]
    for cohort,entry in data['v4']['recognition']['identity_v4'].items():
        for method,v in entry['expanded'].items():rows.append([cohort,method,f"{100*v['accuracy']:.4f}%",f"{v['auc']:.6f}",f"{v['correct_pairs']}/{v['pairs']}"])
    lines+=table(['Cohort','Recognizer','Accuracy','AUC','Correct / total'],rows)
    lines += ['## Native QMUL crop replay: matches the app input policy', '',
              'Preserves native aspect ratio, rejects crops below8px, fits large crops within256px, exports8-bit x4 predictions and resizes to112 for recognition. Thresholds are calibrated separately on supported native validation pairs. This is a different pipeline from forced28, not an extra improvement to the same benchmark.', '']
    rows=[]
    for cohort,entry in data['native']['recognition']['identity_v4'].items():
        for recognizer,v in entry.items():
            rows.append([cohort,recognizer,f"{100*v['supported_pair_accuracy']:.4f}%",f"{100*v['all_pair_success_with_abstentions_counted_unsuccessful']:.4f}%",f"{v['supported_pair_auc']:.6f}",f"{100*v['coverage']:.4f}%",f"{v['correct_pairs']}/{v['supported_pairs']} supported"])
    lines+=table(['Cohort','Recognition','Supported accuracy','All-pair success','Supported AUC','Pair coverage','Correct / supported'],rows)
    lines += ['Main:41 unsupported images and84 affected pairs. Harder cohort:46 unsupported images and186 affected pairs. Report unsupported inputs, not just favorable supported accuracy.', '',
              'Bicubic is a meaningful reference: its native mirrored-feature supported accuracy was70.4339% on main and64.3653% on the harder cohort. Native v4 mirrored-feature accuracy was70.4718% and62.0433%. Thus v4 does not universally improve recognition over simple enlargement.', '',
              'No clean HR QMUL references exist for these crops, so QMUL PSNR/SSIM/LPIPS-to-truth and a restoration percentage are unavailable. No human recognition accuracy has been measured.', '',
              '## Runtime: recorded native crop spot-check', '',
              'RTX4060 Laptop GPU, warm batch1, FP32, TF32 off. These are prior recorded measurements, not a new benchmark of the modified UI. Historical browser-comparison timing included two custom forward passes and PNG/JSON transport, without recognition or final rendering.', '']
    rows=[]
    for crop,group in data['native_runtime']['gpu'].items():
        for path,v in group.items():
            s=v['summary'];rows.append(['19x25' if crop=='median_native' else '59x80',path,f"{s['median_ms']:.3f}",f"{s['p95_ms']:.3f}",s['samples']])
    for crop,v in data['native_runtime']['browser_fetch'].items():
        s=v['round_trip_summary'];rows.append(['19x25' if crop=='median_native' else '59x80','Historical browser request',f"{s['median_ms']:.3f}",f"{s['p95_ms']:.3f}",s['samples']])
    lines+=table(['Input crop','Path','Median ms','p95 ms','Samples'],rows)
    lines += ['Full camera capture, detection, tracking, temporal consistency and sustained multi-face/video FPS remain untested.', '',
              '## All repeated-pass results:128 validation images per condition', '',
              'Same v4 weights. Fixed-size feedback downsamples the previous prediction to the original LR dimensions before each new neural pass. Ensemble methods average inverse-transformed predictions from the original input. Backprojection3 uses one neural pass plus3 residual corrections. No method passed the original all-profile release guards; single remains the default.', '']
    rows=[]
    for p,entry in data['passes']['validation'].items():
        for method,v in entry.items():
            rows.append([p,method,*[f'{v[k]:.5f}' for k in ['psnr_y','ssim_y','lpips_alex','mobileface_cosine']]])
    lines+=table(['Profile','Method','PSNR-Y','SSIM-Y','LPIPS-Alex','Face cosine'],rows)
    lines += ['## Literal repeated x4:16 validation images per condition', '',
              'Dimensions grow28→112→448→1792. All variants are resized to the same112 target for scoring. These16-image means are separate from the128-image table, with matching inputs within this table.', '']
    rows=[]
    for p,entry in data['passes']['literal_cascade_validation16'].items():
        for method,v in entry.items():rows.append([p,method,*[f'{v[k]:.5f}' for k in ['psnr_y','ssim_y','lpips_alex','mobileface_cosine']]])
    lines+=table(['Profile','Method','PSNR-Y','SSIM-Y','LPIPS-Alex','Face cosine'],rows)
    lines += ['Repeating the identical original input produced a maximum output difference of0.0 in the recorded deterministic repeat check.', '',
              '## Pass-count runtime:28x28 batch1', '']
    rows=[]
    for method,v in data['pass_runtime']['timings'].items():rows.append([method,f"{v['median_ms']:.3f}",len(v['samples_ms'])])
    lines+=table(['Method','Median GPU ms','Trials'],rows)
    lines += ['Tiny differences around the one-pass baseline are timing noise; the clean correction is not demonstrated to speed up inference. Literal3 incurs large intermediate tensors, not new observed detail.', '',
              '## Final assessment and improvement priorities', '',
              '- Strongest demonstrated contribution: a479k-parameter network with fast crop inference and higher distortion scores on the declared synthetic degradation than these heavy pretrained baselines.',
              '- Main weakness: perceptual detail and human identity likeness. Heavy models scored better on mean LPIPS; recognition falls to roughly60–64% on harder crops.',
              '- The clean-only correction is a modest conditional improvement, not a general camera restoration fix.',
              '- Better native observations and aligned actual video frames could supply more evidence. Repeating the same pixels does not.',
              '- Future teacher-distillation or explicitly trained recurrent refinement should be separate experiments with validation selection, an untouched final evaluation cohort and human likeness/eye-frame review. No further improvement is guaranteed.', '',
              '## Data and evaluation limits', '',
              'No direct project split overlap was detected under the recorded identity/pixel/strict-near-duplicate checks. Unknown pretrained overlap and unlabelled FFHQ person overlap are not certified absent. External FFHQ pretraining may include project test images. Historical repeated-test research choices remain a limitation. No score supports a claim of85–90% true detail restored.', '',
              '## Source records', '']
    for name,path in paths.items():lines.append(f'- [{name}](../../{path}) — SHA256 `{snapshot["sources"][name]["sha256"]}`')
    lines += ['', 'Exact source values, including thresholds, raw timing samples and uncertainty records, are retained in [SCORES.json](SCORES.json).']
    # Links are relative to work/final_scores: ../../work/... resolves to the workspace.
    document='\n'.join(lines)+'\n'
    # Keep prose readable while retaining exact table values, hashes and paths.
    replacements={'storage is1,':'storage is 1,','1.916MB':'1.916 MB','restore90%':'restore 90%',
        'A3.01dB':'A 3.01 dB','near1':'near 1','closer.0.85':'closer. 0.85','mean85%':'mean 85%',
        'between-1 and1':'between -1 and 1','a50%':'a 50%','thresholds.0.5':'thresholds. 0.5',
        'ranking,1':'ranking, 1','means95%':'means 95%','quality:512':'quality: 512',
        'evaluation on112x112':'evaluation on 112x112','from28x28':'from 28x28','correction:512':'correction: 512',
        'actual8-bit':'actual 8-bit','LPIPS0.17172':'LPIPS 0.17172','latest0.17141':'latest 0.17141',
        'the512-image':'the 512-image','comparison:128':'comparison: 128','to512':'to 512','return112':'return 112',
        'gallery:12':'gallery: 12','Main:9,913':'Main: 9,913','images,10,640':'images, 10,640',
        'cohort:1,423':'cohort: 1,423','from456':'from 456','identities,3,416':'identities, 3,416',
        'Original512-crop':'Original 512-crop','The70.0282%':'The 70.0282%','is7451/10640':'is 7451/10640',
        'only3 above':'only 3 above','exactly70%':'exactly 70%','anchor-cluster95%':'anchor-cluster 95%',
        'above70%':'above 70%','below8px':'below 8px','within256px':'within 256px','exports8-bit':'exports 8-bit',
        'to112':'to 112','Main:41':'Main: 41','and84':'and 84','cohort:46':'cohort: 46','and186':'and 186',
        'was70.4339%':'was 70.4339%','and64.3653%':'and 64.3653%','was70.4718%':'was 70.4718%',
        'and62.0433%':'and 62.0433%','RTX4060':'RTX 4060','batch1':'batch 1','results:128':'results: 128',
        'plus3 residual':'plus 3 residual','x4:16':'x4: 16','grow28':'grow 28','same112':'same 112',
        'These16-image':'These 16-image','the128-image':'the 128-image','of0.0':'of 0.0','runtime:28x28':'runtime: 28x28',
        'a479k-parameter':'a 479k-parameter','to roughly60':'to roughly 60','of85':'of 85'}
    for before,after in replacements.items():document=document.replace(before,after)
    (OUT/'FINAL_SCORES.md').write_text(document,encoding='utf-8',newline='\n')
    print('Written:',OUT/'FINAL_SCORES.md')
    print('Exact source records:',OUT/'SCORES.json')


if __name__=='__main__':main()

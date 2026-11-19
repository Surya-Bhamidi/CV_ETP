"""Export measured native-camera replay, crop latency and practical limits."""
import base64
import csv
import html
import json
import re

from playwright.sync_api import sync_playwright

from field_native_check import OUT
from refine_model import sha,write_json
from write_qmul_study import markdown_table

LABELS={'bicubic':'Bicubic enlargement','app_default':'Existing app default','camera_v3':'Previous camera v3','identity_v4':'Latest identity-pair v4'}


def main():
    native=json.loads((OUT/'native_results.json').read_text());latency=json.loads((OUT/'latency.json').read_text())
    spot=json.loads((OUT/'runtime_spotcheck.json').read_text());checks=json.loads((OUT/'verification_checks.json').read_text())
    lines=['# Practical native-camera and real-time check','',
        'The latest model improves verification on native-size camera crops relative to the previous v3 and app-default models. '
        'Small-crop GPU computation is fast, while app transport/encoding and larger CPU crops add substantial latency. '
        'This is a recorded still-crop replay and local app timing check, not a live camera or physical field trial.','',
        '## Recognition with native camera input','',
        'Every method receives the same original QMUL crop, preserving its native dimensions and aspect ratio. The app minimum8/max256 input policy is applied. '
        'Output is x4, rounded to8-bit RGB as in a PNG download, then resized to112 for the same frozen MobileFace recognizer. '
        'Single-view and normalized original/mirrored recognition features are both reported. No model is trained or selected in this check.','',
        'The original512 validation crops from256 held-out official-training identities provide native-pipeline thresholds. Three validation images are too small; '
        '506 of512 original validation pairs remain supported for calibration. Unsupported evaluation images/pairs are retained in coverage totals, '
        'rather than silently excluded from the headline denominator. These native-pipeline numbers are different from the earlier forced28 benchmark.','']
    headers=['SR method','Recognition','Supported-pair accuracy (%)','All-pair success (%)','Supported AUC','Pair coverage (%)']
    def rows(dataset):
        return [[LABELS[name],mode,f'{v["supported_pair_accuracy"]*100:.2f}',f'{v["all_pair_success_with_abstentions_counted_unsuccessful"]*100:.2f}',f'{v["supported_pair_auc"]:.5f}',f'{v["coverage"]*100:.2f}']
            for name,datasets in native['recognition'].items() for mode,v in datasets[dataset].items()]
    lines+=['','### Main cohort: all9,913 images and10,640 pairs','']+markdown_table(headers,rows('main'))
    lines+=['','There are41 unsupported images and84 affected pairs. Supported-pair accuracy uses10,556 pairs. '
        'All-pair success treats an unsupported pair as unresolved/unsuccessful, not as a made-up class prediction. It is a coverage-aware success measure, not a standard forced-decision accuracy.','',
        '### Additional harder cohort: all1,423 images and3,416 pairs','']+markdown_table(headers,rows('additional'))
    lines+=['','There are46 unsupported images and186 affected pairs, leaving3,230 supported pairs. '
        'Latest v4 single-view accuracy improves over v3, but mirrored recognition does not improve it here. '
        'Bicubic plus mirrored recognition performs better than latest SR plus mirrored recognition on the harder cohort. '
        'SR therefore does not establish a universal recognition advantage over simple enlargement.','',
        'The bootstrap diagnostics in uncertainty.json condition on fixed models and native calibration. They do not fully account for shared negative identities, '
        'calibration fitting, training seeds, historical test reuse or unknown upstream pretraining overlap.','',
        '## Warm batch1 runtime on this laptop','',
        f'GPU: {latency["hardware"]["gpu"]}. Torch: {latency["hardware"]["torch"]}. FP32, TF32 disabled; two Torch CPU threads. '
        'The independent spot-check stores raw timings and repeats50 synchronized GPU measurements per stage and30 browser requests per crop size after warmup. '
        'The representative real camera inputs are19x25 and59x80.','']
    runtime_rows=[]
    for case in ['median_native','larger_native']:
        size='19 x 25' if case=='median_native' else '59 x 80'
        for key,label in [('sr_only','V4 SR only (GPU)'),('sr_plus_single_mobileface','V4 SR + single MobileFace (GPU)')]:
            v=spot['gpu'][case][key]['summary'];runtime_rows.append([size,label,f'{v["median_ms"]:.2f}',f'{v["p95_ms"]:.2f}',v['samples']])
        v=spot['browser_fetch'][case]['round_trip_summary'];runtime_rows.append([size,'Browser upload/comparison request',f'{v["median_ms"]:.2f}',f'{v["p95_ms"]:.2f}',v['samples']])
    lines+=markdown_table(['Input crop','Measured path','Median (ms)','p95 (ms)','Samples'],runtime_rows)
    lines+=['','GPU SR-only timing starts with a prepared GPU tensor and excludes transfers, recognition, comparison inference and image encoding. '
        'The full single-recognizer crop pipeline includes CPU RGB conversion/transfer, SR,8-bit conversion,112 resize, recognition and the embedding copied back to CPU; it excludes input image decode and output encoding.','',
        'Browser request timing uses the same-origin browser fetch path and includes upload handling, both previous/current SR comparisons, PNG/base64 output encoding and JSON parsing. '
        'It does not include recognition or final image rendering. Upload mode is deliberately used without sample_id, avoiding the gallery cache. '
        'A fresh-connection urllib client measured about46ms for the small crop; browser connection reuse measured about25ms. These are different client paths, not a model speed improvement.','',
        'The30Hz budget is33.33ms per crop. Small GPU crop processing and small-crop browser requests fit that budget in this warm run. '
        'The larger browser request does not. Reciprocal latency counts crops/requests per second; it does not measure full-camera video FPS or multi-face throughput.','',
        '### CPU and upload-limit stress checks','']
    cpu_rows=[]
    for case,methods in latency['cpu'].items():
        for name,v in methods.items():cpu_rows.append([str(latency['cases'][case]['input_size']),name,f'{v["median_ms"]:.2f}',f'{v["p95_ms"]:.2f}',v['samples']])
    lines+=markdown_table(['Input','CPU path (two threads)','Median (ms)','p95 (ms)','Samples'],cpu_rows)
    stress=latency['gpu']['max256_stress']['identity_v4_sr_only'];http=latency['http']['max256_stress']['identity_v4']['round_trip']
    lines+=['',f'A256x256 enlarged FFHQ image was used solely to stress the upload limit, not as genuine native camera detail. '
        f'V4 GPU SR alone took{stress["median_ms"]:.2f}ms median; the uncached fresh-connection app request took{http["median_ms"]:.2f}ms median. '
        'This size does not meet a30Hz crop budget. The stress request has only8 recorded samples, so its p95 is a short-run diagnostic, not a production tail-latency guarantee.','',
        'V3 and v4 have the same479,109-parameter architecture. Small timing differences between them are run-to-run variability, not evidence that v4 is a faster architecture.','',
        '## Actual visual improvement and remaining field gaps','',
        'The fixed six-example sheet uses native input dimensions, preserves aspect ratio, and uses nearest-neighbour display enlargement. '
        'Changes between v3 and v4 remain modest, and several crops remain visibly blurred. No clean HR original exists for these QMUL crops; '
        'PSNR/SSIM, recovered texture correctness and a restoration percentage cannot be measured on them. Human recognizability and preference have not been tested.','',
        'There is no original video or full-camera-frame source in the workspace. Capture latency, full-frame face detection, landmark alignment, tracking, '
        'temporal flicker, motion blur across consecutive frames, multi-face scheduling and sustained stream/thermal behavior remain untested. '
        'Sorting independent surveillance crops would not constitute a live video trial. A genuine camera/video source is required to validate those parts.','',
        'Known research limitations remain: these cohorts have been evaluated repeatedly; thresholds are validation-only, but research choices have been test-informed in earlier turns. '
        'The scoped leakage audit found no direct overlap in the checked splits. Unknown pretraining overlap and shared unlabelled people within FFHQ are not certified absent.','',
        '## Verification and artifacts','',
        '- All original pairs, native rejection masks, validation thresholds, supported accuracy, all-pair success and supported AUC were independently reproduced.',
        '- Raw timing samples reproduce the spot-check medians/p95 values. API outputs match offline x4 images within one RGB level in the measured cases.',
        '- [Native recognition CSV](native_recognition.csv)',
        '- [Native results](native_results.json)',
        '- [Full latency sweep](latency.json)',
        '- [Raw GPU/browser timing samples](runtime_spotcheck.json)',
        '- [Verification checks](verification_checks.json)',
        '- [Fixed native before/after sheet](native_comparison.png)',
        '- [Live comparison app](http://127.0.0.1:8765/?split=qmul&model=identity_v4&compare=previous)','']
    # Keep numeric values readable in prose while retaining model identifiers.
    replacements={'minimum8/max256':'minimum 8 / maximum 256','to8-bit':'to 8-bit','to112':'to 112','original512':'original 512','from256':'from 256','of512':'of 512','all9,913':'all 9,913','and10,640':'and 10,640','are41':'are 41','and84':'and 84','uses10,556':'uses 10,556','all1,423':'all 1,423','and3,416':'and 3,416','are46':'are 46','and186':'and 186','leaving3,230':'leaving 3,230','repeats50':'repeats 50','and30':'and 30','are19x25':'are 19x25','and59x80':'and 59x80','SR,8-bit':'SR, 8-bit','conversion,112':'conversion, 112','about46ms':'about 46 ms','about25ms':'about 25 ms','The30Hz':'The 30 Hz','is33.33ms':'is 33.33 ms','A256x256':'A 256x256','took':'took ','a30Hz':'a 30 Hz','only8':'only 8','same479,109':'same 479,109'}
    for i,line in enumerate(lines):
        for a,b in replacements.items():line=line.replace(a,b)
        lines[i]=line
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (OUT/'native_recognition.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['method','cohort','recognition_mode','supported_accuracy_percent','all_pair_success_percent','supported_auc','supported_pairs','all_pairs','pair_coverage_percent'])
        for name,datasets in native['recognition'].items():
            for dataset,modes in datasets.items():
                for mode,v in modes.items():writer.writerow([name,dataset,mode,v['supported_pair_accuracy']*100,v['all_pair_success_with_abstentions_counted_unsuccessful']*100,v['supported_pair_auc'],v['supported_pairs'],v['pairs'],v['coverage']*100])
    inline=lambda text:re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',html.escape(text,quote=True))
    parts=['<!doctype html><meta charset="utf-8"><title>Native-camera and real-time check</title>',
        '<style>body{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18271e}table{border-collapse:collapse;width:100%;font-size:12px}td{border:1px solid #bac5b7;padding:6px}img{width:100%;height:auto}@media print{img{break-inside:avoid}h2,h3{break-after:avoid}}</style>']
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
            elif line.startswith('### '):parts.append('<h3>'+inline(line[4:])+'</h3>')
            elif line:parts.append('<p>'+inline(line)+'</p>')
    if table:parts.append('</table>')
    parts.append('<h2>Actual native before/after crops</h2><img alt="Native input, bicubic, default, v3 and v4 on identical crops" src="data:image/png;base64,'+base64.b64encode((OUT/'native_comparison.png').read_bytes()).decode()+'">')
    (OUT/'presentation.html').write_text('\n'.join(parts),encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu']);page=browser.new_page()
        page.goto((OUT/'presentation.html').as_uri());assert page.locator('img').count()==1
        assert page.locator('img').evaluate_all('(images)=>images.every(image=>image.complete&&image.naturalWidth>0)')
        page.emulate_media(media='print');page.pdf(path=str(OUT/'report.pdf'),format='A4',landscape=True,print_background=True,
            margin={'top':'12mm','bottom':'12mm','left':'10mm','right':'10mm'});browser.close()
    write_json(OUT/'report_checks.json',{'verified_results':all(checks.values()),'source_sha256':sha(__file__),
        'native_results_sha256':sha(OUT/'native_results.json'),'latency_sha256':sha(OUT/'latency.json'),
        'raw_timing_sha256':sha(OUT/'runtime_spotcheck.json'),'portable_native_figure_loads':True,'pdf_exported':True,
        'scope':'Replay and per-crop timing only; no live camera/video validation.'})
    print('FIELD REPORT WRITTEN',flush=True)


if __name__=='__main__':main()

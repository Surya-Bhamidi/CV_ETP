"""Report actual repeated-inference trials, including rejected methods."""
import base64
import html
import json

from evaluate_restoration_passes import OUT


LABELS = {'single':'One SR pass','feedback2':'2 feedback passes','feedback3':'3 feedback passes',
          'feedback5':'5 feedback passes','feedback8':'8 feedback passes',
          'ensemble2':'2 flipped views averaged','ensemble4':'4 transformed views averaged',
          'ensemble8':'8 transformed views averaged','backprojection3':'One SR pass + 3 consistency corrections',
          'cascade1':'1 literal x4 pass','cascade2':'2 literal x4 passes','cascade3':'3 literal x4 passes'}


def image(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()


def table(methods):
    result = ['<table><tr><th>Method</th><th>PSNR-Y ↑</th><th>SSIM-Y ↑</th><th>LPIPS-Alex ↓</th><th>Face-feature cosine ↑</th></tr>']
    for name, scores in methods.items():
        result.append('<tr><td>'+LABELS[name]+'</td>'+''.join(f'<td>{scores[k]:.4f}</td>' for k in ['psnr_y','ssim_y','lpips_alex','mobileface_cosine'])+'</tr>')
    return '\n'.join(result+['</table>'])


def main():
    d=json.loads((OUT/'evaluation.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    clean=json.loads((OUT/'clean_only_followup/evaluation.json').read_text())
    runtime=json.loads((OUT/'runtime.json').read_text())
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Do more restoration passes help?</title>',
          '<style>body{font:16px system-ui;max-width:1220px;margin:30px auto;padding:0 20px;color:#14221a}table{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}td,th{border:1px solid #ccc;text-align:left;padding:8px}.row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:20px 0;break-inside:avoid}.row img{width:100%;aspect-ratio:1;object-fit:contain;image-rendering:auto}.row small{display:block;min-height:40px}.warning{background:#fff1d0;padding:15px}section{break-inside:avoid}@media(max-width:700px){.row{grid-template-columns:repeat(3,1fr)}}@media print{body{font-size:11px;margin:0}table{font-size:10px}.newpage{break-before:page}}</style>',
          '<h1>More passes did not provide a general restoration improvement</h1>',
          '<p>Actual preferred v4 weights, unchanged. No training or clean reference enters prediction. Selection for general use remains single pass. The optional clean-only correction below is a separate validation-selected follow-up.</p>',
          '<p class="warning">Rerunning an unchanged deterministic input produced identical outputs (maximum absolute difference: '+str(d['same_input_repeat_max_abs_difference'])+'). Feeding the previous output back adds no new observations. A smoother or sharper-looking image does not establish recovered identity.</p>',
          '<h2>Three ways of using extra computation</h2><ol><li>Literal cascading: 28→112→448→1792 pixels after 1,2,3 passes. For a fair comparison all outputs are resized to112 and measured against the same112 clean original.</li><li>Fixed-size feedback: resize the previous result to28 and apply the model again, always returning112. Tested 2,3,5,8 passes.</li><li>Self-ensemble: transform the same original input, predict separately, invert transformations and average. Tested 2,4,8 views; this is different from feeding predictions back.</li></ol>',
          '<h2>Validation: first128 fixed images per condition</h2><p>All methods use identical inputs and the same clean target. Three input conditions, 384 paired observations. PSNR/SSIM use float outputs; LPIPS/face-feature diagnostics use exported 8-bit pixels. Historical split reuse is disclosed. No restoration percentage or new full-QMUL recognition accuracy is claimed.</p>']
    for profile,methods in d['validation'].items():
        body.append('<h3>'+html.escape(profile)+'</h3>'+table(methods))
    body += ['<p>All-profile selection: <strong>'+selection['selected']+'</strong>. No ensemble passed the preregistered reconstruction, perceptual and face-feature guards across all three validation conditions. Rejected candidates were not tuned against test results.</p>',
             '<h2 class="newpage">Literal enlargement: first16 identical validation inputs per condition</h2><p>This smaller diagnostic is separate from the128-image trial. All three literal-pass variants share the same16 inputs in each condition. More output pixels are not counted as greater restoration accuracy.</p>']
    for profile,methods in d['literal_cascade_validation16'].items():
        body.append('<h3>'+profile+'</h3>'+table(methods))
    body += ['<h2 class="newpage">A limited alternative: clean-input consistency</h2>',
             '<p>One neural SR prediction is corrected three times using the discrepancy between the observed LR input and the downsampled reconstruction. The fixed correction step is0.2. No clean reference or identity label is used by the correction. This assumes clean bicubic downsampling and is not appropriate for unknown noisy camera degradation.</p>',
             '<p>Clean-only focus was chosen after examining validation results and frozen before the following512-image test diagnostic. The all-profile selection remains single. The project has historically examined this test split; this is not a new untouched cohort. No per-image method selection or weight updates were made.</p>',
             table(clean['test_clean28']),
             '<p>Mean LPIPS decreases by '+f"{100*(1-clean['test_clean28']['backprojection3']['lpips_alex']/clean['test_clean28']['single']['lpips_alex']):.2f}"+'%. This is relative perceptual-distance reduction, not a percentage of image detail restored. The improvement is modest and is not a demonstrated human recognizability gain.</p>']
    folder=OUT/'clean_only_followup'
    for i in range(6):
        body.append(f'<section><h3>Fixed test face{i+1:02}, clean downsampling</h3><div class="row">')
        for label,name in [('Original input, enlarged for display','input'),('Single-pass v4','single'),('One pass + 3 corrections','backprojection3'),('Clean original: comparison only','reference')]:
            body.append(f'<div><small>{label}</small><img alt="{label}" src="{image(folder/f"{i:02}_{name}.png")}"></div>')
        body.append('</div></section>')
    body.append('<h2 class="newpage">What repeated feedback looks like</h2><p>First two validation faces, fixed before evaluation. Watch identity/structure drift with extra feedback passes. These examples are not selected for successful outputs.</p>')
    for profile in ['standard28','clean28']:
        folder=OUT/'validation_examples'/profile
        for i in range(2):
            body.append(f'<section><h3>{profile}, validation face{i+1:02}</h3><div class="row">')
            for label,name in [('Input','input'),('One pass','single'),('2 feedback passes','feedback2'),('3 feedback passes','feedback3'),('8 feedback passes','feedback8'),('Clean original','reference')]:
                body.append(f'<div><small>{label}</small><img alt="{label}" src="{image(folder/f"{i:02}_{name}.png")}"></div>')
            body.append('</div></section>')
    body.append('<h2>Measured cost</h2><p>Warm batch1,28x28 crop, '+html.escape(runtime['device'])+'. Medians from30 trials for fixed-size methods and10 for literal cascades; not camera FPS. Small timing differences near one-pass runtime are noise, not evidence that a correction speeds up inference.</p><table><tr><th>Method</th><th>Median ms</th></tr>')
    for method,t in runtime['timings'].items():body.append('<tr><td>'+LABELS[method]+'</td><td>'+f"{t['median_ms']:.2f}"+'</td></tr>')
    body.append('</table><h2>Better next steps</h2><ul><li>Use a larger native face crop with less compression and better exposure. Enlarging the same pixels does not add evidence.</li><li>For video, align and combine several actual frames: these can provide different observations. This requires source frames and a motion-aware video restoration method; it has not been implemented here.</li><li>For learned multi-step restoration, train a feedback/refinement network explicitly for repeated use. This feed-forward model was not trained as a recurrent refiner.</li><li>For the lightweight model, a separate teacher-distillation experiment could transfer useful detail from heavy models, with identity and eye/frame fidelity checks on held-out validation. That is a proposed training study, not a measured gain.</li></ul>')
    body.append('<p>Sources: <a href="https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf">EDSR self-ensemble</a>; <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.pdf">trained SR feedback networks</a>; <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Timofte_Seven_Ways_to_CVPR_2016_paper.pdf">input consistency/back-projection</a>.</p></html>')
    (OUT/'comparison.html').write_text('\n'.join(body),encoding='utf-8')
    md=['# Repeated restoration passes: measured findings','',
        'The preferred v4 weights are unchanged. All-profile validation selected single pass; no generic multi-pass improvement was established.','',
        '- Same input repeated three times: identical outputs, max absolute difference0.0.',
        '- Fixed-size feedback2/3/5/8 usually worsened structure and face-feature similarity. At standard28, PSNR fell25.5607→25.1393→24.5927→22.0085 for1/2/3/8 passes.',
        '- Flip averaging2 slightly improved PSNR/SSIM and face cosine but increased LPIPS on all three profiles. Four/eight transformed views did not satisfy quality guards.',
        '- Literal two/three passes created448/1792 output dimensions. After fair resizing to112, there was no clear quality gain and cost increased.',
        '- A separate clean-only consistency follow-up improved all four average metrics on512 fixed test images. It uses one neural pass plus3 residual corrections, not3 neural passes.','',
        '## Clean-only test diagnostic','',
        '| Method | PSNR-Y | SSIM-Y | LPIPS-Alex ↓ | Face-feature cosine |','|---|---:|---:|---:|---:|']
    for m,v in clean['test_clean28'].items():md.append('| '+LABELS[m]+' | '+' | '.join(f'{v[k]:.5f}' for k in ['psnr_y','ssim_y','lpips_alex','mobileface_cosine'])+' |')
    md += ['', 'Only known paired/simulated clean bicubic inputs may use this optional mode in the UI. Native uploads, noisy inputs and QMUL stay single-pass. Default v4 and both pretrained baselines remain unchanged. No clean reference is used in inference.', '',
           'The clean-only scope was identified from validation and frozen before the512-image test. Historical split reuse remains a limitation. This is a modest conditional result, not a new untouched benchmark, restored-detail percentage, human-recognition result or new full-QMUL verification score.','',
           '[Live report](http://127.0.0.1:8765/diagnostics/passes) | [Portable comparison](comparison.html) | [PDF](comparison.pdf)','',
           '[Original protocol](protocol.json), [evaluation](evaluation.json), [unchanged all-profile selection](selection.json), [runtime samples](runtime.json), [clean-only protocol](clean_only_followup/protocol.json), [clean-only result](clean_only_followup/evaluation.json).']
    (OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Pass comparison report written.')


if __name__=='__main__':main()

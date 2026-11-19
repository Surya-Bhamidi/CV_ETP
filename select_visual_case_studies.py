"""Explicit post-hoc validation illustrations, including counterexamples."""
import html
import json

from compare_three_models import MODELS, post, write
from check_visual_tradeoff import OUT
from refine_model import sha


def main():
    rows = json.loads((OUT / 'per_image.json').read_text())
    def feature_gap(row):
        return min(row['models']['identity_v4']['mobileface_cosine'] - row['models'][m]['mobileface_cosine'] for m in MODELS[1:])
    def perceptual_gap(row):
        return min(row['models'][m]['lpips_alex'] for m in MODELS[1:]) - row['models']['identity_v4']['lpips_alex']
    strength = [r for r in sorted(rows, key=feature_gap, reverse=True) if feature_gap(r) > 0][:2]
    weakness = [r for r in sorted(rows, key=perceptual_gap) if perceptual_gap(r) < 0][:2]
    cases = [{'index': r['index'], 'role': 'v4 face-feature cosine advantage', 'selection_metric': 'mobileface_cosine'} for r in strength]
    cases += [{'index': r['index'], 'role': 'heavy-model perceptual advantage', 'selection_metric': 'lpips_alex'} for r in weakness]
    manifest = {'split': 'validation', 'profile': 'standard28', 'post_hoc_selection': True,
                'source_evaluation_sha256': sha(OUT / 'per_image.json'),
                'rule': 'Two largest v4 cosine advantages over both baselines, followed by two largest v4 LPIPS disadvantages; illustrative selection, not an unbiased evaluation.',
                'not_training_images': True, 'no_model_or_setting_changes': True, 'cases': cases}
    write(OUT / 'case_studies.json', manifest)
    previous = (OUT / 'comparison.html').read_text(encoding='utf-8')
    body = [previous.split('<section class="case">')[0]]
    body.append('<p><strong>The heavier models have lower mean LPIPS; v4 did not beat both on LPIPS on any of these 128 images.</strong> v4 had higher MobileFace cosine than both on 41/128 images. This feature diagnostic does not establish human recognizability. The four illustrations below were selected after evaluation: two feature-similarity strengths and two perceptual weaknesses. The fixed test gallery remains available.</p>')
    fields = ['psnr', 'ssim', 'lpips_alex', 'mobileface_cosine']
    labels = {'identity_v4': 'Ours v4', 'restoration': 'GFPGAN 1.4', 'codeformer': 'CodeFormer w=0.5', 'bicubic': 'Bicubic'}
    for case in cases:
        i = case['index']; row = next(r for r in rows if r['index'] == i)
        outputs = {m: post({'sample_id': i, 'sample_split': 'validation', 'profile': 'standard28', 'model': m}) for m in MODELS}
        a = outputs[MODELS[0]]
        assert all(r['input'] == a['input'] and r['reference'] == a['reference'] for r in outputs.values())
        body += [f'<section class="case"><h2>Validation face {i+1:02}: {html.escape(case["role"])}</h2>',
                 '<p>Selected after measuring validation outputs. The original is comparison-only; every model receives the same degraded input.</p><div class="row">']
        tiles = [('Exact input', a['input']), ('Bicubic', a['bicubic'])] + [(labels[m], outputs[m]['sr']) for m in MODELS] + [('Clean original', a['reference'])]
        for label, url in tiles:
            body.append(f'<div><small>{label}</small><img src="{url}" alt="{label}"></div>')
        body.append('</div><table><tr><th>Model</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>Face-feature cosine ↑</th></tr>')
        for m, values in row['models'].items():
            body.append('<tr><td>' + labels[m] + '</td>' + ''.join(f'<td>{values[k]:.4f}</td>' for k in fields) + '</tr>')
        body.append('</table></section>')
    body.append('<h2>Answer for the presentation</h2><p>Our network has 0.48 million parameters and runs faster in this measured crop pipeline. It achieves higher PSNR/SSIM under the declared degradation, while the heavier models achieve better perceptual-distance scores on average and can look better. Different training objectives and input conditions can change the ranking; more parameters do not guarantee a win on every metric. We have not demonstrated universally better identity recovery.</p>')
    body.append('<p>Training examples would measure fitting, not generalization. These examples are validation images and explicitly selected illustrations, not unseen test evidence. No new training or reconstruction improvement occurred in this analysis.</p>')
    body.append('<p><a href="https://arxiv.org/abs/1711.06077">Perception–distortion tradeoff</a>; <a href="https://github.com/richzhang/PerceptualSimilarity">LPIPS</a>; <a href="https://github.com/sczhou/CodeFormer#testing">CodeFormer quality/fidelity controls</a>.</p></html>')
    (OUT / 'case_studies.html').write_text('\n'.join(body), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

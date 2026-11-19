"""Current three-model UI contract, reference isolation and crop consistency."""
import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image
from playwright.sync_api import expect, sync_playwright

from compare_three_models import MODELS, OUT, ROOT, URL, post, write


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = json.load(urllib.request.urlopen(URL + '/api/status'))
    assert [m['id'] for m in status['models']] == MODELS
    assert status['default_model'] == 'identity_v4'
    review = json.loads((ROOT / 'work/recognition_v4/review.json').read_text())
    assert status['models'][0]['checkpoint_sha256'] == review['checkpoint_sha256']
    assert status['models'][0]['parameters'] == 479109
    assert status['models'][2]['fidelity'] == 0.5
    checks = {'exactly_three_models_v4_default_and_weights_unchanged': True}
    buffer = io.BytesIO()
    Image.new('RGB', (112, 112), 'black').save(buffer, format='PNG')
    black = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    expected_input = expected_reference = ours = None
    for model in MODELS:
        result = post({'sample_id': 0, 'model': model})
        if expected_input is None:
            expected_input, expected_reference, ours = result['input'], result['reference'], result['sr']
        assert (result['input'], result['reference']) == (expected_input, expected_reference)
        for ref in [None, black]:
            isolated = post({'image': expected_input, 'model': model, 'reference': ref})
            assert isolated['sr'] == result['sr'], 'Reference must not affect inference'
        if model != 'identity_v4':
            assert result['previous'] == ours and result['previous_label'] == 'OUR MODEL (V4)'
            try:
                post({'sample_id': 0, 'model': model, 'strength': 0.5})
            except urllib.error.HTTPError as error:
                assert error.code == 400 and 'blending is disabled' in error.read().decode()
            else:
                raise AssertionError('Blended baseline must be rejected')
    for i in range(6):
        shared = None
        for model in MODELS:
            result = post({'sample_id': i, 'sample_split': 'qmul', 'model': model})
            assert result['metrics'] is None and result['reference'] is None
            assert result['output_size'] == [4 * d for d in result['input_size']]
            if shared is None:
                shared = result['input']
            assert result['input'] == shared
    checks.update(reference_never_changes_any_model_output=True,
                  baseline_blending_disabled=True, identical_native_qmul_crops_for_all_three=True,
                  no_fabricated_qmul_quality_metrics=True, ours_preview_matches_actual_ours_output=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True, args=['--disable-gpu'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(URL, wait_until='networkidle')
        expect(page.locator('#result-name')).to_have_text('Validation face 32', timeout=120000)
        expect(page.locator('.sample-button')).to_have_count(4)
        expect(page.locator('#gallery-help')).to_contain_text('Selected after validation measurement')
        expect(page.locator('#model-choice')).to_have_value('identity_v4')
        expect(page.locator('#model-choice option')).to_have_count(3)
        assert page.locator('#restoration-strength').count() == 0
        for model in MODELS[1:]:
            page.locator('#model-choice').select_option(model)
            expect(page.locator('#model-description')).to_contain_text('without blending', timeout=120000)
            expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
            page.locator('#previous-option').click()
            expect(page.locator('#before-tag')).to_have_text('OUR MODEL (V4)')
        page.locator('#sample-split').select_option('qmul')
        expect(page.locator('#result-name')).to_have_text('QMUL camera crop 01', timeout=120000)
        expect(page.locator('#model-choice')).to_have_value('codeformer')
        expect(page.locator('#original-option')).to_be_disabled()
        page.locator('[data-layout="side"]').click()
        page.screenshot(path=str(OUT / 'desktop_baseline.png'), full_page=True)
        page.goto(URL + '/?split=test&compare=reference', wait_until='networkidle')
        expect(page.locator('#model-choice')).to_have_value('identity_v4', timeout=120000)
        expect(page.locator('#before-tag')).to_have_text('CLEAN ORIGINAL', timeout=120000)
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        page.screenshot(path=str(OUT / 'desktop_ours.png'), full_page=True)
        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        mobile.on('pageerror', lambda error: errors.append(str(error)))
        mobile.goto(URL + '/?model=codeformer&split=qmul&compare=previous', wait_until='networkidle')
        expect(mobile.locator('#result-name')).to_have_text('QMUL camera crop 01', timeout=120000)
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth')
        mobile.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
        report = OUT / 'comparison.html'
        if report.exists():
            page.goto(report.as_uri(), wait_until='load')
            assert page.locator('img').count() == 174
            assert page.locator('img').evaluate_all('(items)=>items.every(i=>i.complete&&i.naturalWidth>0)')
            page.emulate_media(media='print')
            page.pdf(path=str(OUT / 'comparison.pdf'), format='A4', landscape=True, print_background=True,
                     margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'})
            checks['all_174_report_images_loaded_and_pdf_exported'] = True
        assert not errors, errors
        browser.close()
    checks['desktop_mobile_three_choices_and_retired_link_fallback'] = True
    write(OUT / 'ui_checks.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()

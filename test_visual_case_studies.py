"""Validate labeled post-hoc case studies without replacing test evidence."""
import json
import urllib.request

from playwright.sync_api import expect, sync_playwright

from compare_three_models import MODELS, URL, post, write
from check_visual_tradeoff import OUT


def main():
    manifest = json.loads((OUT / 'case_studies.json').read_text())
    indices = [c['index'] for c in manifest['cases']]
    assert manifest['post_hoc_selection'] is True and indices == [31, 72, 42, 41]
    gallery = json.load(urllib.request.urlopen(URL + '/api/samples?split=validation&profile=standard28&focus=cases'))['samples']
    assert [s['id'] for s in gallery] == indices and all(s['case_study'] for s in gallery)
    for index in indices:
        a = post({'sample_id': index, 'sample_split': 'validation', 'model': MODELS[0]})
        for model in MODELS[1:]:
            b = post({'sample_id': index, 'sample_split': 'validation', 'model': model})
            assert a['input'] == b['input'] and a['reference'] == b['reference']
    neutral = json.load(urllib.request.urlopen(URL + '/api/samples?split=test'))['samples']
    assert [s['id'] for s in neutral] == list(range(12)) and all(s['case_study'] is None for s in neutral)
    checks = {'selected_validation_examples_and_roles_explicit': True,
              'fixed_test_gallery_preserved': True, 'same_input_and_reference_for_all_models': True}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True, args=['--disable-gpu'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(URL + '/?compare=reference', wait_until='networkidle')
        expect(page.locator('#result-name')).to_have_text('Validation face 32', timeout=120000)
        expect(page.locator('.sample-button')).to_have_count(4)
        expect(page.locator('#gallery-help')).to_contain_text('Selected after validation measurement')
        expect(page.locator('#result-note')).to_contain_text('not an unbiased test example')
        expect(page.locator('#model-choice option')).to_have_count(3)
        expect(page.locator('#sample-split')).to_have_value('validation')
        page.screenshot(path=str(OUT / 'ui_cases.png'), full_page=True)
        page.locator('#model-choice').select_option('restoration')
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        expect(page.locator('#result-name')).to_have_text('Validation face 32')
        page.locator('#quality-profile').select_option('clean28')
        expect(page.locator('.sample-button')).to_have_count(12, timeout=120000)
        expect(page.locator('#gallery-help')).not_to_contain_text('Selected after')
        page.locator('#validation-cases').click()
        expect(page.locator('.sample-button')).to_have_count(4, timeout=120000)
        expect(page.locator('#sample-split')).to_have_value('validation')
        expect(page.locator('#quality-profile')).to_have_value('standard28')
        page.goto(URL + '/diagnostics/visual', wait_until='load')
        assert page.locator('section.case').count() == 4 and page.locator('img').count() == 24
        assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
        page.locator('section.case').first.screenshot(path=str(OUT / 'first_case.png'))
        page.emulate_media(media='print')
        page.pdf(path=str(OUT / 'case_studies.pdf'), format='A4', landscape=True, print_background=True,
                 margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'})
        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        mobile.goto(URL, wait_until='networkidle')
        expect(mobile.locator('#result-name')).to_have_text('Validation face 32', timeout=120000)
        assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.goto(URL + '/?split=test', wait_until='networkidle')
        expect(page.locator('#result-name')).to_have_text('Test face 01', timeout=120000)
        expect(page.locator('.sample-button')).to_have_count(12)
        page.goto(URL + '/?focus=eyes', wait_until='networkidle')
        expect(page.locator('#sample-split')).to_have_value('test', timeout=120000)
        expect(page.locator('#eye-examples')).to_have_count(0)
        expect(page.locator('#result-name')).to_have_text('Test face 01')
        assert not errors, errors
        browser.close()
    checks.update(selected_cases_default_on_direct_launch=True, removed_eyewear_link_falls_back_to_fixed_gallery=True,
                  case_gallery_switches_are_explicit=True, all_report_images_and_pdf_valid=True,
                  desktop_mobile_and_browser_errors_checked=True)
    write(OUT / 'ui_checks.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()

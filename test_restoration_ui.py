"""Regression checks for external restoration, strength and reference isolation."""
import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    out = ROOT / 'work/ui'
    checks = {}
    def post(body):
        req = urllib.request.Request(url + '/api/compare', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=120))
    def image(data):
        return np.array(Image.open(io.BytesIO(base64.b64decode(data.split(',', 1)[1]))))
    for _ in range(90):
        status = json.load(urllib.request.urlopen(url + '/api/status'))
        if status['state'] == 'ready': break
        if status['state'] == 'error': raise AssertionError(status)
        time.sleep(1)
    assert status['state'] == 'ready'
    info = {item['id']: item for item in status['models']}
    assert status['default_model'] == ('detail' if info.get('detail',{}).get('reconstruction_validated') else 'standard')
    assert info['restoration']['kind'] == 'external_pretrained' and info['restoration']['epochs'] is None
    old = post({'sample_id':5, 'model':'standard'})
    zero = post({'sample_id':5, 'model':'restoration', 'strength':0})
    full = post({'sample_id':5, 'model':'restoration', 'strength':1})
    blend = post({'sample_id':5, 'model':'restoration', 'strength':.75})
    assert zero['sr'] == old['sr']
    assert full['sr'] != old['sr']
    assert full['input'] == old['input'] and full['reference'] == old['reference']
    assert np.abs(image(blend['sr']).astype(float) - (.25 * image(old['sr']).astype(float) + .75 * image(full['sr']).astype(float))).max() <= 1.1
    checks['strength_endpoints_and_real_pixel_blend'] = True
    checks['same_input_and_reference_for_all_models'] = True
    # A supplied clean reference must never affect restoration output.
    payload = {'image':old['input'], 'model':'restoration', 'strength':.75}
    unpaired = post(payload)
    black = io.BytesIO(); Image.new('RGB',(112,112)).save(black,format='PNG')
    other_reference = 'data:image/png;base64,' + base64.b64encode(black.getvalue()).decode()
    paired = post({**payload, 'reference':other_reference})
    assert np.abs(image(unpaired['sr']).astype(float) - image(paired['sr']).astype(float)).max() <= 1
    assert unpaired['metrics'] is None and paired['metrics'] is not None
    checks['clean_reference_never_changes_restoration'] = True
    for value in [-1, 1.1, True, '0.5']:
        try: post({'sample_id':0, 'model':'restoration','strength':value})
        except urllib.error.HTTPError as error: assert error.code == 400
        else: raise AssertionError('Invalid strength accepted')
    rectangle = io.BytesIO(); Image.new('RGB',(20,28)).save(rectangle,format='PNG')
    try:
        post({'model':'restoration','image':'data:image/png;base64,'+base64.b64encode(rectangle.getvalue()).decode()})
    except urllib.error.HTTPError as error:
        assert error.code == 400 and 'square' in error.read().decode()
    else: raise AssertionError('Non-square face was distorted silently')
    checks['invalid_strength_and_non_square_crop_rejected'] = True
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
        page = browser.new_page(viewport={'width':1440,'height':1150})
        errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(url+'/?model=restoration&sample=5&compare=previous&strength=0.75',wait_until='networkidle')
        expect(page.locator('#result-name')).to_have_text('Test face 06',timeout=120000)
        expect(page.locator('#restoration-controls')).to_be_visible()
        expect(page.locator('#model-warning')).to_contain_text('separate from our trained SR model')
        expect(page.locator('#epoch-count')).to_have_text('externally pretrained')
        expect(page.locator('#before-tag')).to_have_text('ORIGINAL CUSTOM SR')
        expect(page.locator('#after-tag-text')).to_have_text('RESTORATION ×4')
        expect(page.locator('#strength-value')).to_have_text('75%')
        before=page.locator('#after-image').get_attribute('src')
        page.locator('#restoration-strength').evaluate("el=>{el.value='1';el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));}")
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        expect(page.locator('#strength-value')).to_have_text('100%')
        assert page.locator('#after-image').get_attribute('src') != before
        page.screenshot(path=str(out/'restoration_desktop.png'),full_page=True)
        page.locator('#model-choice').select_option('standard')
        expect(page.locator('#restoration-controls')).to_be_hidden(timeout=120000)
        expect(page.locator('#model-warning')).to_be_hidden()
        expect(page.locator('#epoch-count')).to_have_text('30 epochs')
        checks['browser_model_provenance_strength_and_switching'] = True
        if 'codeformer' in info:
            page.locator('#model-choice').select_option('codeformer')
            expect(page.locator('#model-description')).to_contain_text('maximum CodeFormer fidelity',timeout=120000)
            expect(page.locator('#model-warning')).to_contain_text('separate pretrained research model')
            expect(page.locator('#restoration-controls')).to_be_visible()
            expect(page.locator('#selected-model-label')).to_have_text('CodeFormer pretrained model')
            assert page.locator('#after-image').get_attribute('src') != full['sr']
            page.screenshot(path=str(out/'codeformer_desktop.png'),full_page=True)
            checks['codeformer_uses_a_separate_real_pretrained_network'] = True
        mobile=browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
        mobile.on('pageerror',lambda error:errors.append(str(error)))
        mobile.goto(url+'/?model=restoration&sample=5',wait_until='networkidle')
        expect(mobile.locator('#restoration-controls')).to_be_visible(timeout=120000)
        assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        mobile.screenshot(path=str(out/'restoration_mobile.png'),full_page=True)
        assert not errors,errors
        checks['restoration_mobile_no_overflow_or_javascript_errors'] = True
        browser.close()
    (out/'restoration_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)


if __name__ == '__main__':
    main()

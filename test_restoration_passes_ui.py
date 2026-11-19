"""Check real clean-only UI scope, unchanged defaults and transform geometry."""
import json
import urllib.error
import urllib.request

import torch
from torch.nn import functional as F
from playwright.sync_api import expect, sync_playwright

from compare_three_models import MODELS, URL, post, write
from evaluate_restoration_passes import OUT
from restoration_passes import predict


def rejected(payload):
    try:
        post(payload)
    except urllib.error.HTTPError as error:
        assert error.code == 400, error.read().decode()
    else:
        raise AssertionError('Unsupported clean correction scope must be rejected')


def main():
    status=json.load(urllib.request.urlopen(URL+'/api/status'))
    assert status['clean_consistency_available'] and [m['id'] for m in status['models']]==MODELS
    raw=post({'sample_id':0,'profile':'clean28'})
    corrected=post({'sample_id':0,'profile':'clean28','inference':'clean_consistency'})
    assert raw['inference_mode']=='single' and raw['correction_iterations']==0
    assert corrected['input']==raw['input'] and corrected['reference']==raw['reference']
    assert corrected['output_size']==raw['output_size']==[112,112]
    assert corrected['previous']==raw['sr'] and corrected['previous_label']=='SINGLE-PASS V4'
    assert corrected['sr']!=raw['sr'] and corrected['neural_passes']==1 and corrected['correction_iterations']==3
    assert post({'sample_id':0,'profile':'clean28','inference':'clean_consistency'})['sr']==corrected['sr']
    for payload in [
        {'sample_id':0,'sample_split':'qmul'},
        {'sample_id':0,'profile':'standard28'},
        {'sample_id':0,'profile':'mild28'},
        {'sample_id':0,'profile':'clean28','model':'restoration'},
        {'sample_id':0,'profile':'clean28','model':'codeformer'},
        {'image':raw['input'],'reference':raw['reference']}]:
        rejected({**payload,'inference':'clean_consistency'})
    simulation=post({'image':raw['reference'],'mode':'simulate','profile':'clean28','inference':'clean_consistency'})
    assert simulation['inference_mode']=='clean_consistency' and simulation['input']==raw['input']
    class EquivariantUpsampler(torch.nn.Module):
        def forward(self,x):return F.interpolate(x,scale_factor=4,mode='nearest')
    model=EquivariantUpsampler();x=torch.rand(1,3,3,5)
    for count in [2,4,8]:
        torch.testing.assert_close(predict(model,x,f'ensemble{count}'),model(x),atol=1e-6,rtol=1e-6)
    for method in ['feedback2','feedback3','feedback5','feedback8','cascade2','cascade3','backprojection3']:
        value=predict(model,x,method)
        assert value.shape==(1,3,12,20) and torch.isfinite(value).all()
    checks={'three_models_and_single_default_unchanged':True,'clean_mode_changes_output_on_identical_input':True,
            'clean_output_compares_with_actual_single_pass':True,'one_neural_pass_three_corrections_labeled':True,
            'native_qmul_noisy_and_external_scopes_rejected':True,'known_clean_simulation_supported':True,
            'non_square_transform_inversion_and_output_shapes':True}
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL+'/?split=test&profile=clean28&inference=clean_consistency&compare=previous',wait_until='networkidle')
        expect(page.locator('#result-name')).to_have_text('Test face 01',timeout=120000)
        expect(page.locator('#inference-mode')).to_have_value('clean_consistency')
        expect(page.locator('#before-tag')).to_have_text('SINGLE-PASS V4')
        expect(page.locator('#previous-option')).to_have_text('Single-pass v4')
        page.locator('[data-layout="side"]').click();page.screenshot(path=str(OUT/'ui_clean.png'),full_page=True)
        page.locator('#model-choice').select_option('restoration')
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        expect(page.locator('#inference-mode')).to_have_value('single')
        expect(page.locator('#inference-mode')).to_be_disabled()
        page.locator('#model-choice').select_option('identity_v4')
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        page.locator('#quality-profile').select_option('standard28')
        expect(page.locator('#loading-overlay')).to_be_hidden(timeout=120000)
        expect(page.locator('#inference-mode')).to_be_disabled()
        page.goto(URL+'/diagnostics/passes',wait_until='load')
        assert page.locator('img').count()==48
        assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
        page.emulate_media(media='print');page.pdf(path=str(OUT/'comparison.pdf'),format='A4',landscape=True,print_background=True,
            margin={'top':'12mm','bottom':'12mm','left':'12mm','right':'12mm'})
        mobile=browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
        mobile.goto(URL+'/?split=test&profile=clean28&inference=clean_consistency',wait_until='networkidle')
        expect(mobile.locator('#result-name')).to_have_text('Test face 01',timeout=120000)
        expect(mobile.locator('#inference-mode')).to_have_value('clean_consistency')
        assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors;browser.close()
    checks['browser_mode_scope_desktop_mobile_and_report_export']=True
    write(OUT/'ui_checks.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

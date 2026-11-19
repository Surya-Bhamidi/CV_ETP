"""Independent timing spot-check with raw samples and actual browser fetches."""
import base64
import datetime
import io
import json
import time

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import sync_playwright
from torch.nn import functional as F

from field_native_check import OUT,CHECKPOINTS
from investigate_identity import load
from measure_field_latency import statistics,tensor
from refine_model import FrozenFace,sha,write_json
from run_project import ROOT,runtime


@torch.inference_mode()
def main():
    if (OUT/'runtime_spotcheck.json').exists():raise ValueError('Preserve raw timing evidence')
    latency=json.loads((OUT/'latency.json').read_text());manifest=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    by_name=dict(zip(manifest['files'],manifest['source_paths']));cases={}
    for name in ['median_native','larger_native']:
        with Image.open(ROOT/by_name[latency['cases'][name]['file']]) as image:cases[name]=np.array(image.convert('RGB'),copy=True)
    n,_=runtime();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load(n,CHECKPOINTS['identity_v4']);face=FrozenFace(ROOT/'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    result={'source_sha256':sha(__file__),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'latency_run_sha256':sha(OUT/'latency.json'),'checkpoint_sha256':sha(CHECKPOINTS['identity_v4']),
        'scope':'Warm batch1 replay. GPU wall timing synchronized. Browser fetch includes server upload handling, two comparison SR models, output PNG/base64 and JSON parse, excludes display rendering/capture/detection/tracking.',
        'gpu':{},'browser_fetch':{}}
    def feature_pipeline(array):
        low=tensor(array,n.device);sr=(model(low).clamp(0,1)*255).round()/255
        normalized=F.interpolate(sr,size=(112,112),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        return face(normalized).cpu().numpy()
    for name,array in cases.items():
        low=tensor(array,n.device);result['gpu'][name]={}
        for key,call in [('sr_only',lambda:model(low).clamp(0,1)),('sr_plus_single_mobileface',lambda:feature_pipeline(array))]:
            for _ in range(8):call()
            torch.cuda.synchronize();values=[]
            for _ in range(50):
                torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000);del value
            result['gpu'][name][key]={'raw_ms':values,'summary':statistics(values)}
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu']);page=browser.new_page()
        page.goto('http://127.0.0.1:8765/?split=qmul&model=identity_v4',wait_until='networkidle')
        for name,array in cases.items():
            buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG')
            payload={'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),'model':'identity_v4'}
            values=page.evaluate('''async payload=>{
              const times=[],compute=[];
              for(let i=0;i<33;i++){
                const start=performance.now();
                const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
                if(!response.ok)throw new Error('Unexpected response '+response.status);
                const value=await response.json();
                if(value.model_id!=='identity_v4')throw new Error('Wrong model');
                const elapsed=performance.now()-start;
                if(i>=3){times.push(elapsed);compute.push(value.processing_ms);}
              }
              return {times,compute};
            }''',payload)
            result['browser_fetch'][name]={'raw_round_trip_ms':values['times'],'raw_server_comparison_ms':values['compute'],
                'round_trip_summary':statistics(values['times']),'server_comparison_summary':statistics(values['compute'])}
            print('BROWSER FETCH',name,json.dumps(result['browser_fetch'][name]['round_trip_summary']),flush=True)
        browser.close()
    write_json(OUT/'runtime_spotcheck.json',result);print('RAW RUNTIME SPOTCHECK COMPLETE',flush=True)


if __name__=='__main__':main()

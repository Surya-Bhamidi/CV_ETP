"""Warm batch1 per-crop GPU/CPU and uncached loopback-app measurements.

Run after native replay has completed, without another active GPU experiment.
Per-crop reciprocal latency is not full-camera/video detector/tracker FPS.
"""
import base64
import io
import json
import platform
import time
import types
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from field_native_check import OUT,CHECKPOINTS
from investigate_identity import load
from refine_model import FrozenFace,sha,write_json
from run_project import ROOT,runtime

URL='http://127.0.0.1:8765'


def statistics(values):
    return {'samples':len(values),'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
            'min_ms':float(np.min(values)),'mean_ms':float(np.mean(values)),
            'median_reciprocal_crops_per_second':float(1000/np.median(values)),
            'within_30hz_crop_budget_at_p95':bool(np.percentile(values,95)<=1000/30)}


def timed(call,count,device,warm=8):
    for _ in range(warm):call()
    if device=='cuda':torch.cuda.synchronize()
    values=[]
    for _ in range(count):
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();value=call()
        if device=='cuda':torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000);del value
    return statistics(values)


def png_url(array):
    stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def post(payload):
    request=urllib.request.Request(URL+'/api/compare',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(request,timeout=120))


def tensor(array,device):return torch.from_numpy(np.array(array,copy=True)).permute(2,0,1).unsqueeze(0).float().div(255).contiguous().to(device)


@torch.inference_mode()
def main():
    if not (OUT/'native_results.json').exists():raise RuntimeError('Complete native replay before isolated timing')
    if (OUT/'latency.json').exists():raise ValueError('Preserve recorded timing run')
    n,_=runtime();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    sizes=np.array(manifest['native_sizes']);cases={}
    for label,target in [('tiny_native',(12,15)),('median_native',(19,25)),('larger_native',(59,80)),('large_native',(94,116))]:
        eligible=sizes.min(1)>=8;distance=np.abs(sizes-np.array(target)).sum(1).astype(float);distance[~eligible]=np.inf
        index=int(distance.argmin())
        with Image.open(ROOT/manifest['source_paths'][index]) as image:array=np.array(image.convert('RGB'),copy=True)
        cases[label]={'array':array,'file':manifest['files'][index],'native':True}
    source=cases['median_native']['array']
    cases['lr28_control']={'array':np.array(Image.fromarray(source).resize((28,28),Image.Resampling.BICUBIC)),
                          'file':cases['median_native']['file'],'native':False,'note':'Forced28 control only; no additional detail'}
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    cases['max256_stress']={'array':np.array(Image.fromarray(np.array(hr[0],copy=True)).resize((256,256),Image.Resampling.BICUBIC)),
                           'file':'FFHQ0 enlarged solely for upload-limit stress timing','native':False}
    models={key:load(n,CHECKPOINTS[key]) for key in ['camera_v3','identity_v4']}
    face=FrozenFace(ROOT/'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    result={'source_sha256':sha(__file__),'hardware':{'gpu':torch.cuda.get_device_name(0) if n.device=='cuda' else None,
        'cpu':platform.processor(),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'device':n.device},
        'protocol':{'precision':'FP32, TF32 disabled','batch_size':1,'warmup':8,
            'gpu_sr_core':'Prepared tensor already on GPU; forward+clamp; synchronized host wall time. No decode/encode, transfer, recognizer or baseline-comparison inference.',
            'gpu_full_crop_pipeline':'CPU RGB tensor conversion and transfer, v4 SR,8bit rounding,112 resize, MobileFace single or mirrored features, embedding copied to CPU. Excludes camera capture, full-frame detection/tracking and image decode/encode.',
            'http_app':'Uncached /api/compare upload, no sample_id; includes loopback transport, server decode, both previous+selected SR comparisons, PNG output encoding and client JSON parsing. No recognition in this app endpoint; no capture/detector/tracker/rendering.',
            'fps_scope':'Reciprocal per-crop latency only, not measured camera/video FPS; no sustained full-stream or multi-face test.'},'cases':{},'gpu':{},'cpu':{},'http':{}}
    for name,case in cases.items():result['cases'][name]={key:value for key,value in case.items() if key!='array'}|{'input_size':list(case['array'].shape[1::-1])}
    def pipeline(array,model,recognizer,device,mirror):
        low=tensor(array,device);sr=model(low).clamp(0,1)
        sr=(sr*255).round().clamp(0,255)/255
        image=F.interpolate(sr,size=(112,112),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        feature=recognizer(image)
        if mirror:feature=F.normalize(feature+recognizer(image.flip(-1)),dim=1)
        return feature.cpu().numpy()
    for name,case in cases.items():
        array=case['array'];low=tensor(array,n.device);result['gpu'][name]={}
        for key,model in models.items():result['gpu'][name][key+'_sr_only']=timed(lambda:model(low).clamp(0,1),50 if name!='max256_stress' else 20,n.device)
        for mirror in [False,True]:
            result['gpu'][name]['v4_plus_mobileface_'+('mirror' if mirror else 'single')]=timed(lambda:pipeline(array,models['identity_v4'],face,n.device,mirror),30 if name!='max256_stress' else 15,n.device)
        print('GPU CROP TIMING',name,json.dumps(result['gpu'][name]),flush=True)
        write_json(OUT/'latency_progress.json',result)
    # CPU measurements use the same two Torch threads as the comparison app.
    cpu_namespace=types.SimpleNamespace(SRNet=n.SRNet,cfg=n.cfg,device='cpu')
    cpu_model=load(cpu_namespace,CHECKPOINTS['identity_v4']);face.to('cpu')
    for name in ['median_native','lr28_control','larger_native']:
        array=cases[name]['array'];low=tensor(array,'cpu')
        result['cpu'][name]={'v4_sr_only':timed(lambda:cpu_model(low).clamp(0,1),20,'cpu',warm=4),
            'v4_plus_mobileface_mirror':timed(lambda:pipeline(array,cpu_model,face,'cpu',True),20,'cpu',warm=4)}
        print('CPU CROP TIMING',name,json.dumps(result['cpu'][name]),flush=True)
    status=json.load(urllib.request.urlopen(URL+'/api/status'));assert status['state']=='ready'
    result['http']['app_models_ready']=[m['id'] for m in status['models']]
    output_checks={}
    for name in ['median_native','lr28_control','larger_native','max256_stress']:
        array=cases[name]['array'];result['http'][name]={}
        for key,model_id in [('camera_v3','restoration_v3'),('identity_v4','identity_v4')]:
            payload={'image':png_url(array),'model':model_id};post(payload);post(payload)
            values=[];reported=[];response=None
            for _ in range(20 if name!='max256_stress' else 8):
                start=time.perf_counter();response=post(payload);values.append((time.perf_counter()-start)*1000)
                reported.append(response['processing_ms'])
            result['http'][name][key]={'round_trip':statistics(values),'server_comparison_compute_ms':statistics(reported)}
            expected=(models[key](tensor(array,n.device)).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            actual=np.array(Image.open(io.BytesIO(base64.b64decode(response['sr'].split(',',1)[1]))))
            difference=np.abs(actual.astype(float)-expected.astype(float));assert difference.max()<=2 and difference.mean()<=.1
            output_checks[name+'_'+key]={'same_x4_shape':list(actual.shape)==[array.shape[0]*4,array.shape[1]*4,3],
                'max_pixel_difference_255':float(difference.max()),'mean_pixel_difference_255':float(difference.mean())}
            assert output_checks[name+'_'+key]['same_x4_shape']
        print('UNCACHED APP TIMING',name,json.dumps(result['http'][name]),flush=True)
    result['api_vs_offline_pixel_checks']=output_checks
    write_json(OUT/'latency.json',result);print('FIELD LATENCY COMPLETE',flush=True)


if __name__=='__main__':main()

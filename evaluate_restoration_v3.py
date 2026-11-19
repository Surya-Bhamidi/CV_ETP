"""Frozen comparison with train-validation-only threshold calibration.

No test thresholds, epochs or release gates are selected here. The additional
cohort and checkpoint are locked before any scores are computed.
"""
import collections
import io
import itertools
import json
import zipfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from audit_restoration_leakage import exact_key
from input_quality import InputDegradation
from investigate_identity import load, save_grid
from quality_metric import load_lpips, load_lpips_alex
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, notebook_definitions, runtime
from train_qmul_domain import CameraSet
from train_restoration_v3 import OUT, BASE


def extra_pairs(paths, hashes):
    groups=collections.defaultdict(list)
    for i,path in enumerate(paths):groups[path.rsplit('/',1)[-1].split('_')[0]].append(i)
    keys=list(groups);positions={key:i for i,key in enumerate(keys)}
    positive=[];negative=[];anchors=[];removed=0
    rng=np.random.default_rng(20261119)
    for person,ids in groups.items():
        for a,b in itertools.combinations(ids,2):
            if hashes[a]==hashes[b]:removed+=1;continue
            other=keys[(positions[person]+int(rng.integers(1,len(keys))))%len(keys)]
            options=[x for x in groups[other] if hashes[x]!=hashes[a]]
            if not options:raise RuntimeError('Cross-identity identical crop in diagnostic set')
            positive.append((a,b));negative.append((a,int(rng.choice(options))));anchors.append(person)
    values=np.array(positive+negative)
    return {'ia':values[:,0],'ib':values[:,1],'labels':np.r_[np.ones(len(positive)),np.zeros(len(negative))],
            'anchor_id':np.array(anchors+anchors),'identical_positive_pairs_excluded':removed}


@torch.inference_mode()
def embeddings(n,model,images,faces,batch=32):
    bank={key:[] for key in faces};examples=[]
    example_ids=np.linspace(0,len(images)-1,6,dtype=int)
    for offset in range(0,len(images),batch):
        low=torch.from_numpy(np.array(images[offset:offset+batch],copy=True)).permute(0,3,1,2).float().to(n.device)/255
        sr=(F.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False) if model is None else model(low)).clamp(0,1)
        assert sr.shape==(len(low),3,112,112) and torch.isfinite(sr).all()
        for name,face in faces.items():bank[name].append(face(sr).cpu().numpy())
        for i in example_ids:
            if offset<=i<offset+len(low):examples.append(sr[i-offset].cpu())
    return {key:np.concatenate(value) for key,value in bank.items()},torch.stack(examples)


def metrics(scores,labels,threshold):
    correct=(scores>=threshold)==labels
    return {'accuracy':float(correct.mean()),'auc':float(roc_auc_score(labels,scores)),
            'threshold_from_training_validation':float(threshold),'correct_pairs':int(correct.sum()),'pairs':len(labels)}


@torch.inference_mode()
def main():
    result_path=OUT/'evaluation.json'
    if result_path.exists():raise ValueError('Preserve frozen evaluation; do not silently rerun or retune')
    selection=json.loads((OUT/'selection.json').read_text())
    new=selection['selected_new_checkpoint']
    selected=ROOT/selection['checkpoint']
    if sha(selected)!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    additional=json.loads((ROOT/'work/leakage_review_v2/additional_cohort.json').read_text())
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    model_paths={'balanced_previous':BASE}
    if new:model_paths['restoration_v3']=selected
    frozen={'checkpoint_sha256':{k:sha(v) for k,v in model_paths.items()},'additional_cohort_sha256':sha(ROOT/'work/leakage_review_v2/additional_cohort.json'),
        'source_sha256':sha(__file__),'selection_sha256':sha(OUT/'selection.json'),
        'threshold_calibration':'Only 512 crops from 256 official-training validation identities; never the verification/test fold scores.',
        'release_selection':'Already frozen in review.json before evaluation; results do not alter availability.',
        'lpips_alex_excluded_from_training_and_selection':True,
        'recognizer_sha256':{k:sha(ROOT/p) for k,p in {'arcface':'work/recognition/w600k_r50.onnx','mobileface':'work/refinement_assets/w600k_mbf.onnx'}.items()}}
    write_json(OUT/'frozen_evaluation.json',frozen)
    n,_=runtime();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'v3_threshold_protocol')
    faces={k:FrozenFace(ROOT/p).to(n.device).eval().requires_grad_(False) for k,p in {
        'arcface':'work/recognition/w600k_r50.onnx','mobileface':'work/refinement_assets/w600k_mbf.onnx'}.items()}
    camera=CameraSet(ROOT/domain['source_archive'],domain['validation_files'])
    calibration=np.stack([camera.pair(i)[0] for i in range(len(camera.files))])
    ia=np.arange(0,len(calibration),2);ib=ia+1
    calib_pairs={'ia':np.r_[ia,ia],'ib':np.r_[ib,np.roll(ib,-1)],'labels':np.r_[np.ones(len(ia)),np.zeros(len(ia))]}
    low_extra=[];hashes=[]
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        for path in additional['files']:
            with Image.open(io.BytesIO(zipped.read(path))) as image:
                image=image.convert('RGB');hashes.append(exact_key(np.asarray(image)))
                low_extra.append(np.asarray(image.resize((28,28),Image.Resampling.BICUBIC)))
    low_extra=np.stack(low_extra);extra=extra_pairs(additional['files'],hashes)
    with np.load(ROOT/'work/qmul_study/cache/pairs.npz') as data:old_pairs={k:data[k] for k in ['ia','ib','labels']}
    original=np.load(ROOT/'work/qmul_study/cache/lr28.npy',mmap_mode='r')
    result={'protocol':frozen,'qmul':{},'paired_ffhq':{},'additional_images':len(low_extra),
        'additional_identities':len(additional['identities']),'additional_pairs':len(extra['labels']),
        'additional_identical_positive_pairs_excluded':extra['identical_positive_pairs_excluded'],
        'additional_scope':additional['scope'],'selection_was_validation_only':True}
    example_rows={'Actual input (28)':torch.from_numpy(low_extra[np.linspace(0,len(low_extra)-1,6,dtype=int)]).permute(0,3,1,2).float()/255}
    (OUT/'emb').mkdir(exist_ok=True)
    for name,path in [('bicubic',None),*model_paths.items()]:
        model=None if path is None else load(n,path)
        cal,_=embeddings(n,model,calibration,faces)
        thresholds={key:r.VerificationEval.best_threshold((value[calib_pairs['ia']]*value[calib_pairs['ib']]).sum(1),calib_pairs['labels']) for key,value in cal.items()}
        result['qmul'][name]={}
        for dataset,images,pairs in [('prior_9913_images',original,old_pairs),('additional_identity_disjoint',low_extra,extra)]:
            bank,examples=embeddings(n,model,images,faces)
            scores={key:(value[pairs['ia']]*value[pairs['ib']]).sum(1) for key,value in bank.items()}
            result['qmul'][name][dataset]={key:metrics(value,pairs['labels'],thresholds[key]) for key,value in scores.items()}
            np.savez(OUT/'emb'/f'{name}_{dataset}.npz',labels=pairs['labels'],**{key+'_scores':value for key,value in scores.items()},
                     **({'anchor_id':pairs['anchor_id']} if 'anchor_id' in pairs else {}))
            if dataset=='additional_identity_disjoint':example_rows[name]=examples
        print('V3 RECOGNITION',name,json.dumps(result['qmul'][name]),flush=True)
        del model;torch.cuda.empty_cache()
    save_grid(example_rows,OUT/'qmul_additional_comparison.png','Additional identity-disjoint QMUL diagnostic | no clean HR original')
    del faces;torch.cuda.empty_cache()
    models={name:load(n,path) for name,path in model_paths.items()}
    squeeze=load_lpips(n.device);alex=load_lpips_alex(n.device)
    # Independent evaluation backbone: identical images must have zero distance.
    sample=torch.rand(1,3,112,112,device=n.device)
    assert abs(float(alex(sample,sample,normalize=True)))<1e-7
    result['lpips_alex_asset_sha256']=sha(ROOT/'work/quality_assets/alexnet-owt-7be5be79.pth')
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as data:test=data['test'][:512]
    for profile in ['standard28','clean28']:
        data=n.FixedEvalSet(hr,test,InputDegradation(n,profile),112,2000)
        scores={name:{k:[] for k in ['psnr_y','ssim_y','lpips_squeeze','lpips_alex']} for name in models}
        rows=None
        for low,target in data.batches(32):
            low,target=low.to(n.device),target.to(n.device)
            views={name:model(low).clamp(0,1) for name,model in models.items()}
            for name,sr in views.items():
                scores[name]['psnr_y'].extend(n.Metrics.psnr_y(sr,target,4).cpu().tolist())
                scores[name]['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(),target.cpu(),4).tolist())
                scores[name]['lpips_squeeze'].extend(squeeze(sr,target,normalize=True).flatten().cpu().tolist())
                scores[name]['lpips_alex'].extend(alex(sr,target,normalize=True).flatten().cpu().tolist())
            if rows is None:rows={'Clean original':target.cpu(),'Actual input (28)':low.cpu(),**{k:v.cpu() for k,v in views.items()}}
        result['paired_ffhq'][profile]={name:{k:float(np.mean(v)) for k,v in values.items()} for name,values in scores.items()}
        write_json(OUT/(profile+'_per_image.json'),scores)
        save_grid(rows,OUT/(profile+'_comparison.png'),'Same held-out FFHQ test images | '+profile+' | no reference enters inference')
        print('V3 PAIRED',profile,json.dumps(result['paired_ffhq'][profile]),flush=True)
    write_json(result_path,result);print('V3 EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()

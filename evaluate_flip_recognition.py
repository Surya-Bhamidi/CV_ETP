"""One fixed recognition-only mirrored-feature ensemble, original512 calibration.

The selected v4 SR pixels/weights are unchanged. This is not a new restoration
network or a new image-restoration percentage. No test threshold is selected.
"""
import io
import json
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from audit_restoration_leakage import exact_key
from evaluate_recognition_v4 import fit_calibration,fuse
from evaluate_restoration_v3 import extra_pairs,metrics
from investigate_identity import load
from refine_model import FrozenFace,sha,write_json
from run_project import ROOT,notebook_definitions,runtime

OUT=ROOT/'work/recognition_flip_v5'
V4=ROOT/'work/recognition_v4'


@torch.inference_mode()
def bank(n,model,images,faces):
    values={key:[] for key in faces}
    for offset in range(0,len(images),32):
        low=torch.from_numpy(np.array(images[offset:offset+32],copy=True)).permute(0,3,1,2).float().to(n.device)/255
        sr=model(low).clamp(0,1)
        for name,face in faces.items():
            feature=F.normalize(face(sr)+face(sr.flip(-1)),dim=1)
            assert torch.isfinite(feature).all() and torch.allclose(feature.norm(dim=1),torch.ones(len(sr),device=n.device),atol=1e-5)
            values[name].append(feature.cpu().numpy())
    return {key:np.concatenate(value) for key,value in values.items()}


def scores(bank,pairs):return {key:(values[pairs['ia']]*values[pairs['ib']]).sum(1) for key,values in bank.items()}


@torch.inference_mode()
def main():
    if OUT.exists():raise ValueError('Preserve recorded fixed attempt')
    OUT.mkdir(parents=True);(OUT/'emb').mkdir()
    selection=json.loads((V4/'selection.json').read_text());path=ROOT/selection['checkpoint']
    assert sha(path)==selection['checkpoint_sha256']
    protocol={'source_sha256':sha(__file__),'checkpoint_sha256':sha(path),'v4_selection_sha256':sha(V4/'selection.json'),
        'method':'Same v4 SR pixels; normalize(sum of original and horizontally mirrored recognition features), per recognizer. Fixed equal negative-validation-standardized ArcFace/MobileFace score fusion.',
        'calibration':'Unchanged original512 crops/256 held-out official-training IDs;256 positive256 negative pairs. No expanded calibration and no test-label threshold search.',
        'reason':'Fixed additional inference attempt after v4 primary fusion69.91%. Historical repeated-test research bias remains.',
        'image_output_changed':False,'recognizer_compute_passes':2,'recognizers_in_fusion':2,'sr_inference_passes':1,
        'test_used_to_choose_threshold_or_weights':False}
    write_json(OUT/'protocol.json',protocol)
    n,_=runtime();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load(n,path)
    faces={key:FrozenFace(ROOT/p).to(n.device).eval().requires_grad_(False) for key,p in {
        'arcface':'work/recognition/w600k_r50.onnx','mobileface':'work/refinement_assets/w600k_mbf.onnx'}.items()}
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    v4_protocol=json.loads((V4/'protocol.json').read_text());index={p:i for i,p in enumerate(v4_protocol['validation_files'])}
    validation_all=np.load(V4/'cache/validation28.npy',mmap_mode='r')
    validation=np.array(validation_all[[index[p] for p in domain['validation_files']]])
    ia=np.arange(0,512,2);ib=ia+1
    pairs={'ia':np.r_[ia,ia],'ib':np.r_[ib,np.roll(ib,-1)],'labels':np.r_[np.ones(256),np.zeros(256)]}
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'flip_recognition_protocol')
    validation_scores=scores(bank(n,model,validation,faces),pairs)
    calibration=fit_calibration(r,validation_scores,pairs['labels'])
    np.savez(OUT/'emb/validation.npz',labels=pairs['labels'],**{key+'_scores':value for key,value in validation_scores.items()})
    write_json(OUT/'calibration.json',calibration)
    print('FLIP CALIBRATION FROZEN',json.dumps(calibration),flush=True)
    # The test sources are read only after all inference settings/calibration freeze.
    original=np.load(ROOT/'work/qmul_study/cache/lr28.npy',mmap_mode='r')
    with np.load(ROOT/'work/qmul_study/cache/pairs.npz') as data:original_pairs={key:data[key] for key in ['ia','ib','labels']}
    additional=json.loads((ROOT/'work/leakage_review_v2/additional_cohort.json').read_text());more=[];hashes=[]
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        for filename in additional['files']:
            with Image.open(io.BytesIO(zipped.read(filename))) as image:
                rgb=image.convert('RGB');hashes.append(exact_key(np.asarray(rgb)));more.append(np.asarray(rgb.resize((28,28),Image.Resampling.BICUBIC)))
    more=np.stack(more);more_pairs=extra_pairs(additional['files'],hashes)
    result={'protocol':protocol,'calibration_sha256':sha(OUT/'calibration.json'),'recognition':{},
        'v4_original_evaluation_sha256':sha(V4/'evaluation.json')}
    for dataset,images,pair_data in [('prior_9913_images',original,original_pairs),('additional_identity_disjoint',more,more_pairs)]:
        values=scores(bank(n,model,images,faces),pair_data)
        values['fusion']=fuse(values,calibration['fusion_parameters'])
        result['recognition'][dataset]={key:metrics(value,pair_data['labels'],calibration['thresholds'][key]) for key,value in values.items()}
        np.savez(OUT/'emb'/f'{dataset}.npz',labels=pair_data['labels'],**{key+'_scores':value for key,value in values.items()})
        print('FLIP RESULT',dataset,json.dumps(result['recognition'][dataset]),flush=True)
    write_json(OUT/'evaluation.json',result);print('FLIP EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()

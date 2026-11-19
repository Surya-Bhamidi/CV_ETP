"""Finalize a checkpoint using validation only, before diagnostic evaluation.

Raw training selection remains archived. If neither full epoch passed, test
two smaller parameter updates under the same numerical validation limits.
"""
import json

import numpy as np
import torch

from input_quality import InputDegradation
from quality_metric import load_lpips
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime
from train_balanced_camera import assess
from train_qmul_domain import CameraSet
from train_restoration_v3 import BASE, OUT


def main():
    if (OUT/'training_selection.json').exists():raise ValueError('Final selection already recorded')
    selection=json.loads((OUT/'selection.json').read_text())
    (OUT/'training_selection.json').write_bytes((OUT/'selection.json').read_bytes())
    record={'source_sha256':sha(__file__),'coefficients_if_no_epoch_passed':[.5,.75],
            'base_checkpoint_sha256':sha(BASE),'texture_checkpoint_sha256':sha(OUT/'last.pt'),
            'selection':'Identical validation limits and score as training. No test data. Parameter interpolation only if no full epoch passed.',
            'evaluated':[]}
    write_json(OUT/'final_selection_manifest.json',record)
    if selection['selected_new_checkpoint']:
        state=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
        chosen_alpha=1.
    else:
        n,_=runtime();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        base=torch.load(BASE,map_location='cpu',weights_only=True)
        candidate=torch.load(OUT/'last.pt',map_location='cpu',weights_only=True)
        model=n.SRNet(n.cfg).to(n.device).eval().requires_grad_(False)
        face=FrozenFace(ROOT/'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
        lpips=load_lpips(n.device)
        hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
        with np.load(ROOT/'work/full/split_ids.npz') as split:ids=split['validation'][:128]
        sets={p:n.FixedEvalSet(hr,ids,InputDegradation(n,p),112,1000) for p in ['standard28','clean28']}
        data=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
        camera=CameraSet(ROOT/data['source_archive'],data['validation_files'])
        baseline=selection['baseline'];state=base;best_score=0.;chosen_alpha=0.
        for alpha in [.5,.75]:
            values={key:value.lerp(candidate['model'][key],alpha) if value.is_floating_point() else value for key,value in base['model'].items()}
            model.load_state_dict(values,strict=True)
            metrics=assess(n,model,face,sets,camera,lpips)
            improvement=float(np.mean([baseline['ffhq'][p]['lpips']-metrics['ffhq'][p]['lpips'] for p in sets]))
            camera_gain=metrics['domain_validation_arcface_auc']-baseline['domain_validation_arcface_auc']
            eligible=improvement>=.005 and camera_gain>=-.002 and all(
                metrics['ffhq'][p]['psnr_y']>=baseline['ffhq'][p]['psnr_y']-.15 and
                metrics['ffhq'][p]['ssim_y']>=baseline['ffhq'][p]['ssim_y']-.003 for p in sets)
            score=2*camera_gain+improvement
            record['evaluated'].append({'coefficient':alpha,'validation':metrics,'eligible':eligible,'score':score})
            if eligible and score>best_score:
                state={**candidate,'model':values,'parameter_update_fraction':alpha};best_score,chosen_alpha=score,alpha
            write_json(OUT/'final_selection_manifest.json',record)
            print('V3 PARTIAL UPDATE',json.dumps(record['evaluated'][-1]),flush=True)
        selection['selected_new_checkpoint']=chosen_alpha>0
        selection['selected_epoch']=2 if chosen_alpha>0 else 0
    state['final_selection']=record
    torch.save(state,OUT/'final.pt')
    selection.update(checkpoint='work/restoration_v3/final.pt',checkpoint_sha256=sha(OUT/'final.pt'),
                     selected_parameter_update_fraction=chosen_alpha,final_selection_manifest_sha256=sha(OUT/'final_selection_manifest.json'))
    write_json(OUT/'selection.json',selection)
    review=json.loads((OUT/'review.json').read_text())
    review.update(available_for_comparison=selection['selected_new_checkpoint'],checkpoint=selection['checkpoint'],checkpoint_sha256=selection['checkpoint_sha256'])
    review['checks']['validation_selected_new_epoch']=selection['selected_new_checkpoint']
    write_json(OUT/'review.json',review)
    print('V3 FINAL SELECTION',chosen_alpha,flush=True)


if __name__=='__main__':main()

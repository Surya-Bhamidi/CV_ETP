"""Separate, explicit clean-bicubic follow-up selected from validation results.

The original all-profile selection remains 'single'. This follow-up is not a
claim that back-projection works for native/noisy camera crops.
"""
import json

import numpy as np
import torch

from evaluate_restoration_passes import OUT, CHECKPOINT, append, means, png, score
from input_quality import InputDegradation
from investigate_identity import load
from quality_metric import load_lpips_alex
from refine_model import FrozenFace, sha, write_json
from restoration_passes import predict
from run_project import ROOT, runtime


@torch.inference_mode()
def main():
    folder = OUT / 'clean_only_followup'
    folder.mkdir(exist_ok=True)
    if (folder / 'evaluation.json').exists():
        raise ValueError('Preserve completed follow-up rather than silently retuning.')
    prior = json.loads((OUT / 'evaluation.json').read_text())
    assert json.loads((OUT / 'selection.json').read_text())['selected'] == 'single'
    single, correction = [prior['validation']['clean28'][m] for m in ['single','backprojection3']]
    passed = (correction['psnr_y'] > single['psnr_y'] and correction['ssim_y'] > single['ssim_y'] and
              correction['lpips_alex'] < single['lpips_alex'] and correction['mobileface_cosine'] > single['mobileface_cosine'])
    protocol = {'source_sha256': sha(__file__), 'inference_source_sha256': sha(ROOT/'restoration_passes.py'),
                'checkpoint_sha256': sha(CHECKPOINT), 'selected_on_validation': 'backprojection3' if passed else 'single',
                'post_hoc_profile_focus': True, 'profile': 'clean28', 'test_indices': list(range(512)),
                'validation_result_sha256': sha(OUT/'evaluation.json'),
                'scope': 'Only known clean bicubic-downsampled paired images; never inferred from the clean reference or selected per-image. No generic native/QMUL deployment.',
                'frozen_before_test': True, 'test_may_change_method': False,
                'default_ui_stays_single': True, 'new_training': False, 'neural_passes': 1, 'correction_iterations': 3}
    write_json(folder/'protocol.json',protocol)
    if not passed:
        return
    n,_ = runtime();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load(n,CHECKPOINT);lpips=load_lpips_alex(n.device)
    face=FrozenFace(ROOT/'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as ids:test=ids['test'][:512]
    data=n.FixedEvalSet(hr,test,InputDegradation(n,'clean28'),112,2000)
    bank={}
    for j,(low,target) in enumerate(data.batches(16)):
        low,target=low.to(n.device),target.to(n.device)
        for method in ['single','backprojection3']:
            value=predict(model,low,method)
            append(bank,method,score(n,value,target,lpips,face))
            if j==0:
                for i in range(6):png(value[i],folder/f'{i:02}_{method}.png')
        if j==0:
            for i in range(6):png(low[i],folder/f'{i:02}_input.png');png(target[i],folder/f'{i:02}_reference.png')
        if (j+1)%8==0:print('CLEAN FOLLOW-UP', (j+1)*16,'/512',flush=True)
    result={'protocol_sha256':sha(folder/'protocol.json'),'test_clean28':means(bank)}
    write_json(folder/'per_image.json',bank);write_json(folder/'evaluation.json',result)
    # Exposure is based on validation; the test is reported, not used to tune a method.
    review={'available_as_clean_only_experiment':True,'validation_selected':True,'test_used_to_select_method':False,
            'checkpoint_sha256':sha(CHECKPOINT),'inference_source_sha256':sha(ROOT/'restoration_passes.py'),
            'method':'backprojection3','allowed_profile':'clean28','native_camera_allowed':False,
            'default_ui_stays_single':True,'warning':'Experimental consistency correction for known clean bicubic inputs only. Not repeated neural SR, not an identity-recovery percentage.'}
    write_json(folder/'review.json',review)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

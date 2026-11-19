"""Paired-identity fine-tuning; no MobileFace or test scores in optimization."""
import collections
import dataclasses
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import load
from prepare_recognition_v4 import BASE,OUT
from quality_metric import load_lpips
from refine_model import FrozenFace,FrozenFeatures,sha,write_json
from run_project import ROOT,ProgressLoader,runtime
from train_balanced_camera import assess
from train_qmul_domain import CameraSet
from train_restoration_v3 import CameraLikeDegradation
from train_visual_sr import distance,laplacian,structure_loss


class PairedCamera(torch.utils.data.Dataset):
    def __init__(self,n,ids,camera):
        self.ff=n.FaceTrainSet(ROOT/'work/data/ffhq128_u8.npy',ids,CameraLikeDegradation(n),112,seed=42)
        groups=collections.defaultdict(list)
        for index,path in enumerate(camera.files):groups[int(path.rsplit('/',1)[-1].split('_')[0])].append(index)
        self.people=sorted(groups);self.groups=groups;self.camera=camera;self.epoch=0
        assert all(len(v)>=2 for v in groups.values())
    def __len__(self):return len(self.ff)
    def set_epoch(self,epoch):self.epoch=epoch;self.ff.set_epoch(epoch)
    def __getitem__(self,index):
        low,target=self.ff[index];person=self.people[index%len(self.people)]
        rng=np.random.default_rng(np.random.SeedSequence([42,self.epoch,index]))
        a,b=rng.choice(self.groups[person],2,replace=False)
        first,obs_a=self.camera.pair(a);second,obs_b=self.camera.pair(b)
        if rng.random()<.5:first,obs_a=first[:,::-1].copy(),obs_a[:,::-1].copy()
        if rng.random()<.5:second,obs_b=second[:,::-1].copy(),obs_b[:,::-1].copy()
        tensor=lambda value:torch.from_numpy(value).permute(2,0,1).float()/255
        return low,target,tensor(first),tensor(second),tensor(obs_a),tensor(obs_b),person


def supervised_contrastive(features,labels,temperature=.1):
    diagonal=torch.eye(len(features),device=features.device,dtype=torch.bool)
    logits=(features@features.T)/temperature
    logits=logits.masked_fill(diagonal,-1e9)
    positives=(labels[:,None]==labels[None,:])&~diagonal
    logprob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    return -((logprob*positives).sum(1)/positives.sum(1).clamp_min(1)).mean()


def main():
    if (OUT/'training_manifest.json').exists():raise ValueError('Preserve recorded training run')
    protocol=json.loads((OUT/'protocol.json').read_text());plan=protocol['training_plan'];weights=plan['weights']
    n,cfg=runtime();torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load(n,BASE).requires_grad_(True)
    face=FrozenFace(ROOT/'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    vgg=FrozenFeatures().to(n.device).eval().requires_grad_(False);lpips=load_lpips(n.device)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as split:train,val=split['train'][:16384],split['validation'][:128]
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    camera=CameraSet(ROOT/domain['source_archive'],domain['train_files'][:32768])
    camera_val=CameraSet(ROOT/domain['source_archive'],domain['validation_files'])
    sets={p:n.FixedEvalSet(hr,val,InputDegradation(n,p),112,1000) for p in ['standard28','clean28']}
    dataset=PairedCamera(n,train,camera)
    loader=n.DataLoader(dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42))
    manifest={'source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'base_checkpoint_sha256':sha(BASE),
        'parameters':n.n_params(model),'architecture':{'use_ssc':True,'use_attn':True},'epochs':2,'updates_per_epoch':len(loader),
        'training_plan':plan,'camera_training_identities':len(dataset.people),'contrastive_temperature':.1,
        'arcface_supervised_camera_pairs_per_batch':8,'mobileface_in_loss_or_selection':False,'test_used_for_selection':False}
    write_json(OUT/'training_manifest.json',manifest);(OUT/'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline=assess(n,model,face,sets,camera_val,lpips);write_json(OUT/'baseline_validation.json',baseline)
    history=[];best_epoch=0;best_score=baseline['domain_validation_arcface_auc']
    def save(path,epoch):
        torch.save({'model':model.state_dict(),'config':dataclasses.asdict(cfg),'epoch':42+epoch,
            'architecture':manifest['architecture'],'manifest':manifest,'history':history,'best_epoch':best_epoch},path)
    save(OUT/'best.pt',0);print('V4 BASELINE',json.dumps(baseline),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2*len(loader),eta_min=1e-6)
    for epoch in range(2):
        start=time.monotonic();model.train();sums={};dataset.set_epoch(160+epoch);loader.generator.manual_seed(202+epoch)
        for low,target,first,second,obs_a,obs_b,person in ProgressLoader(loader):
            low,target,first,second,obs_a,obs_b,person=[v.to(n.device) for v in [low,target,first,second,obs_a,obs_b,person]]
            opt.zero_grad(set_to_none=True);raw=model(torch.cat([low,first,second]));sr=raw.clamp(0,1)
            cam_sr=torch.cat([sr[16:24],sr[32:40]])
            with torch.no_grad():target_vgg=vgg(target[:8]);ff_face=face(target[:8]);cam_face=face(torch.cat([obs_a[:8],obs_b[:8]]))
            camera_features=face(cam_sr)
            parts={'pixel':F.l1_loss(raw[:16],target),'structure':structure_loss(sr[:16],target),
                'laplacian':distance(laplacian(sr[:16]),laplacian(target)),
                'vgg':distance(vgg(sr[:8]),target_vgg),'lpips':lpips(sr[:16],target,normalize=True).mean(),
                'ffhq_identity':1-(face(sr[:8])*ff_face).sum(1).mean(),
                'camera_lr_consistency':F.l1_loss(F.interpolate(sr[16:],size=(28,28),mode='bicubic',align_corners=False,antialias=True),torch.cat([first,second])),
                'camera_identity':1-(camera_features*cam_face).sum(1).mean(),
                'paired_identity_contrastive':supervised_contrastive(camera_features,person[:8].repeat(2))}
            loss=sum(weights[k]*value for k,value in parts.items())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite v4 loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step()
            for key,value in parts.items():sums[key]=sums.get(key,0)+value.detach().item()
        model.eval();value=assess(n,model,face,sets,camera_val,lpips)
        mean_lpips=np.mean([value['ffhq'][p]['lpips']-baseline['ffhq'][p]['lpips'] for p in sets])
        eligible=mean_lpips<=.005 and all(value['ffhq'][p]['psnr_y']>=baseline['ffhq'][p]['psnr_y']-.15 and
            value['ffhq'][p]['ssim_y']>=baseline['ffhq'][p]['ssim_y']-.003 for p in sets)
        history.append({'epoch':epoch+1,'seconds':time.monotonic()-start,'validation':value,'eligible':bool(eligible),'losses':{k:v/len(loader) for k,v in sums.items()}})
        if eligible and value['domain_validation_arcface_auc']>best_score:
            best_epoch,best_score=epoch+1,value['domain_validation_arcface_auc'];save(OUT/'best.pt',epoch+1)
        save(OUT/'last.pt',epoch+1);write_json(OUT/'history.json',history);print('V4 EPOCH',json.dumps(history[-1]),flush=True)
    selection={'selected_epoch':best_epoch,'selected_new_checkpoint':best_epoch>0,'checkpoint':'work/recognition_v4/best.pt',
        'checkpoint_sha256':sha(OUT/'best.pt'),'validation_only':True,'baseline':baseline,'history':history}
    write_json(OUT/'selection.json',selection)
    review={'available_for_comparison':best_epoch>0,'checkpoint':selection['checkpoint'],'checkpoint_sha256':selection['checkpoint_sha256'],
        'checks':{'validation_selected_identity_pair_epoch':best_epoch>0},'release_selected_using_test':False,'default_model_unchanged':True,
        'warning':'Experimental identity-pair training using official training identities. Recognition gains and restored identity are not guaranteed; compare with the preceding v3 output.'}
    write_json(OUT/'review.json',review);print('V4 TRAINING COMPLETE',best_epoch,flush=True)


if __name__=='__main__':main()

"""Validation-only, observation-consistent camera restoration refinement.

The camera constraint lives at observed LR resolution, not against an enlarged
blurred pseudo-HR image. Paired perceptual details come only from FFHQ training.
No new evaluation cohort or verification labels enter training or selection.
"""
import dataclasses
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import load
from quality_metric import load_lpips
from refine_model import FrozenFace, FrozenFeatures, sha, write_json
from run_project import ROOT, ProgressLoader, runtime
from train_balanced_camera import assess
from train_qmul_domain import CameraSet
from train_quality_sr import MixedDegradation
from train_visual_sr import distance, laplacian, structure_loss

OUT = ROOT / 'work/restoration_v3'
BASE = ROOT / 'work/qmul_quality/balanced_merge_v1/best.pt'


class CameraLikeDegradation:
    """Mixture of existing paired profiles and explicit variable camera sampling."""
    def __init__(self, n): self.mixed = MixedDegradation(n)
    def __call__(self, hr, rng):
        if rng.random() < .5: return self.mixed(hr, rng)
        image = hr
        if rng.random() < .7:
            sigma_x, sigma_y = rng.uniform(.25, 1.8, 2)
            image = cv2.GaussianBlur(image, (7,7), sigmaX=float(sigma_x), sigmaY=float(sigma_y))
        width = int(rng.integers(14,41)); height = int(np.clip(width*rng.uniform(.8,1.5), 12,48))
        image = np.asarray(Image.fromarray(image).resize((width,height), Image.Resampling.BICUBIC))
        image = np.clip(image.astype(float)+rng.normal(0,rng.uniform(0,5),image.shape),0,255).astype('uint8')
        if rng.random() < .8:
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(40,96))])
            if not ok: raise RuntimeError('JPEG failed')
            image = cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        return np.asarray(Image.fromarray(image).resize((28,28),Image.Resampling.BICUBIC))


class Dataset(torch.utils.data.Dataset):
    def __init__(self,n,ids,camera):
        self.ff = n.FaceTrainSet(ROOT/'work/data/ffhq128_u8.npy',ids,CameraLikeDegradation(n),112,seed=42)
        self.camera,self.epoch = camera,0
    def __len__(self): return len(self.ff)
    def set_epoch(self,epoch): self.epoch=epoch; self.ff.set_epoch(epoch)
    def __getitem__(self,index):
        low, target = self.ff[index]
        camera_low, observed = self.camera.pair(index)
        if np.random.default_rng(np.random.SeedSequence([42,self.epoch,index])).random() < .5:
            camera_low, observed = camera_low[:,::-1].copy(),observed[:,::-1].copy()
        tensor=lambda value: torch.from_numpy(value).permute(2,0,1).float()/255
        return low,target,tensor(camera_low),tensor(observed)


def main():
    if OUT.exists(): raise ValueError('Preserve recorded run; choose a fresh directory')
    audit=json.loads((ROOT/'work/leakage_review_v2/audit.json').read_text())
    if not audit['checks_no_detected_direct_leakage']: raise RuntimeError('Resolve blocking audit findings before training')
    OUT.mkdir(parents=True)
    n,cfg=runtime(); torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load(n,BASE).requires_grad_(True)
    face=FrozenFace(ROOT/'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False)
    vgg=FrozenFeatures().to(n.device).eval().requires_grad_(False)
    perceptual=load_lpips(n.device).eval().requires_grad_(False)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as split: train,val=split['train'][:32768],split['validation'][:128]
    data=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    camera=CameraSet(ROOT/data['source_archive'],data['train_files'][:32768])
    camera_val=CameraSet(ROOT/data['source_archive'],data['validation_files'])
    sets={p:n.FixedEvalSet(hr,val,InputDegradation(n,p),112,1000) for p in ['standard28','clean28']}
    dataset=Dataset(n,train,camera)
    loader=n.DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True,num_workers=0,pin_memory=True,
                        generator=torch.Generator().manual_seed(42))
    weights={'pixel':1.,'structure':.5,'laplacian':.1,'vgg':.02,'lpips':.04,
             'ffhq_identity':.02,'camera_lr_consistency':.1,'camera_identity':.02}
    manifest={'source_sha256':sha(__file__),'base_checkpoint_sha256':sha(BASE),'seed':42,'epochs':2,
        'updates_per_epoch':len(loader),'ffhq_train_images':len(train),'qmul_train_images':len(camera.files),
        'parameters':n.n_params(model),'architecture':{'use_ssc':True,'use_attn':True},'weights':weights,
        'audit_sha256':sha(ROOT/'work/leakage_review_v2/audit.json'),
        'additional_cohort_sha256':sha(ROOT/'work/leakage_review_v2/additional_cohort.json'),
        'training_data_manifest_sha256':sha(ROOT/'work/qmul_domain/data_manifest.json'),
        'selection':'Validation only: mean FFHQ LPIPS improvement >=0.005, each PSNR >= base-0.15 dB and SSIM >= base-0.003, camera training-validation ArcFace AUC >= base-0.002. Among eligible epochs maximize 2*(camera AUC-base AUC) + mean LPIPS improvement. Retain baseline if no eligible epoch.',
        'test_influences_training_epoch_or_release_selection':False,
        'scope':'Exploratory refinement; earlier repeated-test and unknown pretraining overlap limitations remain.',
        'camera_hr_supervision':False,'camera_constraint':'Downsample SR to28 and compare with observed input pixels; native enlarged observation only supplies frozen ArcFace distillation.'}
    write_json(OUT/'manifest.json',manifest);(OUT/'runner_snapshot.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    baseline=assess(n,model,face,sets,camera_val,perceptual);write_json(OUT/'baseline_validation.json',baseline)
    best_score,best_epoch,history=0.,0,[]
    def save(path,epoch):
        torch.save({'model':model.state_dict(),'config':dataclasses.asdict(cfg),'epoch':40+epoch,
                    'architecture':manifest['architecture'],'manifest':manifest,'history':history,'best_epoch':best_epoch},path)
    save(OUT/'best.pt',0)
    print('V3 BASELINE',json.dumps(baseline),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2*len(loader),eta_min=1e-6)
    for epoch in range(2):
        start=time.monotonic();model.train();sums={}
        dataset.set_epoch(120+epoch);loader.generator.manual_seed(162+epoch)
        for low,target,camera_low,observed in ProgressLoader(loader):
            low,target,camera_low,observed=[v.to(n.device) for v in [low,target,camera_low,observed]]
            opt.zero_grad(set_to_none=True);raw=model(torch.cat([low,camera_low]));sr=raw.clamp(0,1)
            with torch.no_grad(): target_vgg=vgg(target[:8]);ff_face=face(target[:8]);cam_face=face(observed[:8])
            parts={'pixel':F.l1_loss(raw[:32],target),'structure':structure_loss(sr[:32],target),
                'laplacian':distance(laplacian(sr[:32]),laplacian(target)),
                'vgg':distance(vgg(sr[:8]),target_vgg),'lpips':perceptual(sr[:16],target[:16],normalize=True).mean(),
                'ffhq_identity':1-(face(sr[:8])*ff_face).sum(1).mean(),
                'camera_lr_consistency':F.l1_loss(F.interpolate(sr[32:],size=(28,28),mode='bicubic',align_corners=False,antialias=True),camera_low),
                'camera_identity':1-(face(sr[32:40])*cam_face).sum(1).mean()}
            loss=sum(weights[k]*v for k,v in parts.items())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step()
            for k,v in parts.items():sums[k]=sums.get(k,0)+v.detach().item()
        model.eval();value=assess(n,model,face,sets,camera_val,perceptual)
        improvement=float(np.mean([baseline['ffhq'][p]['lpips']-value['ffhq'][p]['lpips'] for p in sets]))
        camera_gain=value['domain_validation_arcface_auc']-baseline['domain_validation_arcface_auc']
        eligible=improvement>=.005 and camera_gain>=-.002 and all(
            value['ffhq'][p]['psnr_y']>=baseline['ffhq'][p]['psnr_y']-.15 and
            value['ffhq'][p]['ssim_y']>=baseline['ffhq'][p]['ssim_y']-.003 for p in sets)
        score=2*camera_gain+improvement
        history.append({'epoch':epoch+1,'seconds':time.monotonic()-start,'validation':value,'eligible':eligible,
                        'selection_score':score,'losses':{k:v/len(loader) for k,v in sums.items()}})
        if eligible and score>best_score:best_epoch,best_score=epoch+1,score;save(OUT/'best.pt',epoch+1)
        save(OUT/'last.pt',epoch+1);write_json(OUT/'history.json',history)
        print('V3 EPOCH',json.dumps(history[-1]),flush=True)
    selection={'selected_epoch':best_epoch,'selected_new_checkpoint':best_epoch>0,'validation_only':True,
               'checkpoint_sha256':sha(OUT/'best.pt'),'baseline':baseline,'history':history}
    write_json(OUT/'selection.json',selection)
    review={'available_for_comparison':best_epoch>0,'checkpoint':'work/restoration_v3/best.pt',
            'checkpoint_sha256':sha(OUT/'best.pt'),'checks':{'validation_selected_new_epoch':best_epoch>0,
            'direct_leakage_audit_passed':audit['checks_no_detected_direct_leakage']},'experimental':True,
            'release_selected_using_test':False,'default_model_unchanged':True,
            'warning':'Validation-selected observation-consistent restoration. No clean HR QMUL targets; generated facial detail is not verified identity recovery. Test results are reported separately and do not select this option.'}
    write_json(OUT/'review.json',review);print('V3 TRAINING COMPLETE',best_epoch,flush=True)


if __name__=='__main__':main()

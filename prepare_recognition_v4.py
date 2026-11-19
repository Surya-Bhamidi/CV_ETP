"""Freeze the v4 experiment and an expanded, identity-disjoint validation pool."""
import collections
import io
import itertools
import json
import zipfile

import numpy as np
from PIL import Image

from audit_restoration_leakage import exact_key
from refine_model import sha, write_json
from run_project import ROOT

OUT=ROOT/'work/recognition_v4'
BASE=ROOT/'work/restoration_v3/final.pt'


def main():
    if OUT.exists():raise ValueError('Preserve existing experiment')
    OUT.mkdir(parents=True);(OUT/'cache').mkdir()
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    old=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    additional=json.loads((ROOT/'work/leakage_review_v2/additional_cohort.json').read_text())
    validation=set(domain['validation_identities'])
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        groups=collections.defaultdict(list)
        for path in zipped.namelist():
            if '__MACOSX' not in path and '/training_set/' in path and path.endswith('.jpg') and path.rsplit('/',1)[-1].split('_')[0] in validation:
                groups[path.rsplit('/',1)[-1].split('_')[0]].append(path)
        paths=[p for person in sorted(groups,key=int) for p in sorted(groups[person])]
        low=[];keys=[]
        for path in paths:
            with Image.open(io.BytesIO(zipped.read(path))) as image:
                rgb=image.convert('RGB');keys.append(exact_key(np.asarray(rgb)))
                low.append(np.asarray(rgb.resize((28,28),Image.Resampling.BICUBIC)))
        low=np.stack(low);np.save(OUT/'cache/validation28.npy',low)
        pool_hashes=set(keys)
        test_by_name={p.rsplit('/',1)[-1]:p for p in zipped.namelist() if '__MACOSX' not in p and '/Face_' in p and p.endswith('.jpg')}
        comparisons={'train':domain['train_files'][:32768],
                     'prior_verification':[test_by_name[p] for p in old['files']],
                     'additional_evaluation':additional['files']}
        duplicates=[]
        for name,items in comparisons.items():
            for path in items:
                with Image.open(io.BytesIO(zipped.read(path))) as image:key=exact_key(np.asarray(image.convert('RGB')))
                if key in pool_hashes:duplicates.append([name,path])
        if duplicates:raise RuntimeError('Expanded validation duplicates another split: '+str(duplicates[:5]))
    index={path:i for i,path in enumerate(paths)}
    old_indices=np.array([index[p] for p in domain['validation_files']])
    ia=np.arange(0,512,2);ib=ia+1
    primary={'ia':old_indices[np.r_[ia,ia]],'ib':old_indices[np.r_[ib,np.roll(ib,-1)]],
             'labels':np.r_[np.ones(256),np.zeros(256)]}
    rng=np.random.default_rng(20261120);pos=[];neg=[]
    people=sorted(groups,key=int)
    for person in people:
        ids=[index[p] for p in groups[person]]
        # Cap the number of pairs per identity, not the input image pool.
        pairs=list(itertools.combinations(ids,2));order=rng.permutation(len(pairs))
        selected=[]
        for offset in order:
            a,b=pairs[offset]
            if keys[a]!=keys[b]:selected.append((a,b))
            if len(selected)==16:break
        for a,b in selected:
            other=person
            while other==person:other=people[int(rng.integers(len(people)))]
            c=index[groups[other][int(rng.integers(len(groups[other])))]]
            pos.append((a,b));neg.append((a,c))
    all_pairs=np.array(pos+neg)
    np.savez(OUT/'cache/calibration_pairs.npz',primary_ia=primary['ia'],primary_ib=primary['ib'],primary_labels=primary['labels'],
             expanded_ia=all_pairs[:,0],expanded_ib=all_pairs[:,1],expanded_labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))])
    manifest={'base_checkpoint_sha256':sha(BASE),'source_sha256':sha(__file__),'validation_files':paths,
        'validation_images':len(paths),'validation_identities':len(groups),'expanded_pairs':len(all_pairs),
        'validation28_sha256':sha(OUT/'cache/validation28.npy'),'pairs_sha256':sha(OUT/'cache/calibration_pairs.npz'),
        'expanded_validation_exact_cross_split_duplicates':duplicates,
        'old512_primary_calibration_unchanged':True,'primary_goal':'MobileFace verification >70% on the unchanged 9913-image/10640-pair cohort with original512 validation-only calibration.',
        'recognition_fusion':'Separate diagnostic: fixed equal average of ArcFace and MobileFace pair scores standardized using negative validation pairs; threshold fit only on validation. Never claim fusion changes reconstructed pixels.',
        'secondary_calibration':'Expanded held-out-training-identity calibration is a separately labeled pipeline; report the original512 result alongside it.',
        'test_used_for_training_or_checkpoint_selection':False,'repeated_test_bias':'These evaluation cohorts have already been inspected in earlier work; not a new untouched benchmark.',
        'training_plan':{'epochs':2,'ffhq_images':16384,'audited_camera_pool':32768,'batch_ffhq':16,'batch_camera':32,'lr':1e-5,
          'weights':{'pixel':1.,'structure':.5,'laplacian':.1,'vgg':.02,'lpips':.04,'ffhq_identity':.02,'camera_lr_consistency':.2,'camera_identity':.02,'paired_identity_contrastive':.005},
          'selection':'Max original512 ArcFace validation AUC, retaining only epochs with mean paired FFHQ validation LPIPS<=base+0.005, each PSNR>=base-0.15dB and SSIM>=base-0.003. Baseline retained if no eligible AUC improvement.'}}
    write_json(OUT/'protocol.json',manifest)
    print('V4 PROTOCOL FROZEN',len(paths),'validation crops',len(all_pairs),'expanded pairs',flush=True)


if __name__=='__main__':main()

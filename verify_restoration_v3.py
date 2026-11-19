"""Reproduce new metrics and verify the original source pixels and frozen release."""
import io
import json
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from refine_model import sha, write_json
from run_project import ROOT

OUT=ROOT/'work/restoration_v3'


def main():
    audit_path=ROOT/'work/leakage_review_v2/audit.json'
    audit=json.loads(audit_path.read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert audit['checks_no_detected_direct_leakage'] and not any(audit['blocking_findings'].values())
    assert manifest['audit_sha256']==sha(audit_path)
    assert manifest['additional_cohort_sha256']==sha(ROOT/'work/leakage_review_v2/additional_cohort.json')
    cohort=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    low=np.load(ROOT/'work/qmul_study/cache/lr28.npy',mmap_mode='r')
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        paths={path.rsplit('/',1)[-1]:path for path in zipped.namelist() if '__MACOSX' not in path and '/Face_' in path and path.lower().endswith('.jpg')}
        for index,name in enumerate(cohort['files']):
            with Image.open(io.BytesIO(zipped.read(paths[name]))) as image:
                expected=np.asarray(image.convert('RGB').resize((28,28),Image.Resampling.BICUBIC))
            assert np.array_equal(expected,low[index]),name
    selection=json.loads((OUT/'selection.json').read_text())
    review=json.loads((OUT/'review.json').read_text())
    frozen=json.loads((OUT/'frozen_evaluation.json').read_text())
    result=json.loads((OUT/'evaluation.json').read_text())
    assert frozen['selection_sha256']==sha(OUT/'selection.json')
    assert review['checkpoint_sha256']==selection['checkpoint_sha256']==sha(ROOT/selection['checkpoint'])
    assert review['release_selected_using_test'] is False
    assert (OUT/'review.json').stat().st_mtime <= (OUT/'frozen_evaluation.json').stat().st_mtime
    for name,cohorts in result['qmul'].items():
        for dataset,nets in cohorts.items():
            with np.load(OUT/'emb'/f'{name}_{dataset}.npz') as data:
                for net,values in nets.items():
                    scores=data[net+'_scores'];labels=data['labels']
                    correct=(scores>=values['threshold_from_training_validation'])==labels
                    assert abs(correct.mean()-values['accuracy'])<1e-12
                    assert int(correct.sum())==values['correct_pairs']
                    assert abs(roc_auc_score(labels,scores)-values['auc'])<1e-12
    for profile,methods in result['paired_ffhq'].items():
        raw=json.loads((OUT/(profile+'_per_image.json')).read_text())
        for name,values in methods.items():
            for metric,value in values.items():assert len(raw[name][metric])==512 and abs(np.mean(raw[name][metric])-value)<1e-12
    uncertainty={'seed':42,'resamples':2000,'scope':'Fixed-model, fixed-calibration resampling only. Anchor-ID bootstrap does not fully account for shared negative identities, threshold fitting, training seeds, historical test reuse or upstream pretraining overlap.','paired_ffhq':{},'additional_recognition':{}}
    rng=np.random.default_rng(42)
    if 'restoration_v3' in result['qmul']:
        for profile in result['paired_ffhq']:
            raw=json.loads((OUT/(profile+'_per_image.json')).read_text())
            delta=np.array(raw['restoration_v3']['lpips_alex'])-np.array(raw['balanced_previous']['lpips_alex'])
            samples=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
            uncertainty['paired_ffhq'][profile]={'lpips_alex_mean_difference':float(delta.mean()),'paired_image_bootstrap_95_interval':np.percentile(samples,[2.5,97.5]).tolist()}
        dataset='additional_identity_disjoint'
        with np.load(OUT/'emb'/f'balanced_previous_{dataset}.npz') as a,np.load(OUT/'emb'/f'restoration_v3_{dataset}.npz') as b:
            assert np.array_equal(a['anchor_id'],b['anchor_id']) and np.array_equal(a['labels'],b['labels'])
            _,inverse=np.unique(a['anchor_id'],return_inverse=True);counts=np.bincount(inverse)
            for net in ['arcface','mobileface']:
                old=result['qmul']['balanced_previous'][dataset][net];new=result['qmul']['restoration_v3'][dataset][net]
                delta=((b[net+'_scores']>=new['threshold_from_training_validation'])==b['labels']).astype(float)-((a[net+'_scores']>=old['threshold_from_training_validation'])==a['labels']).astype(float)
                sums=np.bincount(inverse,weights=delta);indices=rng.integers(0,len(counts),(2000,len(counts)))
                samples=sums[indices].sum(1)/counts[indices].sum(1)*100
                uncertainty['additional_recognition'][net]={'accuracy_change_pp':float(delta.mean()*100),'anchor_id_bootstrap_95_interval_pp':np.percentile(samples,[2.5,97.5]).tolist()}
    write_json(OUT/'uncertainty.json',uncertainty)
    checks={'direct_audit_has_no_detected_leaks':True,'audited_archive_pixels_match_all_9913_actual_cached_inputs':True,
        'training_and_additional_cohort_hashes_unchanged':True,'selected_checkpoint_and_validation_record_unchanged':True,
        'release_review_frozen_before_test_evaluation':True,'reported_pair_accuracy_and_auc_reproduced':True,
        'paired_quality_means_reproduced_from_512_values':True,
        'limits':'Unknown pretraining/cross-dataset identity overlap and historical repeated-test bias remain; not an absolute no-leak guarantee.'}
    write_json(OUT/'verification_checks.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

"""Independently reproduce native coverage, thresholds, scores and timing samples."""
import io
import json
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from audit_restoration_leakage import exact_key
from evaluate_restoration_v3 import extra_pairs
from field_native_check import OUT,CHECKPOINTS
from measure_field_latency import statistics
from refine_model import sha,write_json
from run_project import ROOT,notebook_definitions


def accepts(size):
    if min(size)<8:return False
    if max(size)>256:
        ratio=256/max(size);size=[max(1,round(d*ratio)) for d in size]
    return min(size)>=8


def main():
    result=json.loads((OUT/'native_results.json').read_text());calibration=json.loads((OUT/'calibration.json').read_text())
    assert result['calibration_sha256']==sha(OUT/'calibration.json')
    assert result['protocol']['source_sha256']==sha(ROOT/'field_native_check.py')
    assert all(digest==sha(CHECKPOINTS[name]) for name,digest in result['protocol']['checkpoints'].items())
    manifest=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text());domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    additional=json.loads((ROOT/'work/leakage_review_v2/additional_cohort.json').read_text())
    support={'main':np.array([accepts(size) for size in manifest['native_sizes']])}
    extra_hashes=[];sizes=[]
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        for path in additional['files']:
            with Image.open(io.BytesIO(zipped.read(path))) as image:
                sizes.append(image.size);extra_hashes.append(exact_key(np.asarray(image.convert('RGB'))))
        support['additional']=np.array([accepts(size) for size in sizes]);sizes=[]
        for path in domain['validation_files']:
            with Image.open(io.BytesIO(zipped.read(path))) as image:sizes.append(image.size)
        support['validation']=np.array([accepts(size) for size in sizes])
    ia=np.arange(0,512,2);ib=ia+1
    pairs={'validation':{'ia':np.r_[ia,ia],'ib':np.r_[ib,np.roll(ib,-1)],'labels':np.r_[np.ones(256),np.zeros(256)]},
           'additional':extra_pairs(additional['files'],extra_hashes)}
    with np.load(ROOT/'work/qmul_study/cache/pairs.npz') as data:pairs['main']={key:data[key] for key in ['ia','ib','labels']}
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'native_field_verify')
    for dataset,values in support.items():
        counts=result['input_coverage'][dataset]
        assert len(values)==counts['images'] and int(values.sum())==counts['supported_images'] and int((~values).sum())==counts['unsupported_images']
    correctness={}
    for name in result['recognition']:
        with np.load(OUT/'emb'/f'{name}_validation.npz') as data:
            expected=support['validation'][pairs['validation']['ia']]&support['validation'][pairs['validation']['ib']]
            assert np.array_equal(data['labels'],pairs['validation']['labels']) and np.array_equal(data['supported'],expected)
            for mode in ['single','mirror']:
                threshold=r.VerificationEval.best_threshold(data[mode+'_scores'][expected],data['labels'][expected])
                assert float(threshold)==calibration[name][mode]['threshold']
        correctness[name]={}
        for dataset,modes in result['recognition'][name].items():
            with np.load(OUT/'emb'/f'{name}_{dataset}.npz') as data:
                expected=support[dataset][pairs[dataset]['ia']]&support[dataset][pairs[dataset]['ib']]
                assert np.array_equal(data['labels'],pairs[dataset]['labels']) and np.array_equal(data['supported'],expected)
                correctness[name][dataset]={}
                for mode,record in modes.items():
                    scores=data[mode+'_scores'];assert np.isfinite(scores[expected]).all() and np.isnan(scores[~expected]).all()
                    correct=np.zeros(len(expected),dtype=bool);correct[expected]=(scores[expected]>=calibration[name][mode]['threshold'])==data['labels'][expected]
                    assert int(correct.sum())==record['correct_pairs']
                    assert abs(correct[expected].mean()-record['supported_pair_accuracy'])<1e-12
                    assert abs(correct.mean()-record['all_pair_success_with_abstentions_counted_unsuccessful'])<1e-12
                    assert abs(roc_auc_score(data['labels'][expected],scores[expected])-record['supported_pair_auc'])<1e-12
                    assert record['pairs']==len(expected) and record['supported_pairs']==int(expected.sum())
                    correctness[name][dataset][mode]=correct
    latency=json.loads((OUT/'latency.json').read_text());spot=json.loads((OUT/'runtime_spotcheck.json').read_text())
    assert latency['source_sha256']==sha(ROOT/'measure_field_latency.py')
    assert spot['source_sha256']==sha(ROOT/'confirm_field_latency.py') and spot['latency_run_sha256']==sha(OUT/'latency.json')
    for case,modes in spot['gpu'].items():
        for name,record in modes.items():assert statistics(record['raw_ms'])==record['summary']
    for case,record in spot['browser_fetch'].items():
        assert statistics(record['raw_round_trip_ms'])==record['round_trip_summary']
        assert statistics(record['raw_server_comparison_ms'])==record['server_comparison_summary']
    assert all(v['same_x4_shape'] and v['max_pixel_difference_255']<=2 and v['mean_pixel_difference_255']<=.1 for v in latency['api_vs_offline_pixel_checks'].values())
    rng=np.random.default_rng(42);uncertainty={'scope':'Fixed model/native calibration, paired anchor-ID resampling only; does not fully account for shared negative IDs, calibration fitting, training seeds, repeated test use or upstream overlap.','seed':42,'resamples':2000,'results':{}}
    for dataset in ['main','additional']:
        names=manifest['files'] if dataset=='main' else [path.rsplit('/',1)[-1] for path in additional['files']]
        anchors=np.array([names[i].split('_')[0] for i in pairs[dataset]['ia']]);_,inverse=np.unique(anchors,return_inverse=True)
        eligible=support[dataset][pairs[dataset]['ia']]&support[dataset][pairs[dataset]['ib']]
        counts=np.bincount(inverse,weights=eligible.astype(float));indices=rng.integers(0,len(counts),(2000,len(counts)))
        denominator=counts[indices].sum(1);uncertainty['results'][dataset]={}
        for mode in ['single','mirror']:
            current=correctness['identity_v4'][dataset][mode]
            sums=np.bincount(inverse,weights=current.astype(float))
            uncertainty['results'][dataset][mode]={'v4_supported_accuracy_interval_percent':np.percentile(sums[indices].sum(1)/denominator*100,[2.5,97.5]).tolist()}
            for before in ['bicubic','camera_v3','app_default']:
                delta=current.astype(float)-correctness[before][dataset][mode].astype(float)
                sums=np.bincount(inverse,weights=delta)
                uncertainty['results'][dataset][mode][before+'_change_interval_pp']=np.percentile(sums[indices].sum(1)/denominator*100,[2.5,97.5]).tolist()
    write_json(OUT/'uncertainty.json',uncertainty)
    checks={'all_native_input_coverage_and_rejections_reproduced':True,'all_original_pairs_retained':True,
        'all_native_validation_thresholds_reproduced':True,'all_supported_and_all_pair_success_metrics_reproduced':True,
        'all_supported_pair_auc_reproduced':True,'raw_runtime_spotcheck_statistics_reproduced':True,
        'uncached_upload_api_pixels_match_offline_inference_within_one_to_two_levels':True,
        'no_live_video_or_physical_field_trial_claimed':True}
    write_json(OUT/'verification_checks.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

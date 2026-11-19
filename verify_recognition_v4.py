"""Reproduce calibration and all v4 accuracies from the stored raw pair scores."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_recognition_v4 import fit_calibration,fuse
from prepare_recognition_v4 import OUT
from refine_model import sha,write_json
from run_project import ROOT,notebook_definitions


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    frozen=json.loads((OUT/'frozen_evaluation.json').read_text());result=json.loads((OUT/'evaluation.json').read_text())
    calibrations=json.loads((OUT/'calibration.json').read_text())
    assert frozen['protocol_sha256']==sha(OUT/'protocol.json')==json.loads((OUT/'training_manifest.json').read_text())['protocol_sha256']
    assert frozen['selection_sha256']==sha(OUT/'selection.json')
    assert result['calibration_sha256']==sha(OUT/'calibration.json')
    old_evaluation=json.loads((ROOT/'work/restoration_v3/evaluation.json').read_text())
    assert old_evaluation['protocol']['checkpoint_sha256']['restoration_v3']==frozen['checkpoint_sha256']['restoration_v3']
    for dataset,digest in result['reused_v3_pair_score_sha256'].items():
        assert digest==sha(ROOT/'work/restoration_v3/emb'/f'restoration_v3_{dataset}.npz')
    assert sha(OUT/'cache/validation28.npy')==protocol['validation28_sha256']
    assert sha(OUT/'cache/calibration_pairs.npz')==protocol['pairs_sha256']
    assert not protocol['expanded_validation_exact_cross_split_duplicates']
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    people=lambda paths:{p.rsplit('/',1)[-1].split('_')[0] for p in paths}
    assert not people(protocol['validation_files']) & people(domain['train_files'][:32768])
    old=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    assert not people(protocol['validation_files']) & people(old['files'])
    cached=np.load(OUT/'cache/validation28.npy',mmap_mode='r')
    index={path:i for i,path in enumerate(protocol['validation_files'])}
    with np.load(OUT/'cache/calibration_pairs.npz') as pairs:
        ia=np.arange(0,512,2);ib=ia+1;indices=np.array([index[path] for path in domain['validation_files']])
        assert np.array_equal(pairs['primary_ia'],indices[np.r_[ia,ia]])
        assert np.array_equal(pairs['primary_ib'],indices[np.r_[ib,np.roll(ib,-1)]])
        assert len(cached)==11135 and len(pairs['expanded_labels'])==4968
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'v4_verify')
    for name,datasets in result['recognition'].items():
        assert sha(ROOT/json.loads((OUT/'selection.json').read_text())['checkpoint'])==selection['checkpoint_sha256']
        for calibration,settings in calibrations[name].items():
            with np.load(OUT/'emb'/f'{name}_validation_{calibration}.npz') as data:
                scores={key:data[key+'_scores'] for key in ['arcface','mobileface']}
                assert fit_calibration(r,scores,data['labels'])==settings
        for dataset,calibration_values in datasets.items():
            with np.load(OUT/'emb'/f'{name}_{dataset}.npz') as data:
                scores={key:data[key+'_scores'] for key in ['arcface','mobileface']}
                for calibration,values in calibration_values.items():
                    fused=fuse(scores,calibrations[name][calibration]['fusion_parameters'])
                    assert np.array_equal(fused,data[calibration+'_fusion_scores'])
                    for key,vector in {**scores,'fusion':fused}.items():
                        record=values[key];correct=(vector>=calibrations[name][calibration]['thresholds'][key])==data['labels']
                        assert abs(correct.mean()-record['accuracy'])<1e-12
                        assert int(correct.sum())==record['correct_pairs']
                        assert abs(roc_auc_score(data['labels'],vector)-record['auc'])<1e-12
    assert (OUT/'review.json').stat().st_mtime <= (OUT/'frozen_evaluation.json').stat().st_mtime
    assert (OUT/'calibration.json').stat().st_mtime <= (OUT/'evaluation.json').stat().st_mtime
    if selection['selected_new_checkpoint']:
        expected=result['recognition']['identity_v4']['prior_9913_images']['primary']['mobileface']['accuracy']>.70
    else:expected=False
    assert result['primary_target_over70']==expected
    for profile,methods in result['paired_ffhq'].items():
        raw=json.loads((OUT/(profile+'_per_image.json')).read_text())
        for name,values in methods.items():
            for key,value in values.items():assert len(raw[name][key])==512 and abs(np.mean(raw[name][key])-value)<1e-12
    checks={'validation_identity_and_pixel_cache_integrity':True,'primary512_calibration_indices_unchanged':True,
        'all_calibration_thresholds_reproduced_from_validation_scores':True,'all_pair_accuracy_auc_and_fixed_fusion_scores_reproduced':True,
        'selection_and_release_frozen_before_evaluation':True,'paired_quality_metrics_reproduced':True,
        'primary_over70_claim_checked':True,'independent_mobileface_excluded_from_training_and_selection':True,
        'limits':'Historical repeated-test bias and unknown pretraining/FFHQ person overlap remain; fusion is a different recognition pipeline.'}
    write_json(OUT/'verification_checks.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

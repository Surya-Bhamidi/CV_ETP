"""Check complete-study cohort, pair decisions, identity separation and checkpoints."""
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark_qmul import OUT
from refine_model import sha, write_json
from run_project import ROOT
from write_qmul_study import DOMAIN_METHODS, INTERVENTIONS, MAIN


def main():
    names = list(dict.fromkeys(['native_bicubic'] + MAIN + INTERVENTIONS + DOMAIN_METHODS))
    manifest = json.loads((OUT / 'cache/manifest.json').read_text())
    with np.load(OUT / 'cache/pairs.npz') as cache:
        labels, folds = cache['labels'], cache['folds']
    assert len(labels) == 10640 and labels.sum() == 5320 and set(folds) == set(range(10))
    source_hash = sha(OUT / 'cache/lr28.npy')
    evaluator_hashes = set(); row_checks = {}
    for name in names:
        result = json.loads((OUT / 'results' / (name + '.json')).read_text())
        assert result['images'] == 9913 and result['pairs'] == 10640
        assert result['protocol_sha256'] == manifest['protocol_sha256']
        assert result['metadata']['lr28_sha256'] == source_hash
        evaluator_hashes.add(json.dumps(result['metadata']['recognizer_sha256'], sort_keys=True))
        with np.load(OUT / 'emb' / (name + '_pairs.npz')) as data:
            assert np.array_equal(data['labels'], labels) and np.array_equal(data['folds'], folds)
            for net in ['arcface', 'mobileface']:
                scores = data[net + '_scores']; correct = data[net + '_correct']
                metrics = result['evaluators'][net]
                assert np.isfinite(scores).all() and len(scores) == len(labels)
                thresholds = np.array(metrics['thresholds'])
                decisions = scores >= thresholds[folds]
                assert np.array_equal(correct, decisions == labels)
                assert abs(correct.mean() - metrics['acc']) < 1e-12
                assert int(correct.sum()) == metrics['correct_pairs']
                assert abs(roc_auc_score(labels, scores) - metrics['auc']) < 1e-12
        with np.load(OUT / 'examples' / (name + '.npz')) as images:
            assert np.array_equal(images['indices'], np.linspace(0, 9912, 6, dtype=int))
            assert images['images'].shape == (6, 112, 112, 3)
        row_checks[name] = True
    assert len(evaluator_hashes) == 1
    domain = json.loads((ROOT / 'work/qmul_domain/data_manifest.json').read_text())
    train = {p.rsplit('/', 1)[-1].split('_')[0] for p in domain['train_files'][:16384]}
    val = {p.rsplit('/', 1)[-1].split('_')[0] for p in domain['validation_files']}
    test = {p.split('_')[0] for p in manifest['files']}
    assert not train & val and not (train | val) & test
    for name in DOMAIN_METHODS:
        folder = ROOT / 'work/qmul_domain' / name
        state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        selection = json.loads((folder / 'validation_result.json').read_text())
        assert len(selection['history']) == 2 and state['manifest']['updates_per_epoch'] == 512
        assert state['manifest']['test_used_for_training_or_selection'] is False
        assert state['manifest']['data_manifest_sha256'] == sha(ROOT / 'work/qmul_domain/data_manifest.json')
        if name == 'domain_no_attention': assert not any('.attn.' in key for key in state['model'])
        if name == 'domain_no_ssc': assert not any('.spatial.offset.' in key for key in state['model'])
    checks = {'all_method_rows_verified': row_checks, 'one_full_cohort_and_input_pixel_hash': True,
              'same_frozen_recognizers_for_every_method': True, 'raw_pair_decisions_reproduce_accuracy_and_auc': True,
              'all_example_indices_fixed_not_quality_selected': True, 'official_training_validation_test_identity_separation': True,
              'matched_training_budget_and_component_removal_verified': True,
              'camera_observation_targets_are_not_claimed_as_clean_hr': domain['observed112_is_clean_hr'] is False}
    write_json(OUT / 'validation_checks.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__': main()

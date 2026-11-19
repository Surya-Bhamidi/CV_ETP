"""Cache a balanced subset of official QMUL training identities, without extracting paths.

Targets are resized native LR observations, never clean HR ground truth.
Validation identities are excluded from training; all official test IDs excluded.
"""
import collections
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8'); temporary.replace(path)

OUT = ROOT / 'work/qmul_domain'


def main():
    cache = OUT / 'cache'; cache.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'work/recognition/qmul_dl/QMUL-SurvFace-v1.zip'
    if (OUT / 'data_manifest.json').exists(): raise ValueError('Domain cache already prepared; preserve its manifest')
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if '__MACOSX' not in name and name.lower().endswith(('.jpg', '.png', '.jpeg'))]
        groups = collections.defaultdict(list)
        for name in names:
            if '/training_set/' in name: groups[str(int(name.rsplit('/', 1)[-1].split('_')[0]))].append(name)
        test_prefixes = [name.rsplit('/', 1)[-1].split('_')[0] for name in names if '/Face_' in name]
        test_ids = {str(int(prefix)) for prefix in test_prefixes if prefix.isdigit()}
        assert not set(groups) & test_ids
        rng = np.random.default_rng(42)
        ids = np.array(sorted(groups, key=int)); ids = ids[rng.permutation(len(ids))]
        val_ids = ids[:256].tolist(); train_ids = ids[256:].tolist()
        for identity in groups:
            group = sorted(groups[identity]); order = rng.permutation(len(group))
            groups[identity] = [group[i] for i in order]
        val_files = [name for identity in val_ids for name in groups[identity][:2]]
        assert len(val_files) == 512
        train_files = []
        for position in range(max(map(len, groups.values()))):
            for identity in train_ids:
                if position < len(groups[identity]): train_files.append(groups[identity][position])
                if len(train_files) == 32768: break
            if len(train_files) == 32768: break
    manifest = {'source_archive_sha256': sha(archive), 'official_training_images_in_archive': sum(map(len, groups.values())),
        'official_training_identities': len(groups), 'training_identities': train_ids, 'validation_identities': val_ids,
        'train_images': len(train_files), 'validation_images': len(val_files), 'train_files': train_files, 'validation_files': val_files,
        'train_validation_identity_intersection': [], 'official_train_test_identity_intersection': [],
        'observed112_is_clean_hr': False, 'selection': 'Seed 42; 256 held-out training identities; balanced round-robin training subset',
        'cache_policy': 'Decode original training JPEGs from the source zip on demand; no large image cache',
        'source_archive': archive.relative_to(ROOT).as_posix()}
    write_json(OUT / 'data_manifest.json', manifest)
    print('DOMAIN DATA PREPARED', manifest['train_images'], manifest['validation_images'], flush=True)


if __name__ == '__main__': main()

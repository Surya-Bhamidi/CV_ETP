"""Audit actual split contents and freeze an additional diagnostic QMUL cohort.

Exact/near duplicate checks cannot prove unknown cross-dataset identities or
external pretraining separation. Historical repeated-test bias is disclosed.
"""
import collections
import hashlib
import io
import json
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from refine_model import sha, write_json
from run_project import ROOT

OUT = ROOT / 'work/leakage_review_v2'


def signature(image):
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype('float32')
    values = cv2.dct(gray)[:8, :8].flatten()[1:]
    bits = values > np.median(values)
    return int.from_bytes(np.packbits(bits).tobytes(), 'big'), small


def exact_key(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with np.load(ROOT / 'work/full/split_ids.npz') as data: split = {k: data[k] for k in data.files}
    assert set(split) == {'train', 'validation', 'test'}
    sets = {k: set(map(int, v)) for k, v in split.items()}
    assert sum(map(len, sets.values())) == len(set.union(*sets.values())) == 70000
    ff = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    exact, center_exact, phash, thumbnails = {}, {}, {}, {}
    for name, ids in split.items():
        for index in ids:
            image = np.asarray(ff[index]); key = exact_key(image)
            exact.setdefault(key, []).append((name, int(index)))
            center_exact.setdefault(exact_key(image[8:120,8:120]), []).append((name, int(index)))
            code, small = signature(image)
            phash[int(index)], thumbnails[int(index)] = code, small
    exact_cross = [group for group in exact.values() if len({row[0] for row in group}) > 1]
    center_cross = [group for group in center_exact.values() if len({row[0] for row in group}) > 1]
    # Five disjoint hash bands guarantee that Hamming<=4 candidates share a band.
    bands = [collections.defaultdict(list) for _ in range(5)]
    shifts, widths = [0, 13, 26, 39, 52], [13, 13, 13, 13, 12]
    for index in split['train']:
        code = phash[int(index)]
        for band, shift, width in zip(bands, shifts, widths): band[(code >> shift) & ((1 << width)-1)].append(int(index))
    near, candidates = [], []
    for name in ['validation', 'test']:
        for index in split[name]:
            index = int(index); code = phash[index]; matches = set()
            for band, shift, width in zip(bands, shifts, widths): matches.update(band.get((code >> shift) & ((1 << width)-1), []))
            for train in matches:
                distance = (code ^ phash[train]).bit_count()
                if distance > 4: continue
                mae = float(np.abs(thumbnails[index].astype(float)-thumbnails[train]).mean())
                record = {'train': train, name: index, 'phash_hamming': distance, 'rgb32_mae_255': mae}
                candidates.append(record)
                if mae <= 3.: near.append(record)
    print('FFHQ AUDIT', len(exact_cross), 'exact split duplicates;', len(near), 'strict near duplicates', flush=True)
    domain = json.loads((ROOT / 'work/qmul_domain/data_manifest.json').read_text())
    cohort = json.loads((ROOT / 'work/qmul_study/cache/manifest.json').read_text())
    identity = lambda path: str(int(path.rsplit('/', 1)[-1].split('_')[0]))
    train_files = domain['train_files'][:32768]
    camera_sets = {'train': train_files, 'validation': domain['validation_files']}
    previous_ids = {identity(path) for path in cohort['files']}
    with zipfile.ZipFile(ROOT / domain['source_archive']) as zipped:
        groups = collections.defaultdict(dict)
        for path in zipped.namelist():
            if '__MACOSX' not in path and '/Face_' in path and path.lower().endswith('.jpg') and path.rsplit('/',1)[-1].split('_')[0].isdigit():
                groups[identity(path)][path.rsplit('/', 1)[-1]] = path
        new_ids = sorted(set(groups) - previous_ids, key=int)
        rng = np.random.default_rng(20261119)
        fresh_files = []
        for person in new_ids:
            paths = sorted(groups[person].values())
            chosen = rng.choice(len(paths), min(4, len(paths)), replace=False)
            fresh_files.extend(paths[i] for i in sorted(chosen))
        fresh = {'files': fresh_files, 'identities': new_ids, 'seed': 20261119,
                 'selection': 'All available official-test identities absent from the prior 9913-image pair cohort; up to four random crops per identity, no restoration/recognition quality selection',
                 'scope': 'Additional diagnostic cohort, not the official verification protocol and not guaranteed historically unseen: the original pixel model previously ran a larger identification evaluation.',
                 'training_or_selection_allowed': False, 'prior_recent_cohort_identity_overlap': 0,
                 'source_archive_sha256': domain['source_archive_sha256']}
        frozen = OUT / 'additional_cohort.json'
        if frozen.exists(): assert json.loads(frozen.read_text()) == fresh
        else: write_json(frozen, fresh)
        camera_sets['previous_verification'] = [groups[identity(path)][path] for path in cohort['files']]
        camera_sets['additional_evaluation'] = fresh_files
        hashes = collections.defaultdict(list); resized_hashes = collections.defaultdict(list)
        for name, paths in camera_sets.items():
            for i, path in enumerate(paths):
                with Image.open(io.BytesIO(zipped.read(path))) as image:
                    rgb = np.asarray(image.convert('RGB')); low = np.asarray(image.convert('RGB').resize((28, 28), Image.Resampling.BICUBIC))
                hashes[exact_key(rgb)].append((name, path))
                resized_hashes[exact_key(low)].append((name, path))
            print('QMUL HASHED', name, len(paths), flush=True)
    ids = {name: {identity(path) for path in paths} for name, paths in camera_sets.items()}
    intersections = {a + '_vs_' + b: sorted(ids[a] & ids[b], key=int) for i, a in enumerate(ids) for b in list(ids)[i+1:]}
    camera_duplicates = [group for group in hashes.values() if len({item[0] for item in group}) > 1]
    resized_duplicates = [group for group in resized_hashes.values() if len({item[0] for item in group}) > 1]
    # Include all FFHQ vs camera exact low28 hashes to catch literal cross-dataset reuse.
    ff_low_hashes = set()
    for image in ff:
        ff_low_hashes.add(exact_key(np.asarray(Image.fromarray(np.asarray(image)[8:120, 8:120]).resize((28,28), Image.Resampling.BICUBIC))))
    cross_data = [group for key, group in resized_hashes.items() if key in ff_low_hashes]
    blocking = {'ffhq_split_id_overlap': False, 'ffhq_exact_cross_split_duplicates': bool(exact_cross),
                'ffhq_center_crop_cross_split_duplicates': bool(center_cross),
                'ffhq_strict_near_cross_split_duplicates': bool(near),
                'qmul_cross_split_identity_overlap': any(intersections.values()),
                'qmul_cross_split_exact_decoded_duplicates': bool(camera_duplicates),
                'qmul_cross_split_exact_lr28_duplicates': bool(resized_duplicates),
                'ffhq_qmul_exact_center_crop_lr28_overlap': bool(cross_data)}
    result = {'checks_no_detected_direct_leakage': not any(blocking.values()), 'blocking_findings': blocking,
        'ffhq': {'counts': {k: len(v) for k,v in split.items()}, 'exact_cross_split_groups': exact_cross,
                 'center_crop_exact_cross_split_groups': center_cross,
                 'strict_near_duplicates': near, 'phash_candidates_hamming_le4': candidates,
                 'near_duplicate_rule': 'pHash Hamming<=4 AND mean RGB difference of area-resized32<=3/255; heuristic, not identity verification'},
        'qmul': {'counts': {k: len(v) for k,v in camera_sets.items()}, 'identity_intersections': intersections,
                 'decoded_exact_cross_split_groups': camera_duplicates, 'lr28_exact_cross_split_groups': resized_duplicates},
        'cross_dataset_exact_lr28_matches': cross_data, 'fresh_cohort_sha256': sha(frozen),
        'source_sha256': sha(__file__), 'seconds': time.monotonic()-start,
        'limits': ['FFHQ has no identity labels in the cache; image-level disjointness does not establish identity-level disjointness.',
                  'Unknown shared people across FFHQ, QMUL and recognizer/generative-model pretraining cannot be certified absent.',
                  'Previous repeated-test comparisons and test-informed UI release gates create selection bias even without training pixel overlap.',
                  'Prior 10-fold pair calibration reuses identities/images across folds; this is not identity-disjoint threshold calibration.',
                  'New comparison will freeze settings before evaluating and calibrate thresholds only using official-training validation identities.']}
    write_json(OUT / 'audit.json', result)
    print('AUDIT COMPLETE', json.dumps(blocking), flush=True)


if __name__ == '__main__': main()

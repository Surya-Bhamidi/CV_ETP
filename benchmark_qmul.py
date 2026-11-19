"""Same-cohort QMUL model comparison, cached inputs, fixed folds and two evaluators.

No clean HR reference exists; outputs are evaluated for pair verification only.
Each named method processes all 9,913 verification images, not a selected subset.
"""
import argparse
import copy
import gc
import hashlib
import json
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, notebook_definitions, runtime
from sparse_attention import SelfEnsemble

OUT = ROOT / 'work/qmul_study'
MIXED = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'
METHODS = ['native_bicubic', 'bicubic', 'conv_baseline', 'ours_original', 'ours_reconstruction',
           'ours_mixed', 'ours_flip2', 'ours_no_attention', 'ours_fixed_sampling', 'ours_dense_keys',
           'ours_fidelity50', 'realesrgan', 'gfpgan', 'codeformer']
DOMAIN_METHODS = ['domain_full', 'domain_ffhq_control', 'domain_no_attention', 'domain_no_ssc']


def prepare():
    folder = OUT / 'cache'; folder.mkdir(parents=True, exist_ok=True)
    r = notebook_definitions('sr_part2a_recognition.ipynb', [1, 6, 7, 8, 9, 11], 'qmul_comparison_protocol')
    qmul = r.QMULData(str(ROOT / 'work/recognition'))
    if qmul.missing: raise RuntimeError('QMUL coverage is incomplete')
    names = sorted({name for pair in qmul.pairs for name in pair})
    index = {name: i for i, name in enumerate(names)}
    ia = np.array([index[a] for a, b in qmul.pairs]); ib = np.array([index[b] for a, b in qmul.pairs])
    labels = qmul.pair_labels
    rng = np.random.default_rng(0); folds = np.zeros(len(labels), int)
    for label in (0, 1):
        ids = np.where(labels == label)[0]; folds[rng.permutation(ids)] = np.arange(len(ids)) % 10
    protocol = {'files': names, 'pairs': [list(pair) for pair in qmul.pairs], 'labels': labels.tolist(), 'folds': folds.tolist()}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (folder / 'manifest.json').exists():
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['protocol_sha256'] != digest: raise ValueError('Cohort changed; do not reuse this cache')
        if sha(folder / 'lr28.npy') != manifest['lr28_sha256']:
            raise ValueError('Input cache checksum changed')
        if 'source_paths' not in manifest:
            manifest['source_paths'] = [qmul.paths[name].relative_to(ROOT).as_posix() for name in names]
            manifest['native_cache_note'] = 'Native112 cache removed to save disk; reconstruct directly from these same source paths with PIL bicubic'
            write_json(folder / 'manifest.json', manifest)
        return r, manifest
    low = np.lib.format.open_memmap(folder / 'lr28.npy', mode='w+', dtype='uint8', shape=(len(names), 28, 28, 3))
    native_digest = hashlib.sha256()
    sizes = []
    for i, name in enumerate(names):
        with Image.open(qmul.paths[name]) as image:
            image = image.convert('RGB'); sizes.append(list(image.size))
            low[i] = np.asarray(image.resize((28, 28), Image.Resampling.BICUBIC))
            native_digest.update(np.asarray(image.resize((112, 112), Image.Resampling.BICUBIC)).tobytes())
    low.flush(); del low
    np.savez(folder / 'pairs.npz', ia=ia, ib=ib, labels=labels, folds=folds)
    manifest = {'images': len(names), 'pairs': len(labels), 'positive_pairs': int(labels.sum()),
                'files': names, 'native_sizes': sizes, 'protocol_sha256': digest,
                'lr28_sha256': sha(folder / 'lr28.npy'), 'native112_sha256': native_digest.hexdigest(),
                'source_paths': [qmul.paths[name].relative_to(ROOT).as_posix() for name in names],
                'selection': 'All unique images referenced by the full official pair lists; sorted filenames',
                'clean_hr_reference_available': False, 'folds': 'Fixed stratified 10-fold assignment, seed 0; calibration excludes each evaluation fold'}
    write_json(folder / 'manifest.json', manifest)
    return r, manifest


class FixedSampling(nn.Module):
    def __init__(self, original):
        super().__init__(); self.weight, self.bias = original.weight, original.bias
    def forward(self, x): return F.conv2d(x, self.weight, self.bias, padding=1)


class FidelityBlend(nn.Module):
    def __init__(self, model, weight=.5): super().__init__(); self.model, self.weight = model, weight
    def forward(self, x):
        return self.weight * self.model(x).clamp(0, 1) + (1 - self.weight) * F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)


def factory(n, name):
    paths = {'conv_baseline': ROOT / 'work/full/ckpt/baseline_conv_best.pt',
             'ours_original': ROOT / 'work/full/ckpt/ours_ssc_attn_best.pt',
             'ours_reconstruction': ROOT / 'work/visual_sr/full_reconstruction/best.pt'}
    meta = {'method': name, 'role': 'project-trained', 'input': 'same cached LR28 RGB pixels', 'output': '112x112 RGB'}
    if name in {'native_bicubic', 'bicubic'}:
        meta['role'] = 'interpolation'; return None, meta
    if name in {'gfpgan', 'codeformer', 'realesrgan'}:
        meta['role'] = 'external_pretrained'
        if name == 'gfpgan':
            from restoration_model import FaceRestoration, WEIGHTS
            model = FaceRestoration(n.device); assets = [WEIGHTS]; meta['settings'] = {'version': '1.4', 'strength': 1., 'internal_size': 512}
        elif name == 'codeformer':
            from codeformer_model import CodeFormerRestoration, WEIGHTS
            model = CodeFormerRestoration(n.device, fidelity=1.); assets = [WEIGHTS]; meta['settings'] = {'fidelity': 1., 'adain': True, 'internal_size': 512}
        else:
            from real_esrgan_model import GenericRestoration
            model = GenericRestoration(n.device, denoise=.5)
            assets = [ROOT / 'work/realesrgan_assets' / f for f in ['realesr-general-x4v3.pth', 'realesr-general-wdn-x4v3.pth']]
            meta['settings'] = {'model': 'general-x4v3 compact', 'denoise': .5}
        meta['asset_sha256'] = {p.name: sha(p) for p in assets}
        meta['alignment'] = 'Already cropped faces; no detector, no extra landmark alignment; unsuitable pose/alignment is a limitation'
    else:
        path = ROOT / 'work/qmul_domain' / name / 'best.pt' if name in DOMAIN_METHODS else paths.get(name, MIXED)
        state = torch.load(path, map_location='cpu', weights_only=True)
        flags = state.get('architecture', {}) if name in DOMAIN_METHODS else ({'use_ssc': False, 'use_attn': False} if name == 'conv_baseline' else {})
        model = n.SRNet(n.cfg, **flags)
        model.load_state_dict(state['model'], strict=True)
        if name in DOMAIN_METHODS: meta['training_manifest'] = state['manifest']
        del state
        model.to(n.device).eval().requires_grad_(False)
        meta['checkpoint'] = path.relative_to(ROOT).as_posix(); meta['checkpoint_sha256'] = sha(path)
        if name == 'ours_flip2': model = SelfEnsemble(model, 2); meta['inference_passes'] = 2
        if name == 'ours_no_attention':
            for block in model.body: block.attn = None
            meta['role'] = 'inference_intervention'; meta['intervention'] = 'Attention bypassed in the mixed checkpoint; not independently retrained'
        if name == 'ours_fixed_sampling':
            for block in model.body: block.spatial = FixedSampling(block.spatial)
            meta['role'] = 'inference_intervention'; meta['intervention'] = 'Learned offsets disabled; original convolution coefficients retained; no retraining'
        if name == 'ours_dense_keys':
            for block in model.body: block.attn.keep_ratio = 1.
            meta['role'] = 'inference_intervention'; meta['intervention'] = 'All keys retained instead of 50%; no retraining'
        if name == 'ours_fidelity50':
            model = FidelityBlend(model, .5); meta['role'] = 'inference_intervention'; meta['intervention'] = 'Fixed 50% mixed-SR + 50% bicubic blend; not tuned on QMUL'
    model.eval().requires_grad_(False)
    meta['parameters'] = n.n_params(model)
    return model, meta


def evaluate(r, manifest, name, embeddings, meta):
    with np.load(OUT / 'cache/pairs.npz') as data:
        ia, ib, labels, folds = [data[key] for key in ['ia', 'ib', 'labels', 'folds']]
    result = {'method': name, 'metadata': meta, 'images': manifest['images'], 'pairs': manifest['pairs'],
              'protocol_sha256': manifest['protocol_sha256'], 'evaluators': {}}
    per_pair = {'labels': labels, 'folds': folds}
    for net, embedding in embeddings.items():
        score = (embedding[ia] * embedding[ib]).sum(1).astype(np.float64)
        prediction = np.zeros(len(labels), bool); thresholds = []
        for fold in range(10):
            threshold = r.VerificationEval.best_threshold(score[folds != fold], labels[folds != fold])
            prediction[folds == fold] = score[folds == fold] >= threshold; thresholds.append(float(threshold))
        metrics = r.VerificationEval()(embedding[ia], embedding[ib], labels, folds)
        assert abs(metrics['acc'] - (prediction == labels).mean()) < 1e-12
        metrics.update(correct_pairs=int((prediction == labels).sum()), thresholds=thresholds)
        result['evaluators'][net] = metrics
        per_pair[net + '_scores'], per_pair[net + '_correct'] = score, prediction == labels
    np.savez(OUT / 'emb' / (name + '_pairs.npz'), **per_pair)
    write_json(OUT / 'results' / (name + '.json'), result)
    print('RESULT', name, json.dumps({k: {m: v[m] for m in ['acc', 'auc', 'tar_far1']} for k, v in result['evaluators'].items()}), flush=True)
    return result


@torch.inference_mode()
def run(n, r, manifest, name, faces, batch):
    model, meta = factory(n, name)
    meta.update(source_sha256=sha(__file__), recognizer_sha256={
        'arcface': sha(ROOT / 'work/recognition/w600k_r50.onnx'),
        'mobileface': sha(ROOT / 'work/refinement_assets/w600k_mbf.onnx')},
        lr28_sha256=manifest['lr28_sha256'], native112_sha256=manifest['native112_sha256'],
        runtime={'torch': torch.__version__, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
                 'matmul_tf32': False, 'cudnn_tf32': False}, batch=batch)
    fingerprint = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    destination = OUT / 'results' / (name + '.json')
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved['metadata'].get('fingerprint') != fingerprint: raise ValueError(f'Completed-result mismatch for {name}; do not silently reuse')
        print('VERIFIED COMPLETED RESULT', name, flush=True); return
    meta['fingerprint'] = fingerprint
    low = np.load(OUT / 'cache/lr28.npy', mmap_mode='r')
    bank = {key: [] for key in faces}; examples = []; example_ids = np.linspace(0, len(low) - 1, 6, dtype=int).tolist()
    start = last = time.monotonic(); inference_seconds = 0.
    for offset in range(0, len(low), batch):
        def tensor(array): return torch.from_numpy(np.array(array, copy=True)).permute(0, 3, 1, 2).float().to(n.device) / 255
        x = tensor(low[offset:offset + batch])
        torch.cuda.synchronize(); tick = time.perf_counter()
        if name == 'native_bicubic':
            images = []
            for relative in manifest['source_paths'][offset:offset + len(x)]:
                with Image.open(ROOT / relative) as im: images.append(np.asarray(im.convert('RGB').resize((112, 112), Image.Resampling.BICUBIC)))
            output = tensor(np.stack(images))
        elif model is None: output = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        else: output = model(x).clamp(0, 1)
        torch.cuda.synchronize(); inference_seconds += time.perf_counter() - tick
        assert output.shape == (len(x), 3, 112, 112) and torch.isfinite(output).all()
        for net, face in faces.items(): bank[net].append(face(output).cpu().numpy())
        for i in example_ids:
            if offset <= i < offset + len(x): examples.append((i, (output[i-offset].permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')))
        if time.monotonic() - last > 25:
            print('BENCHMARK', name, min(offset + len(x), len(low)), '/', len(low), 'seconds', round(time.monotonic() - start, 1), flush=True)
            last = time.monotonic()
    embeddings = {key: np.concatenate(values) for key, values in bank.items()}
    meta.update(total_seconds=time.monotonic() - start, reconstruction_seconds=inference_seconds,
                mean_amortized_reconstruction_ms=1000 * inference_seconds / len(low),
                latency_note='Actual batched run, includes cold model kernels; excludes recognition; not batch-1 latency')
    np.savez(OUT / 'examples' / (name + '.npz'), indices=np.array([i for i, value in examples]), images=np.stack([value for i, value in examples]))
    evaluate(r, manifest, name, embeddings, meta)
    del model, bank, embeddings; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS + DOMAIN_METHODS, default=METHODS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    for name in ['cache', 'emb', 'results', 'examples']: (OUT / name).mkdir(parents=True, exist_ok=True)
    r, manifest = prepare()
    if args.prepare_only: return
    n, _ = runtime()
    if n.device != 'cuda': raise RuntimeError('This full benchmark expects the available CUDA GPU')
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    faces = {'arcface': FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False),
             'mobileface': FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)}
    for name in args.methods: run(n, r, manifest, name, faces, args.batch)
    print('BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__': main()

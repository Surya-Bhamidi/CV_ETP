"""Full-cohort heavy baselines using AMP, plus official 23-block RRDB x4plus.

Restoration compute precision is recorded; both recognition evaluators stay FP32.
AMP/FP32 agreement is checked on a fixed diagnostic slice, never pair labels.
"""
import gc
import json
import argparse

import numpy as np
import torch
from torch import nn

import benchmark_qmul as bench
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime


class AutocastRestorer(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.float16): result = self.model(x)
        return result.float()


class TF32Restorer(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x):
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=True):
                return self.model(x).float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


class RRDBRestorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        from restoration_vendor.rrdbnet_arch import RRDBNet
        self.net = RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        state = torch.load(ROOT / 'work/rrdb_assets/RealESRGAN_x4plus.pth', map_location='cpu', weights_only=True)
        self.net.load_state_dict(state['params_ema'], strict=True)
        self.to(device).eval().requires_grad_(False)
    def forward(self, x): return self.net(x.contiguous()).clamp(0, 1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['gfpgan_amp', 'codeformer_tf32', 'realesrgan_x4plus'], default=['gfpgan_amp', 'codeformer_tf32', 'realesrgan_x4plus'])
    args = parser.parse_args()
    r, manifest = bench.prepare(); n, _ = runtime()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    faces = {'arcface': FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False),
             'mobileface': FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)}
    base_factory = bench.factory
    diagnostic_path = bench.OUT / 'precision_diagnostics.json'
    diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    def factory(n, name):
        if name == 'realesrgan_x4plus':
            model = RRDBRestorer(n.device)
            return model, {'method': name, 'role': 'external_pretrained', 'parameters': n.n_params(model),
                'asset_sha256': sha(ROOT / 'work/rrdb_assets/RealESRGAN_x4plus.pth'),
                'source_sha256': sha(ROOT / 'restoration_vendor/rrdbnet_arch.py'),
                'input': 'same cached LR28 RGB pixels', 'output': '112x112 RGB',
                'settings': {'RRDB_blocks': 23, 'channels': 64, 'precision': 'FP32', 'face_enhancement': False}}
        base_name = 'codeformer' if name == 'codeformer_tf32' else name.removesuffix('_amp')
        model, meta = base_factory(n, base_name)
        low = np.load(bench.OUT / 'cache/lr28.npy', mmap_mode='r')[:8]
        x = torch.from_numpy(np.array(low, copy=True)).permute(0, 3, 1, 2).float().to(n.device).contiguous() / 255
        full = model(x).clamp(0, 1)
        if base_name == 'codeformer': model.chunk_size = 2
        wrapped = (TF32Restorer(model) if base_name == 'codeformer' else AutocastRestorer(model)).eval().requires_grad_(False)
        amp = wrapped(x).clamp(0, 1)
        assert torch.isfinite(amp).all() and amp.shape == full.shape
        cosine = (faces['mobileface'](full) * faces['mobileface'](amp)).sum(1)
        diagnostics[name] = {'images': 8, 'rgb_mean_absolute_difference_255': float((full-amp).abs().mean()*255),
                             'max_rgb_difference_255': float((full-amp).abs().max()*255),
                             'mobileface_output_agreement_mean_cosine': float(cosine.mean()),
                             'selection': 'First eight filenames; numeric diagnostic only, not a claim of global equivalence'}
        write_json(bench.OUT / 'precision_diagnostics.json', diagnostics)
        precision = 'FP32 tensors with TF32 restoration contractions; FP32 recognizers with TF32 disabled' if base_name == 'codeformer' else 'CUDA autocast FP16 with FP32 weights; FP32 recognizers'
        meta.update(method=name, restoration_precision=precision,
                    wrapper_source_sha256=sha(__file__), precision_diagnostic=diagnostics[name],
                    restoration_micro_batch=2)
        return wrapped, meta
    bench.factory = factory
    for name in args.methods:
        bench.run(n, r, manifest, name, faces, 32)
        gc.collect(); torch.cuda.empty_cache()
    print('EXTERNAL FULL BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__': main()

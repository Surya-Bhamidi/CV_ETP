"""Verify real checkpoint attention, cost, masks, gradients and validation policies."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from input_quality import InputDegradation
from investigate_identity import load
from refine_model import sha, write_json
from run_project import ROOT, runtime
from sparse_attention import SpatialSparseLinearAttention, install_attention

OUT = ROOT / 'work/attention_investigation'
BASE = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'


def correctness(n):
    checks = {}
    torch.manual_seed(42)
    old = n.AdaptiveSparseLinearAttention(8).double()
    new = SpatialSparseLinearAttention(8).double()
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 8, 5, 7, dtype=torch.float64, requires_grad=True)
    # Associative factorization must match explicit normalized NxN attention.
    q, k, v = old.qkv(x).flatten(2).chunk(3, dim=1)
    q, k = F.elu(q) + 1, (F.elu(k) + 1) * old.key_mask(x)[:, None]
    weights = q.transpose(1, 2) @ k
    expected = old.proj((v @ (weights / (weights.sum(-1, keepdim=True) + 1e-6)).transpose(1, 2)).reshape_as(x))
    assert torch.allclose(old(x), expected, atol=1e-10, rtol=1e-9)
    checks['factorized_attention_matches_explicit_normalized_kernel_attention'] = True
    # Training forward/backward must preserve the original scorer surrogate.
    a, b = old(x), new(x)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-10)
    a.square().mean().backward(); b.square().mean().backward()
    assert all(torch.allclose(p.grad, q.grad, atol=1e-12, rtol=1e-9) for p, q in zip(old.parameters(), new.parameters()))
    assert old.score[0].weight.grad.abs().sum() > 0
    checks['training_outputs_and_all_parameter_gradients_preserved'] = True
    errors = []
    for h, w in [(1, 1), (5, 7), (8, 8)]:
        for ratio in [.25, .5, .75, 1.]:
            value = torch.randn(2, 8, h, w, dtype=torch.float64)
            old.keep_ratio = new.keep_ratio = ratio
            old.eval(); new.eval()
            error = (old(value) - new(value)).abs().max().item(); errors.append(error)
            assert error < 1e-10
            for policy in ['global', 'quadrant']:
                new.policy = policy
                mask = new.key_mask(value)
                assert torch.all(mask.round().sum(1) == max(1, int(ratio * h * w)))
            new.policy = 'global'
    checks['gathered_eval_matches_dense_for_12_shape_budget_cases'] = True
    checks['both_policies_keep_exact_key_budget_including_odd_shapes'] = True
    checks['max_float64_gather_error'] = max(errors)
    conv = n.SparseSamplingConv(8)
    value = torch.randn(2, 8, 8, 8, requires_grad=True)
    assert torch.allclose(conv(value), F.conv2d(value, conv.weight, conv.bias, padding=1), atol=2e-6)
    conv(value).square().mean().backward()
    assert conv.offset.weight.grad.abs().sum() > 0
    checks['ssc_is_deformable_conv_with_learnable_offsets'] = True
    return checks


@torch.inference_mode()
def measure(fn, value, device, repeats=40):
    for _ in range(10): fn(value)
    times = []
    for _ in range(repeats):
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter(); fn(value)
        if device == 'cuda': torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)), 'repeats': repeats}


@torch.inference_mode()
def validation(n, model, sets):
    model.eval(); values = {}
    for profile, data in sets.items():
        psnr, ssim = [], []
        for low, target in data.batches(32):
            target = target.to(n.device); sr = model(low.to(n.device)).clamp(0, 1)
            psnr.extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
            ssim.extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
        values[profile] = {'psnr_y': float(np.mean(psnr)), 'ssim_y': float(np.mean(ssim))}
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-count', type=int, default=256)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n, _ = runtime(); n.torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    checks = correctness(n)
    model = load(n, BASE)
    count = n.n_params(model)
    assert count == 479109 and len(model.body) == 6
    assert all(isinstance(block.spatial, n.SparseSamplingConv) and isinstance(block.attn, n.AdaptiveSparseLinearAttention) for block in model.body)
    report = {'checks': checks, 'checkpoint': BASE.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(BASE),
              'parameters': count, 'fp32_weight_bytes': count * 4, 'blocks': 6, 'channels': 48,
              'attention_parameters': sum(n.n_params(b.attn) for b in model.body),
              'ssc_parameters': sum(n.n_params(b.spatial) for b in model.body),
              'query_positions_at_28': 784, 'key_positions_at_28': 392,
              'attention_math': 'Normalized ELU+1 kernel attention; no softmax NxN matrix. O(N*C*C) contractions at fixed C.',
              'sparsity': 'Content-adaptive positions, fixed 50% count. Original implementation masks dense tensors; it does not gather or skip keys.',
              'sampling': '9 learned deformable offsets per position, tanh bounded +/-2 around the 3x3 grid, within 7x7 support; not correlation-selected ASC-SRN.',
              'exact_paper_reproduction_established': False,
              'hardware': torch.cuda.get_device_name(0) if n.device == 'cuda' else 'CPU', 'latency': {}}
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split: ids = split['validation'][:args.val_count]
    sets = {p: n.FixedEvalSet(hr, ids, InputDegradation(n, p), 112, 1000) for p in ['standard28', 'mild28', 'clean28']}
    low, _ = next(sets['standard28'].batches(2)); low = low.to(n.device)
    capture = []
    def hook(layer, args):
        x = args[0]; mask = layer.key_mask(x).round()
        capture.append({'kept_per_image': mask.sum(1).cpu().tolist(),
                        'mask_disagreement_two_faces': (mask[0] != mask[1]).float().mean().item(),
                        'scorer_score_std': layer.score(x).sigmoid().std().item()})
    hooks = [b.attn.register_forward_pre_hook(hook) for b in model.body]
    with torch.inference_mode(): model(low)
    for h in hooks: h.remove()
    report['trained_key_masks'] = capture
    assert all(c['kept_per_image'] == [392., 392.] for c in capture)
    assert any(c['mask_disagreement_two_faces'] > .05 for c in capture)
    optimized = install_attention(copy.deepcopy(model))
    with torch.inference_mode():
        original_output, gathered_output = model(low), optimized(low)
        error = (original_output - gathered_output).abs().max().item()
        pixel_error = ((original_output.clamp(0, 1) * 255).round() - (gathered_output.clamp(0, 1) * 255).round()).abs().max().item()
        # FP32 reduction order can perturb subsequent hard top-k boundaries.
        # Mathematical equivalence was checked separately in float64 above.
        assert error < 1 / 255 and pixel_error <= 1
    report['checks']['whole_trained_network_gather_max_error'] = error
    report['checks']['whole_trained_network_png_max_difference_levels'] = pixel_error
    report['gather_equivalence_scope'] = 'Float64 module equivalence; FP32 complete-network agreement within one 8-bit level on these two validation inputs, not bitwise equality.'
    # Warm timing of both attention and the entire model, with identical weights.
    for size in [28, 56, 112]:
        x = torch.rand(1, 3, size, size, device=n.device)
        report['latency'][str(size)] = {'original_model': measure(model, x, n.device),
                                       'gather_model': measure(optimized, x, n.device)}
        z = torch.rand(1, 48, size, size, device=n.device)
        report['latency'][str(size)]['original_attention'] = measure(model.body[0].attn, z, n.device)
        report['latency'][str(size)]['gather_attention'] = measure(optimized.body[0].attn, z, n.device)
    write_json(OUT / 'audit.json', report)
    print('AUDIT', json.dumps(report), flush=True)
    policies = [('global50', 'global', .5, 1.), ('global25', 'global', .25, 1.),
                ('global75', 'global', .75, 1.), ('all_keys', 'global', 1., 1.),
                ('quadrant50', 'quadrant', .5, 1.), ('attention_half', 'global', .5, .5),
                ('attention_off', 'global', .5, 0.)]
    results = {'validation_images_per_profile': len(ids), 'test_used_for_selection': False,
               'scope': 'Inference interventions on one trained checkpoint; not independently trained architecture ablations', 'policies': {}}
    for label, policy, ratio, strength in policies:
        candidate = install_attention(copy.deepcopy(model), policy, ratio, strength=strength)
        results['policies'][label] = validation(n, candidate, sets)
        write_json(OUT / 'validation_attention.json', results)
        print('ATTENTION VALIDATION', label, json.dumps(results['policies'][label]), flush=True)
        del candidate
    print('AUDIT COMPLETE', flush=True)


if __name__ == '__main__': main()

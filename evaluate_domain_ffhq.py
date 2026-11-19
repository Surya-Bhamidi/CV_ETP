"""Check the preselected full domain candidate on paired FFHQ test images."""
import json

import numpy as np
import torch

from benchmark_qmul import factory
from input_quality import InputDegradation
from investigate_identity import load, save_grid
from refine_model import write_json
from run_project import ROOT, runtime


@torch.inference_mode()
def main():
    n, _ = runtime(); torch.backends.cudnn.benchmark = False
    old = load(n, ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt')
    new, _ = factory(n, 'domain_full')
    hr = np.load(ROOT / 'work/data/ffhq128_u8.npy', mmap_mode='r')
    with np.load(ROOT / 'work/full/split_ids.npz') as split: ids = split['test'][:512]
    out = ROOT / 'work/qmul_study'; result = {'images_per_profile': len(ids), 'checkpoint_chosen_before_test': True, 'profiles': {}}
    for profile in ['standard28', 'clean28']:
        data = n.FixedEvalSet(hr, ids, InputDegradation(n, profile), 112, 2000)
        values, rows = {name: {'psnr_y': [], 'ssim_y': []} for name in ['ours_mixed', 'domain_full']}, None
        for low, target in data.batches(32):
            low, target = low.to(n.device), target.to(n.device)
            views = {'ours_mixed': old(low).clamp(0, 1), 'domain_full': new(low).clamp(0, 1)}
            for name, sr in views.items():
                values[name]['psnr_y'].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
                values[name]['ssim_y'].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
            if rows is None: rows = {'Clean reference': target.cpu(), 'Input (28)': low.cpu(), **{k: v.cpu() for k, v in views.items()}}
        result['profiles'][profile] = {name: {key: float(np.mean(v)) for key, v in scores.items()} for name, scores in values.items()}
        save_grid(rows, out / ('domain_ffhq_' + profile + '.png'), 'FFHQ TEST | ' + profile + ' | camera adaptation tradeoff')
    result['quality_guards'] = {profile: values['domain_full']['psnr_y'] >= values['ours_mixed']['psnr_y'] - .5 and values['domain_full']['ssim_y'] >= values['ours_mixed']['ssim_y'] - .01 for profile, values in result['profiles'].items()}
    write_json(out / 'domain_ffhq_test.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

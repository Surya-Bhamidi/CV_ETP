"""Measure the validation-selected two-pass option on all QMUL verification pairs."""
import torch

from audit_attention import BASE, OUT
from evaluate_identity_release import surveillance
from investigate_identity import load
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, runtime
from sparse_attention import SelfEnsemble


def main():
    n, _ = runtime(); torch.backends.cudnn.benchmark = False
    single = load(n, BASE)
    flip = SelfEnsemble(single, 2).eval()
    faces = {'arcface': FrozenFace(ROOT / 'work/recognition/w600k_r50.onnx').to(n.device).eval().requires_grad_(False),
             'mobileface': FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)}
    folder = OUT / 'flip2_qmul'; folder.mkdir(parents=True, exist_ok=True)
    result = surveillance(n, single, flip, faces, folder)
    for key in ['conditions', 'mobileface_conditions']:
        result[key]['single_pass'] = result[key].pop('current')
        result[key]['flip2'] = result[key].pop('mixed_quality')
    result['checkpoint_sha256'] = sha(BASE)
    result['selection'] = 'Two passes selected on FFHQ validation before held-out reconstruction and QMUL evaluation'
    result['additional_model_parameters'] = 0
    write_json(OUT / 'qmul_ensemble_verification.json', result)
    print('QMUL FLIP EVALUATION COMPLETE', flush=True)


if __name__ == '__main__': main()

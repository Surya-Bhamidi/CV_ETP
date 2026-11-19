"""Sequence GPU experiments after heavy baseline evaluation finishes."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'work/qmul_study'


def status(phase, **details):
    (OUT / 'pipeline_status.json').write_text(json.dumps({'phase': phase, **details}, indent=2), encoding='utf-8')


def main():
    status('waiting_for_external_baselines')
    needed = ['gfpgan_amp', 'codeformer_tf32', 'realesrgan_x4plus']
    while not all((OUT / 'results' / (name + '.json')).exists() for name in needed): time.sleep(5)
    time.sleep(3)
    variants = ['domain_full', 'domain_ffhq_control', 'domain_no_attention', 'domain_no_ssc']
    try:
        for variant in variants:
            status('training', variant=variant)
            with (ROOT / 'work' / (variant + '.log')).open('w') as output, (ROOT / 'work' / (variant + '.err')).open('w') as errors:
                subprocess.run([sys.executable, '-u', str(ROOT / 'train_qmul_domain.py'), '--variant', variant], cwd=ROOT, stdout=output, stderr=errors, check=True)
            print('TRAINED', variant, flush=True)
        status('evaluating_domain_variants_on_full_fixed_qmul_cohort')
        with (ROOT / 'work/qmul_domain_benchmark.log').open('w') as output, (ROOT / 'work/qmul_domain_benchmark.err').open('w') as errors:
            subprocess.run([sys.executable, '-u', str(ROOT / 'benchmark_qmul.py'), '--batch', '32', '--methods', *variants], cwd=ROOT, stdout=output, stderr=errors, check=True)
        status('all_results_ready')
        print('DOMAIN STUDY COMPLETE', flush=True)
    except Exception as error:
        status('failed', error=str(error)); raise


if __name__ == '__main__': main()

"""Download pinned official GFPGAN inference sources and weights for comparison.

Downloads only; this script does not import or execute downloaded source code.
"""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'work/restoration_assets'


def fetch(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'CV-ETP-local-research'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as target:
        expected = int(response.headers.get('Content-Length', 0))
        count = 0
        last = time.monotonic()
        while chunk := response.read(4 * 1024 * 1024):
            target.write(chunk)
            count += len(chunk)
            if time.monotonic() - last > 15:
                print(f'{path.name}: {count/1e6:.1f}/{expected/1e6:.1f} MB', flush=True)
                last = time.monotonic()
    if expected and expected != count:
        raise IOError('Incomplete download')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit_file = OUT / 'commit.json'
    fetch('https://api.github.com/repos/TencentARC/GFPGAN/commits/master', commit_file)
    commit = json.loads(commit_file.read_text(encoding='utf-8'))['sha']
    sources = {}
    for name in ['gfpganv1_clean_arch.py', 'stylegan2_clean_arch.py']:
        sources[name] = f'https://raw.githubusercontent.com/TencentARC/GFPGAN/{commit}/gfpgan/archs/{name}'
    sources['LICENSE'] = f'https://raw.githubusercontent.com/TencentARC/GFPGAN/{commit}/LICENSE'
    sources['GFPGANv1.4.pth'] = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    records = {}
    for name, url in sources.items():
        path = OUT / name
        fetch(url, path)
        records[name] = {'url': url, 'bytes': path.stat().st_size,
                         'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
        print('Downloaded', name, records[name]['bytes'], flush=True)
    (OUT / 'provenance.json').write_text(json.dumps({'commit': commit, 'files': records,
        'role': 'External pretrained baseline; not the custom project model and not trained in this project.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

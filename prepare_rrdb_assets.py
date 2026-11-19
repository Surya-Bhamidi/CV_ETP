"""Download official RRDB x4plus weights and pinned architecture/license as data."""
import hashlib
import json
from pathlib import Path

from prepare_restoration import fetch


def main():
    out = Path(__file__).resolve().parent / 'work/rrdb_assets'; out.mkdir(exist_ok=True)
    fetch('https://api.github.com/repos/XPixelGroup/BasicSR/commits/master', out / 'commit.json')
    commit = json.loads((out / 'commit.json').read_text(encoding='utf-8'))['sha']
    sources = {'rrdbnet_arch.py': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{commit}/basicsr/archs/rrdbnet_arch.py',
               'LICENSE.txt': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{commit}/LICENSE.txt',
               'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'}
    records = {}
    for name, url in sources.items():
        fetch(url, out / name)
        with (out / name).open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records[name] = {'url': url, 'sha256': digest, 'bytes': (out / name).stat().st_size}
    (out / 'provenance.json').write_text(json.dumps({'commit': commit, 'files': records, 'role': 'Official 23-RRDB x4 model; downloaded source is not executed'}, indent=2), encoding='utf-8')
    print('RRDB DOWNLOAD COMPLETE', flush=True)


if __name__ == '__main__': main()

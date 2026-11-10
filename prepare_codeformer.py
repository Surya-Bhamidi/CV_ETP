"""Fetch the official CodeFormer baseline without running downloaded code."""
import hashlib
import json
from pathlib import Path
from prepare_restoration import fetch


def main():
    out = Path(__file__).resolve().parent / 'work/codeformer_assets'
    out.mkdir(parents=True, exist_ok=True)
    fetch('https://api.github.com/repos/sczhou/CodeFormer/commits/master', out / 'commit.json')
    commit = json.loads((out / 'commit.json').read_text(encoding='utf-8'))['sha']
    sources = {name:f'https://raw.githubusercontent.com/sczhou/CodeFormer/{commit}/basicsr/archs/{name}'
               for name in ['codeformer_arch.py','vqgan_arch.py']}
    sources['LICENSE'] = f'https://raw.githubusercontent.com/sczhou/CodeFormer/{commit}/LICENSE'
    sources['codeformer.pth'] = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    records = {}
    for name,url in sources.items():
        fetch(url,out/name)
        with (out/name).open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[name]={'url':url,'sha256':digest,'bytes':(out/name).stat().st_size}
        print('Downloaded',name,records[name]['bytes'],flush=True)
    (out/'provenance.json').write_text(json.dumps({'commit':commit,'files':records,
        'role':'External pretrained research baseline; not trained as part of this project.'},indent=2))


if __name__ == '__main__':
    main()

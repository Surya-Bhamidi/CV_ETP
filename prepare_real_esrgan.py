"""Download an official generic SR baseline, without running remote code."""
import hashlib,json
from pathlib import Path
from prepare_restoration import fetch


if __name__=='__main__':
    out=Path(__file__).resolve().parent/'work/realesrgan_assets';out.mkdir(parents=True,exist_ok=True)
    fetch('https://api.github.com/repos/xinntao/Real-ESRGAN/commits/master',out/'commit.json')
    commit=json.loads((out/'commit.json').read_text(encoding='utf-8'))['sha']
    sources={'srvgg_arch.py':f'https://raw.githubusercontent.com/xinntao/Real-ESRGAN/{commit}/realesrgan/archs/srvgg_arch.py',
             'LICENSE':f'https://raw.githubusercontent.com/xinntao/Real-ESRGAN/{commit}/LICENSE'}
    for name in ['realesr-general-x4v3.pth','realesr-general-wdn-x4v3.pth']:
        sources[name]='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'+name
    records={}
    for name,url in sources.items():
        fetch(url,out/name)
        with (out/name).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[name]={'url':url,'sha256':digest,'bytes':(out/name).stat().st_size}
        print('Downloaded',name,records[name]['bytes'],flush=True)
    (out/'provenance.json').write_text(json.dumps({'commit':commit,'files':records,'role':'External generic SR baseline, not trained in this project.'},indent=2))

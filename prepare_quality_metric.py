"""Download the official LPIPS SqueezeNet backbone into the workspace."""
import hashlib
import json
from pathlib import Path
from torchvision.models import SqueezeNet1_1_Weights
from prepare_restoration import fetch


if __name__ == '__main__':
    out = Path(__file__).resolve().parent / 'work/quality_assets'
    out.mkdir(parents=True, exist_ok=True)
    url = SqueezeNet1_1_Weights.IMAGENET1K_V1.url
    path = out / 'squeezenet1_1-b8a52dc0.pth'
    fetch(url, path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert digest.startswith('b8a52dc0')
    (out / 'provenance.json').write_text(json.dumps({'url': url, 'sha256': digest,
        'metric': 'Official LPIPS 0.1.4, SqueezeNet v0.1 calibrated weights packaged with lpips.',
        'role': 'Evaluation only, never used in training or checkpoint selection.'}, indent=2))
    print('LPIPS evaluation backbone ready', flush=True)

"""Fetch frozen feature networks used for supervised SR refinement."""
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "work" / "refinement_assets"


def download(url, destination):
    if destination.exists(): return
    partial = destination.with_suffix(destination.suffix + ".part")
    last = time.monotonic()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        count = 0
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk); count += len(chunk)
            if time.monotonic() - last > 15:
                print(f"{destination.name}: {count / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                last = time.monotonic()
    if total and count != total: raise IOError("Incomplete feature-network download")
    partial.replace(destination)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from torchvision.models import VGG16_Weights
    vgg_url = VGG16_Weights.IMAGENET1K_V1.url
    vgg = OUT / "vgg16-397923af.pth"
    download(vgg_url, vgg)
    vgg_sha = digest(vgg)
    if not vgg_sha.startswith("397923af"): raise ValueError("VGG16 checksum mismatch")
    url = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip"
    archive = OUT / "buffalo_sc.zip"
    download(url, archive)
    mbf = OUT / "w600k_mbf.onnx"
    if not mbf.exists():
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.endswith("w600k_mbf.onnx"))
            mbf.write_bytes(z.read(name))
    provenance = {"vgg16": {"url": vgg_url, "sha256": vgg_sha},
                  "mobilefacenet": {"url": url, "archive_sha256": digest(archive), "onnx_sha256": digest(mbf)},
                  "purpose": "VGG and ResNet50 ArcFace guide training; MobileFaceNet is used only for validation and evaluation."}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("Refinement assets ready", OUT, flush=True)


if __name__ == "__main__": main()

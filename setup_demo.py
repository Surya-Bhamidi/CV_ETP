"""Fetch the two official baseline checkpoints referenced by the comparison UI."""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from face_sr.paths import ROOT


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_weights(folder, filename):
    directory = ROOT / "work" / folder
    target = directory / filename
    record = json.loads((directory / "provenance.json").read_text())["files"][filename]
    if target.is_file() and checksum(target) == record["sha256"]:
        print(f"Verified {filename}")
        return
    request = urllib.request.Request(record["url"], headers={"User-Agent": "CV-ETP"})
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        if checksum(temporary) != record["sha256"]:
            raise ValueError(f"Downloaded checkpoint failed verification: {filename}")
        os.replace(temporary, target)
        print(f"Installed {filename}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    prepare_weights("restoration_assets", "GFPGANv1.4.pth")
    prepare_weights("codeformer_assets", "codeformer.pth")

"""Download the notebook's FFHQ128 source and build its uint8 cache."""
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parent / "work" / "data"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "ffhq128_u8.npy"
    if cache.exists():
        print("Existing cache:", np.load(cache, mmap_mode="r").shape, flush=True)
        return
    archive = root / "thumbnails128x128.zip"
    url = "https://huggingface.co/datasets/nuwandaa/ffhq128/resolve/main/thumbnails128x128.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        offset = 0
                    total = offset + int(response.headers.get("Content-Length", 0))
                    with partial.open("ab" if offset else "wb") as out:
                        last = time.monotonic()
                        while block := response.read(4 * 1024 * 1024):
                            out.write(block)
                            offset += len(block)
                            if time.monotonic() - last > 15:
                                print(f"FFHQ download: {offset / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                                last = time.monotonic()
                if total and partial.stat().st_size != total:
                    raise IOError("Incomplete FFHQ archive")
                partial.replace(archive)
                break
            except Exception as exc:
                print(f"Download attempt {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
    temp_cache = root / "ffhq128_u8.building.npy"
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")) and "__MACOSX" not in n)
        if not names:
            raise ValueError("FFHQ archive contains no images")
        arr = np.lib.format.open_memmap(temp_cache, mode="w+", dtype=np.uint8, shape=(len(names), 128, 128, 3))
        for i, name in enumerate(names):
            with z.open(name) as f, Image.open(f) as image:
                image = image.convert("RGB")
                if image.size != (128, 128):
                    image = image.resize((128, 128), Image.Resampling.BICUBIC)
                arr[i] = np.asarray(image)
            if (i + 1) % 10000 == 0:
                print(f"FFHQ cached: {i + 1}/{len(names)}", flush=True)
        arr.flush()
        del arr
    temp_cache.replace(cache)
    (root / "source.json").write_text(json.dumps({"dataset": "nuwandaa/ffhq128", "url": url, "images": len(names), "shape": [len(names), 128, 128, 3], "ordering": "sorted archive image paths"}, indent=2))
    print("FFHQ ready:", cache, "images:", len(names), flush=True)


if __name__ == "__main__":
    main()

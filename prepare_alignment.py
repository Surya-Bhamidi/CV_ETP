"""Fetch the small official OpenCV YuNet landmark detector for alignment trials."""
import hashlib
import json
from pathlib import Path
from prepare_restoration import fetch


if __name__ == '__main__':
    out=Path(__file__).resolve().parent/'work/alignment_assets'
    out.mkdir(parents=True,exist_ok=True)
    sources={'face_detection_yunet_2023mar.onnx':'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
             'LICENSE':'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE'}
    records={}
    for name,url in sources.items():
        fetch(url,out/name)
        with (out/name).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[name]={'url':url,'sha256':digest,'bytes':(out/name).stat().st_size}
    (out/'provenance.json').write_text(json.dumps(records,indent=2))
    print('YuNet alignment assets ready',flush=True)

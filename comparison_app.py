"""Local image comparison app backed by the fully trained SR checkpoint."""
import argparse
import base64
import io
import hashlib
import json
import logging
import math
import socket
import threading
import time
import urllib.request
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
import uvicorn

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "comparison_ui"
APP_ID = "cv-etp-compare-v1"
CHECKPOINT = ROOT / "work/full/ckpt/ours_ssc_attn_best.pt"
PROMOTION = ROOT / "work/refinement/promoted.json"
EXPERIMENTAL = ROOT / "work/refinement/experimental.json"
RECONSTRUCTION = ROOT / "work/visual_sr/selected.json"
MAX_BODY = 24 * 1024 * 1024
SERVER = None
log = logging.getLogger("comparison")


def png_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("Choose a PNG, JPEG, or WebP image.")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 8 MB.")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Supported image formats are PNG, JPEG, and WebP.")
        if image.width * image.height > 25_000_000:
            raise ValueError("Choose an image smaller than 25 megapixels.")
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB").copy()


class ReconstructionService:
    def __init__(self):
        self.state = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.sample_lock = threading.Lock()
        self.cache = {}
        self.test_ids = []
        self.faces = None
        self.default_model = "standard"

    def load(self):
        try:
            from run_project import notebook_definitions
            self.n = notebook_definitions("sr_part1_clean.ipynb", [1, 3, 5, 6], "comparison_runtime")
            torch = self.n.torch
            torch.set_num_threads(2)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cudnn.benchmark = False
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            settings = checkpoint.get("config", {})
            allowed = self.n.Config.__dataclass_fields__
            self.cfg = self.n.Config(**{k: v for k, v in settings.items() if k in allowed})
            self.model = self.n.SRNet(self.cfg)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.to(self.device).eval().requires_grad_(False)
            self.epoch = int(checkpoint.get("epoch", 0))
            del checkpoint
            self.models = {"standard": self.model}
            self.model_info = {"standard": {"id": "standard", "label": "Pixel model (original)", "epochs": self.epoch,
                                           "description": "30 epochs · pixel L1 reconstruction", "finetune_epochs": 0,
                                           "kind": "custom", "parameters": self.n.n_params(self.model)}}
            if RECONSTRUCTION.exists():
                try:
                    selected = json.loads(RECONSTRUCTION.read_text(encoding="utf-8"))
                    path = Path(selected["checkpoint"])
                    path = (path if path.is_absolute() else ROOT / path).resolve()
                    path.relative_to((ROOT / "work/visual_sr").resolve())
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if not selected.get("passes_reconstruction_checks") or digest != selected["checkpoint_sha256"]:
                        raise ValueError("Reconstruction checkpoint approval/checksum mismatch")
                    refined = torch.load(path, map_location="cpu", weights_only=True)
                    config = self.n.Config(**{k: v for k, v in refined["config"].items() if k in allowed})
                    model = self.n.SRNet(config)
                    model.load_state_dict(refined["model"], strict=True)
                    self.models["detail"] = model.to(self.device).eval().requires_grad_(False)
                    extra = int(refined["finetune_epoch"])
                    self.model_info["detail"] = {"id": "detail", "label": "Reconstruction refinement (ours)",
                        "epochs": int(refined["epoch"]), "finetune_epochs": extra, "kind": "custom", "parameters": self.n.n_params(model),
                        "description": f"30 base + {extra} fine-tune epochs · structure and perceptual reconstruction",
                        "reconstruction_validated": True,
                        "warning": "Fine-tuned for reconstruction on synthetic FFHQ pairs. Compare with Original to inspect remaining missing or changed facial details."}
                    if selected.get("default_model") == "detail":
                        self.default_model = "detail"
                    del refined
                except Exception:
                    log.exception("Reconstruction refinement unavailable; original remains available")
            elif PROMOTION.exists() or EXPERIMENTAL.exists():
                try:
                    approved = json.loads((PROMOTION if PROMOTION.exists() else EXPERIMENTAL).read_text(encoding="utf-8"))
                    path = Path(approved["checkpoint"])
                    path = (path if path.is_absolute() else ROOT / path).resolve()
                    path.relative_to((ROOT / "work/refinement").resolve())
                    if approved.get("passes_release") or approved.get("available_for_comparison"):
                        refined = torch.load(path, map_location="cpu", weights_only=True)
                        config = self.n.Config(**{k: v for k, v in refined["config"].items() if k in allowed})
                        model = self.n.SRNet(config)
                        model.load_state_dict(refined["model"], strict=True)
                        self.models["detail"] = model.to(self.device).eval().requires_grad_(False)
                        extra = int(refined["finetune_epoch"])
                        experimental = not bool(approved.get("passes_release"))
                        self.model_info["detail"] = {"id": "detail", "label": "Detail refinement (experimental)" if experimental else "Detail-refined model", "epochs": int(refined["epoch"]),
                            "experimental": experimental, "finetune_epochs": extra,
                            "kind": "custom", "parameters": self.n.n_params(model),
                            "warning": "Previous experimental refinement: recognizer features improved but image structure and the independent CCTV check regressed." if experimental else None,
                            "description": f"30 base + {extra} fine-tune epochs · feature and edge preservation"}
                        del refined
                except Exception:
                    log.exception("Detail model unavailable; original model remains available")
            from restoration_model import FaceRestoration, WEIGHTS
            if WEIGHTS.exists():
                try:
                    restoration = FaceRestoration(self.device)
                    self.models["restoration"] = restoration
                    self.model_info["restoration"] = {"id": "restoration", "label": "GFPGAN 1.4 (pretrained)",
                        "epochs": None, "finetune_epochs": 0, "kind": "external_pretrained", "parameters": self.n.n_params(restoration),
                        "description": "External pretrained restoration · aligned face crops · adjustable strength",
                        "warning": "Pretrained generative restoration can change facial features. These are estimated details, not verified recovery. This is GFPGAN, separate from our trained SR model. FFHQ pretraining overlap is not excluded."}
                except Exception:
                    log.exception("Optional pretrained restoration unavailable; custom models remain available")
            from codeformer_model import CodeFormerRestoration, WEIGHTS as CODEFORMER_WEIGHTS
            codeformer_review = ROOT / "work/codeformer_assets/ui_review.json"
            codeformer_enabled = False
            try:
                if codeformer_review.exists():
                    codeformer_enabled = bool(json.loads(codeformer_review.read_text(encoding="utf-8")).get("enabled", False))
            except (ValueError, OSError):
                log.warning("CodeFormer review metadata unavailable; leaving this optional model disabled")
            if CODEFORMER_WEIGHTS.exists() and codeformer_enabled:
                try:
                    restoration = CodeFormerRestoration(self.device, fidelity=1.)
                    self.models["codeformer"] = restoration
                    self.model_info["codeformer"] = {"id": "codeformer", "label": "CodeFormer (pretrained, fidelity 1)",
                        "epochs": None, "finetune_epochs": 0, "kind": "external_pretrained", "parameters": self.n.n_params(restoration),
                        "description": "External pretrained restoration · maximum CodeFormer fidelity · adjustable blend",
                        "warning": "CodeFormer is a separate pretrained research model. High fidelity aims to preserve the input, but generated facial details can still be wrong. FFHQ pretraining overlap is not excluded."}
                except Exception:
                    log.exception("Optional CodeFormer unavailable; other models remain available")
            with torch.inference_mode():
                for model in self.models.values(): model(torch.zeros(1, 3, 28, 28, device=self.device))
                if self.device == "cuda": torch.cuda.synchronize()
            self.hardware = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
            data, splits = ROOT / "work/data/ffhq128_u8.npy", ROOT / "work/full/split_ids.npz"
            if data.exists() and splits.exists():
                self.faces = np.load(data, mmap_mode="r")
                with np.load(splits) as z: self.test_ids = z["test"].tolist()
            self.state = "ready"
            log.info("Trained model ready on %s", self.hardware)
        except Exception:
            log.exception("Could not load trained model")
            self.error = "The trained checkpoint could not be loaded. Check the server log."
            self.state = "error"

    def require_ready(self):
        if self.state != "ready":
            raise HTTPException(503, self.error or "The trained model is still loading.")

    def status(self):
        return {"app_id": APP_ID, "state": self.state, "error": self.error,
                "hardware": getattr(self, "hardware", None), "scale": 4,
                "epochs": getattr(self, "epoch", None), "samples_available": bool(self.test_ids),
                "models": list(getattr(self, "model_info", {}).values()),
                "default_model": self.default_model}

    def pair(self, index):
        self.require_ready()
        if not self.test_ids or not 0 <= index < len(self.test_ids):
            raise ValueError("This test example is not available.")
        original = np.array(self.faces[self.test_ids[index], 8:120, 8:120], copy=True)
        degraded = self.n.Degrader(4)(original, np.random.default_rng(2000 + index))
        return Image.fromarray(degraded), Image.fromarray(original)

    def samples(self):
        self.require_ready()
        result = []
        for i in range(min(12, len(self.test_ids))):
            low, original = self.pair(i)
            result.append({"id": i, "name": f"Test face {i + 1:02}", "thumbnail": png_url(original), "input_size": list(low.size)})
        return result

    def tensor(self, image):
        return self.n.torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().unsqueeze(0).div(255).contiguous()

    def to_image(self, tensor):
        array = tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
        return Image.fromarray((array * 255).round().astype(np.uint8))

    def compare(self, image, reference=None, name="Your image", note=None, model_id="standard", strength=.75):
        self.require_ready()
        if model_id not in self.models: raise ValueError("This reconstruction model is not available.")
        if min(image.size) < 8:
            raise ValueError("Choose a face crop at least 8 pixels wide and high.")
        if model_id in {"restoration", "codeformer"} and image.width != image.height:
            raise ValueError("Pretrained face restoration needs a square, approximately aligned face crop. Crop the face to a square or choose a custom SR model.")
        if max(image.size) > 256:
            before = image.size
            ratio = 256 / max(before)
            new_size = tuple(max(1, round(d * ratio)) for d in before)
            if min(new_size) < 8:
                raise ValueError("Choose a less narrow image, ideally a cropped face.")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            note = f"The upload was resized from {before[0]} × {before[1]} to {image.width} × {image.height} for local processing."
        output_size = (image.width * 4, image.height * 4)
        if reference is not None and reference.size != output_size:
            raise ValueError(f"The clean original must be {output_size[0]} × {output_size[1]} pixels to match this input at ×4. It was not resized automatically.")
        torch = self.n.torch
        with self.lock, torch.inference_mode():
            low = self.tensor(image).to(self.device)
            if self.device == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            previous_gpu = self.model(low).clamp(0, 1) if model_id != "standard" else None
            if model_id in {"restoration", "codeformer"}:
                if strength == 0:
                    reconstructed = previous_gpu
                else:
                    restored = self.models[model_id](low).clamp(0, 1)
                    reconstructed = previous_gpu.lerp(restored, strength)
            else:
                reconstructed = self.models[model_id](low).clamp(0, 1)
            if self.device == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            sr = reconstructed.cpu()
            bicubic = self.n.F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1).cpu()
            previous = previous_gpu.cpu() if previous_gpu is not None else None
        metrics = None
        if reference is not None:
            target = self.tensor(reference)
            metrics = {}
            for key, value in [("sr", sr), ("bicubic", bicubic)]:
                metrics[key] = {"psnr": float(self.n.Metrics.psnr_y(value, target, 4).item()),
                                "ssim": float(self.n.Metrics.ssim_y(value, target, 4).item())}
        return {"name": name, "input": png_url(image), "sr": png_url(self.to_image(sr)),
                "bicubic": png_url(self.to_image(bicubic)), "reference": png_url(reference) if reference is not None else None,
                "input_size": list(image.size), "output_size": list(output_size), "metrics": metrics,
                "processing_ms": round(elapsed * 1000, 1), "note": note,
                "previous": png_url(self.to_image(previous)) if previous is not None else None,
                "model_id": model_id, "model_info": self.model_info[model_id],
                "strength": strength if model_id in {"restoration", "codeformer"} else None,
                "filename": "codeformer_restored_x4.png" if model_id == "codeformer" else "gfpgan_restored_x4.png" if model_id == "restoration" else "reconstructed_detail_x4.png" if model_id == "detail" else "reconstructed_x4.png"}

    def sample_result(self, index, model_id="standard", strength=.75):
        with self.sample_lock:
            key = (model_id, index, strength if model_id in {"restoration", "codeformer"} else None)
            if key not in self.cache:
                low, original = self.pair(index)
                self.cache[key] = self.compare(low, original, f"Test face {index + 1:02}",
                                                "FFHQ example held out from our custom model · identical degraded input for every option. External pretraining overlap is not excluded." if model_id in {"restoration", "codeformer"} else "Held-out FFHQ example · the same fixed degradation used in full-run testing.", model_id=model_id, strength=strength)
                if len(self.cache) > 48:
                    self.cache.pop(next(iter(self.cache)))
            return self.cache[key]


service = ReconstructionService()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=service.load, daemon=True, name="load-sr-model").start()
    yield


app = FastAPI(title="Face reconstruction lab", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.url.hostname not in {"127.0.0.1", "localhost"}:
        return JSONResponse({"detail": "Use the local application URL."}, status_code=403)
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://127.0.0.1:{request.url.port}", f"http://localhost:{request.url.port}"}:
            return JSONResponse({"detail": "This endpoint accepts local application requests only."}, status_code=403)
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            return JSONResponse({"detail": "Expected a JSON request."}, status_code=415)
        try:
            size = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid request size."}, status_code=400)
        if size > MAX_BODY:
            return JSONResponse({"detail": "The upload is too large."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/status")
def get_status(): return service.status()


@app.get("/api/samples")
def get_samples(): return {"samples": service.samples()}


@app.post("/api/compare")
def compare(payload: dict):
    service.require_ready()
    try:
        model_id = payload.get("model", service.default_model)
        if not isinstance(model_id, str) or model_id not in service.models: raise ValueError("Choose an available reconstruction model.")
        strength = payload.get("strength", .75)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Restoration strength must be between 0 and 1.")
        if "sample_id" in payload:
            index = payload["sample_id"]
            if not isinstance(index, int) or isinstance(index, bool): raise ValueError("Choose a valid test example.")
            return service.sample_result(index, model_id, strength)
        image = decode_image(payload.get("image"))
        if min(image.size) < 8: raise ValueError("Choose a face crop at least 8 pixels wide and high.")
        mode = payload.get("mode", "native")
        if mode not in {"native", "simulate"}: raise ValueError("Choose a valid input mode.")
        name = str(payload.get("name", "Your image"))[:100]
        if mode == "simulate":
            original = ImageOps.fit(image, (112, 112), method=Image.Resampling.BICUBIC)
            low = service.n.Degrader(4)(np.array(original), np.random.default_rng(42))
            return service.compare(Image.fromarray(low), original, name, "Simulation: a 112 × 112 center-fitted original was degraded to 28 × 28 before reconstruction.", model_id=model_id, strength=strength)
        reference = decode_image(payload["reference"]) if payload.get("reference") else None
        return service.compare(image, reference, name, model_id=model_id, strength=strength)
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        log.exception("Image reconstruction failed")
        raise HTTPException(500, "Reconstruction failed. Try a smaller cropped face image.") from error


@app.post("/api/shutdown")
def shutdown(payload: dict):
    if SERVER is not None:
        threading.Timer(.2, setattr, args=(SERVER, "should_exit", True)).start()
    return {"stopping": True}


@app.get("/")
def index(): return FileResponse(WEB / "index.html", media_type="text/html")


@app.get("/assets/{name}")
def asset(name: str):
    allowed = {"app.js": "application/javascript", "styles.css": "text/css"}
    if name not in allowed: raise HTTPException(404)
    return FileResponse(WEB / name, media_type=allowed[name])


def existing_app(url):
    try:
        with urllib.request.urlopen(url + "/api/status", timeout=2) as response:
            return json.load(response).get("app_id") == APP_ID
    except Exception: return False


def main():
    global SERVER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if existing_app(url):
        if args.stop:
            request = urllib.request.Request(url + "/api/shutdown", data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3): pass
            print("Comparison app stopped.")
        else:
            print("Comparison app is already running:", url)
            if args.open_browser: webbrowser.open(url)
        return
    if args.stop:
        print("The comparison app is not running.")
        return
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is busy. Choose another with --port.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("Comparison UI:", url, flush=True)
    if args.open_browser:
        def open_when_ready():
            for _ in range(40):
                if existing_app(url): webbrowser.open(url); return
                time.sleep(.25)
        threading.Thread(target=open_when_ready, daemon=True).start()
    SERVER = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False))
    SERVER.run()


if __name__ == "__main__":
    main()

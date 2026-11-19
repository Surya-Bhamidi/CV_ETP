"""HTTP routes and lifecycle for the local comparison app."""

import argparse
import json
import logging
import math
import socket
import threading
import time
import urllib.request
import webbrowser
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from face_sr.service import ReconstructionService
from face_sr.images import decode_image
from face_sr.paths import ROOT, WEB, APP_ID
from input_quality import InputDegradation, PROFILE_LABELS

MAX_BODY = 24 * 1024 * 1024
SERVER = None
log = logging.getLogger("comparison")


# %% Server lifecycle and request handlers
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
        if origin and origin not in {
            f"http://127.0.0.1:{request.url.port}",
            f"http://localhost:{request.url.port}",
        }:
            return JSONResponse(
                {"detail": "This endpoint accepts local application requests only."},
                status_code=403,
            )
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
def get_status():
    return service.status()


@app.get("/api/samples")
def get_samples(split: str = "test", profile: str = "standard28", focus: str = "all"):
    try:
        return {"samples": service.samples(split, profile, focus)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/compare")
def compare(payload: dict):
    service.require_ready()
    try:
        model_id = payload.get("model", service.default_model)
        if not isinstance(model_id, str) or model_id not in service.models:
            raise ValueError("Choose an available reconstruction model.")
        inference = payload.get("inference", "single")
        if not isinstance(inference, str) or inference not in {"single", "clean_consistency"}:
            raise ValueError("Choose an available refinement mode.")
        strength = payload.get("strength", 1.0)
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("Restoration strength must be between 0 and 1.")
        if "sample_id" in payload:
            index = payload["sample_id"]
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("Choose a valid test example.")
            return service.sample_result(
                index,
                model_id,
                strength,
                payload.get("sample_split", "test"),
                payload.get("profile", "standard28"),
                inference,
            )
        image = decode_image(payload.get("image"))
        if min(image.size) < 8:
            raise ValueError("Choose a face crop at least 8 pixels wide and high.")
        mode = payload.get("mode", "native")
        if mode not in {"native", "simulate"}:
            raise ValueError("Choose a valid input mode.")
        name = str(payload.get("name", "Your image"))[:100]
        if mode == "simulate":
            if min(image.size) < 112:
                raise ValueError(
                    "Simulation needs a clear photo at least 112 pixels in both dimensions. Use native mode for an already small image."
                )
            original = ImageOps.fit(image, (112, 112), method=Image.Resampling.BICUBIC)
            profile = payload.get("profile", "standard28")
            if not isinstance(profile, str) or profile not in PROFILE_LABELS:
                raise ValueError("Choose a valid degradation profile.")
            low = InputDegradation(service.n, profile)(
                np.array(original), np.random.default_rng(42)
            )
            return service.compare(
                Image.fromarray(low),
                original,
                name,
                f"Simulation: 112 × 112 original to 28 × 28 input | {PROFILE_LABELS[profile]}. This input quality is chosen for the experiment; it is not measured from a real camera.",
                model_id=model_id,
                strength=strength,
                inference=inference,
                known_clean_bicubic=(profile == "clean28"),
            )
        reference = decode_image(payload["reference"]) if payload.get("reference") else None
        return service.compare(
            image, reference, name, model_id=model_id, strength=strength, inference=inference
        )
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        log.exception("Image reconstruction failed")
        raise HTTPException(
            500, "Reconstruction failed. Try a smaller cropped face image."
        ) from error


@app.post("/api/shutdown")
def shutdown(payload: dict):
    if SERVER is not None:
        threading.Timer(0.2, setattr, args=(SERVER, "should_exit", True)).start()
    return {"stopping": True}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", media_type="text/html")


@app.get("/diagnostics/visual")
def visual_diagnostic():
    report = ROOT / "work/visual_tradeoff/case_studies.html"
    if not report.is_file():
        raise HTTPException(
            404, "Run check_visual_tradeoff.py to prepare the validation case studies."
        )
    return FileResponse(report, media_type="text/html")


@app.get("/diagnostics/passes")
def pass_diagnostic():
    report = ROOT / "work/restoration_passes_v1/comparison.html"
    if not report.is_file():
        raise HTTPException(404, "The multi-pass report is not available.")
    return FileResponse(report, media_type="text/html")


@app.get("/assets/{name}")
def asset(name: str):
    allowed = {"app.js": "application/javascript", "styles.css": "text/css"}
    if name not in allowed:
        raise HTTPException(404)
    return FileResponse(WEB / name, media_type=allowed[name])


# %% Command-line launcher
def existing_app(url):
    try:
        with urllib.request.urlopen(url + "/api/status", timeout=2) as response:
            return json.load(response).get("app_id") == APP_ID
    except Exception:
        return False


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
            request = urllib.request.Request(
                url + "/api/shutdown", data=b"{}", headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=3):
                pass
            print("Comparison app stopped.")
        else:
            print("Comparison app is already running:", url)
            if args.open_browser:
                webbrowser.open(url)
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
                if existing_app(url):
                    webbrowser.open(url)
                    return
                time.sleep(0.25)

        threading.Thread(target=open_when_ready, daemon=True).start()
    SERVER = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    SERVER.run()


if __name__ == "__main__":
    main()

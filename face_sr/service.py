"""Load checkpoints, prepare face crops and run paired or native comparisons."""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

from input_quality import InputDegradation, PROFILE_LABELS
from .images import png_url
from .paths import ROOT, V4_REVIEW, VISUAL_CASES, CLEAN_CONSISTENCY_REVIEW, APP_ID

log = logging.getLogger("comparison")


class ReconstructionService:
    def __init__(self):
        self.state = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.sample_lock = threading.Lock()
        self.cache = {}
        self.test_ids = []
        self.validation_ids = []
        self.qmul_examples = []
        self.faces = None
        self.default_model = "identity_v4"
        self.clean_consistency_available = False

    # %% Load and verify the three checkpoints
    def load(self):
        try:
            from . import runtime

            self.n = runtime
            torch = self.n.torch
            torch.set_num_threads(2)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cudnn.benchmark = False
            # One selected project checkpoint; historical experiments remain on disk.
            review = json.loads(V4_REVIEW.read_text(encoding="utf-8"))
            if (
                not review.get("available_for_comparison")
                or not review.get("checks")
                or not all(review["checks"].values())
            ):
                raise ValueError("The selected project checkpoint is not available")
            path = (ROOT / review["checkpoint"]).resolve()
            path.relative_to((ROOT / "work/recognition_v4").resolve())
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != review["checkpoint_sha256"]:
                raise ValueError("Project checkpoint checksum mismatch")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["architecture"] != {"use_ssc": True, "use_attn": True}:
                raise ValueError("Selected model must use both SSC and attention")
            allowed = self.n.Config.__dataclass_fields__
            self.cfg = self.n.Config(
                **{k: v for k, v in checkpoint.get("config", {}).items() if k in allowed}
            )
            self.model = self.n.SRNet(self.cfg)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.to(self.device).eval().requires_grad_(False)
            self.epoch = int(checkpoint["epoch"])
            del checkpoint
            self.default_model = "identity_v4"
            self.models = {"identity_v4": self.model}
            self.model_info = {
                "identity_v4": {
                    "id": "identity_v4",
                    "label": "Ours - lightweight SSC + attention (v4)",
                    "epochs": self.epoch,
                    "finetune_epochs": self.epoch - 30,
                    "kind": "custom",
                    "parameters": self.n.n_params(self.model),
                    "checkpoint_sha256": digest,
                    "training_domains": [
                        "FFHQ project training split",
                        "Official QMUL training identities",
                    ],
                    "description": "Your selected v4 model: sparse-sampling convolution and adaptive sparse linear attention.",
                    "warning": "Eye and spectacle detail remains imperfect. FFHQ has clean comparison targets; QMUL has no paired clean originals. Recognizer accuracy is not a restoration percentage.",
                }
            }
            if CLEAN_CONSISTENCY_REVIEW.is_file():
                correction_review = json.loads(CLEAN_CONSISTENCY_REVIEW.read_text(encoding="utf-8"))
                source_digest = hashlib.sha256(
                    (ROOT / "restoration_passes.py").read_bytes()
                ).hexdigest()
                self.clean_consistency_available = bool(
                    correction_review.get("available_as_clean_only_experiment")
                    and correction_review.get("validation_selected")
                    and correction_review.get("checkpoint_sha256")
                    == self.model_info["identity_v4"]["checkpoint_sha256"]
                    and correction_review.get("inference_source_sha256") == source_digest
                )
            from restoration_model import FaceRestoration
            from codeformer_model import CodeFormerRestoration

            baselines = [
                (
                    "restoration",
                    "GFPGAN 1.4 (official pretrained)",
                    FaceRestoration,
                    ROOT / "work/restoration_assets/provenance.json",
                    "GFPGANv1.4.pth",
                    "Official pretrained GFPGAN 1.4, full model output without blending.",
                ),
                (
                    "codeformer",
                    "CodeFormer (official pretrained, w=0.5)",
                    lambda device: CodeFormerRestoration(device, fidelity=0.5),
                    ROOT / "work/codeformer_assets/provenance.json",
                    "codeformer.pth",
                    "Official pretrained CodeFormer, documented cropped-face setting w=0.5, without blending.",
                ),
            ]
            for model_id, label, factory, provenance_path, filename, description in baselines:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                weights = provenance_path.parent / filename
                with weights.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != provenance["files"][filename]["sha256"]:
                    raise ValueError(f"{model_id} official weights checksum mismatch")
                model = factory(self.device)
                self.models[model_id] = model
                self.model_info[model_id] = {
                    "id": model_id,
                    "label": label,
                    "epochs": None,
                    "finetune_epochs": 0,
                    "kind": "external_pretrained",
                    "parameters": self.n.n_params(model),
                    "checkpoint_sha256": digest,
                    "training_domains": [
                        "FFHQ (official pretraining; different split and procedure)"
                    ],
                    "fidelity": 0.5 if model_id == "codeformer" else None,
                    "description": description,
                    "warning": "Shared FFHQ dataset, different training splits and budget; external pretraining overlap is not excluded. Both baselines internally resize the same supplied crop to 512 x 512, then return x4 at its original aspect ratio. No detector or extra alignment is applied. Generated details may change identity.",
                }
            with torch.inference_mode():
                for model in self.models.values():
                    model(torch.zeros(1, 3, 28, 28, device=self.device))
                if self.device == "cuda":
                    torch.cuda.synchronize()
            self.hardware = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
            data, splits = ROOT / "work/data/ffhq128_u8.npy", ROOT / "work/full/split_ids.npz"
            if data.exists() and splits.exists():
                self.faces = np.load(data, mmap_mode="r")
                with np.load(splits) as z:
                    self.test_ids = z["test"].tolist()
                    self.validation_ids = z["validation"].tolist()
            demo = ROOT / "work/identity_investigation/mixed_quality_v1/qmul_examples.json"
            qmul_root = (
                ROOT
                / "work/recognition/qmul_data/QMUL-SurvFace/Face_Verification_Test_Set/verification_images"
            )
            if demo.exists():
                names = json.loads(demo.read_text(encoding="utf-8")).get("files", [])
                self.qmul_examples = [
                    qmul_root / name
                    for name in names
                    if isinstance(name, str)
                    and Path(name).name == name
                    and (qmul_root / name).is_file()
                ]
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
        return {
            "app_id": APP_ID,
            "state": self.state,
            "error": self.error,
            "hardware": getattr(self, "hardware", None),
            "scale": 4,
            "epochs": getattr(self, "epoch", None),
            "samples_available": bool(self.test_ids or self.qmul_examples),
            "available_sample_splits": (["test", "validation"] if self.test_ids else [])
            + (["qmul"] if self.qmul_examples else []),
            "models": list(getattr(self, "model_info", {}).values()),
            "default_model": self.default_model,
            "clean_consistency_available": self.clean_consistency_available,
            "default_gallery": (
                {"split": "validation", "profile": "standard28", "focus": "cases"}
                if self.validation_ids and VISUAL_CASES.is_file()
                else {"split": "test", "profile": "standard28", "focus": "all"}
            ),
            "qmul_examples_available": bool(self.qmul_examples),
            "degradation_profiles": PROFILE_LABELS,
            "dataset_usage": {
                "training": "FFHQ: 67,000 images",
                "validation": "FFHQ: 1,000 images",
                "test": "FFHQ: 2,000 images",
                "qmul": "Ours uses official QMUL training identities, excluding test identities. Pretrained baselines have different training procedures. All receive the same evaluation inputs. No paired clean camera HR target.",
            },
        }

    # %% Prepare identical input pixels for each method
    def pair(self, index, split="test", profile="standard28"):
        self.require_ready()
        if not isinstance(split, str) or split not in {"test", "validation", "qmul"}:
            raise ValueError("Choose test, validation or QMUL examples.")
        if not isinstance(profile, str) or profile not in PROFILE_LABELS:
            raise ValueError("Choose a valid degradation profile.")
        if split == "qmul":
            if not 0 <= index < len(self.qmul_examples):
                raise ValueError("This QMUL example is not available.")
            with Image.open(self.qmul_examples[index]) as image:
                low = image.convert("RGB").copy()
            return low, None
        ids = self.test_ids if split == "test" else self.validation_ids
        if not ids or not 0 <= index < len(ids):
            raise ValueError("This paired example is not available.")
        original = np.array(self.faces[ids[index], 8:120, 8:120], copy=True)
        degraded = InputDegradation(self.n, profile)(
            original, np.random.default_rng((2000 if split == "test" else 1000) + index)
        )
        return Image.fromarray(degraded), Image.fromarray(original)

    def samples(self, split="test", profile="standard28", focus="all"):
        self.require_ready()
        result = []
        if split not in {"test", "validation", "qmul"}:
            raise ValueError("Choose a valid dataset split.")
        if focus not in {"all", "cases"}:
            raise ValueError("Choose all faces or selected validation cases.")
        count = (
            len(self.qmul_examples)
            if split == "qmul"
            else len(self.test_ids if split == "test" else self.validation_ids)
        )
        indices = list(range(min(12, count)))
        case_labels = {}
        if focus == "cases":
            if split != "validation" or profile != "standard28":
                raise ValueError(
                    "Selected validation case studies use validation images and benchmark degradation."
                )
            manifest = json.loads(VISUAL_CASES.read_text(encoding="utf-8"))
            if (
                not manifest.get("post_hoc_selection")
                or manifest["split"] != split
                or manifest["profile"] != profile
            ):
                raise ValueError("Validation case-study metadata does not match this condition.")
            indices = [case["index"] for case in manifest["cases"]]
            if any(type(i) is not int or not 0 <= i < count for i in indices):
                raise ValueError("Invalid validation case-study index.")
            case_labels = {case["index"]: case["role"] for case in manifest["cases"]}
        for i in indices:
            low, original = self.pair(i, split, profile)
            name = (
                f"QMUL camera crop {i + 1:02}"
                if split == "qmul"
                else f"{split.title()} face {i + 1:02}"
            )
            result.append(
                {
                    "id": i,
                    "name": name,
                    "thumbnail": png_url(original if original is not None else low),
                    "input_size": list(low.size),
                    "case_study": case_labels.get(i),
                }
            )
        return result

    def tensor(self, image):
        return (
            self.n.torch.from_numpy(np.array(image, copy=True))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .div(255)
            .contiguous()
        )

    def to_image(self, tensor):
        array = tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
        return Image.fromarray((array * 255).round().astype(np.uint8))

    # %% Run the selected model and score its output
    def compare(
        self,
        image,
        reference=None,
        name="Your image",
        note=None,
        model_id="identity_v4",
        strength=1.0,
        inference="single",
        known_clean_bicubic=False,
    ):
        self.require_ready()
        if model_id not in self.models:
            raise ValueError("This reconstruction model is not available.")
        if inference not in {"single", "clean_consistency"}:
            raise ValueError("Choose an available refinement mode.")
        if inference == "clean_consistency" and not (
            self.clean_consistency_available and model_id == "identity_v4" and known_clean_bicubic
        ):
            raise ValueError(
                "Clean-input consistency is available only for our model on known clean bicubic paired examples or simulations. It is disabled for native uploads and QMUL."
            )
        if min(image.size) < 8:
            raise ValueError("Choose a face crop at least 8 pixels wide and high.")
        if model_id in {"restoration", "codeformer"} and strength != 1:
            raise ValueError("Pretrained comparisons use full model output; blending is disabled.")
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
            raise ValueError(
                f"The clean original must be {output_size[0]} × {output_size[1]} pixels to match this input at ×4. It was not resized automatically."
            )
        torch = self.n.torch
        with self.lock, torch.inference_mode():
            low = self.tensor(image).to(self.device)
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if inference == "clean_consistency":
                from restoration_passes import predict

                reconstructed = predict(self.model, low, "backprojection3")
            else:
                reconstructed = self.models[model_id](low).clamp(0, 1)
            if not torch.isfinite(reconstructed).all():
                raise ValueError("The model produced non-finite output for this input.")
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # Comparison preview is excluded from the selected-model inference timer.
            previous_gpu = (
                self.model(low).clamp(0, 1)
                if model_id != "identity_v4" or inference != "single"
                else None
            )
            sr = reconstructed.cpu()
            bicubic = (
                self.n.F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False)
                .clamp(0, 1)
                .cpu()
            )
            previous = previous_gpu.cpu() if previous_gpu is not None else None
        metrics = None
        if reference is not None:
            target = self.tensor(reference)
            metrics = {}
            for key, value in [("sr", sr), ("bicubic", bicubic)]:
                metrics[key] = {
                    "psnr": float(self.n.Metrics.psnr_y(value, target, 4).item()),
                    "ssim": float(self.n.Metrics.ssim_y(value, target, 4).item()),
                }
        return {
            "name": name,
            "input": png_url(image),
            "sr": png_url(self.to_image(sr)),
            "bicubic": png_url(self.to_image(bicubic)),
            "reference": png_url(reference) if reference is not None else None,
            "input_size": list(image.size),
            "output_size": list(output_size),
            "metrics": metrics,
            "processing_ms": round(elapsed * 1000, 1),
            "note": note,
            "previous": png_url(self.to_image(previous)) if previous is not None else None,
            "previous_label": "SINGLE-PASS V4" if inference != "single" else "OUR MODEL (V4)",
            "model_id": model_id,
            "model_info": self.model_info[model_id],
            "inference_mode": inference,
            "neural_passes": 1,
            "correction_iterations": 3 if inference == "clean_consistency" else 0,
            "strength": strength if model_id in {"restoration", "codeformer"} else None,
            "input_guidance": (
                "This small input contains limited facial detail. Prefer a larger native face crop or additional video frames; enlarging the same pixels cannot restore missing identity evidence."
                if min(image.size) <= 32
                else "Native input detail is retained. Avoid unnecessary downsampling or repeated JPEG compression."
            ),
            "filename": "reconstructed_identity_v4_clean_consistency_x4.png"
            if inference == "clean_consistency"
            else {
                "identity_v4": "reconstructed_identity_v4_x4.png",
                "restoration": "gfpgan_restored_x4.png",
                "codeformer": "codeformer_w05_restored_x4.png",
            }[model_id],
        }

    # %% Cache a result by input condition and model
    def sample_result(
        self,
        index,
        model_id="identity_v4",
        strength=1.0,
        split="test",
        profile="standard28",
        inference="single",
    ):
        if not isinstance(split, str) or split not in {"test", "validation", "qmul"}:
            raise ValueError("Choose test, validation or QMUL examples.")
        if not isinstance(profile, str) or profile not in PROFILE_LABELS:
            raise ValueError("Choose a valid degradation profile.")
        with self.sample_lock:
            key = (
                model_id,
                index,
                strength if model_id in {"restoration", "codeformer"} else None,
                split,
                profile,
                inference,
            )
            if key not in self.cache:
                low, original = self.pair(index, split, profile)
                note = (
                    f"FFHQ {split} example | {PROFILE_LABELS[profile]}. "
                    + (
                        "Validation is used for model selection. "
                        if split == "validation"
                        else "Held out from custom-model training; repeatedly evaluated. "
                    )
                    + "Every model receives identical input pixels. The clean original is used only for comparison."
                    + (
                        " External pretraining overlap is not excluded."
                        if model_id in {"restoration", "codeformer"}
                        else ""
                    )
                )
                name = f"{split.title()} face {index + 1:02}"
                if split == "qmul":
                    name = f"QMUL camera crop {index + 1:02}"
                    note = (
                        f"Real QMUL-SurvFace crop: {self.qmul_examples[index].name} | Native {low.width} x {low.height} pixels, no extra degradation. "
                        "No paired clean original exists for this crop, so PSNR/SSIM and a restoration percentage cannot be measured. "
                        "QMUL verification accuracy is evaluated over labelled same/different-person pairs, not on this individual image."
                    )
                if inference == "clean_consistency":
                    note += " Experimental clean-input consistency: one neural pass plus three residual corrections; not three neural passes. This setting is not validated for noisy/native camera inputs."
                self.cache[key] = self.compare(
                    low,
                    original,
                    name,
                    note,
                    model_id=model_id,
                    strength=strength,
                    inference=inference,
                    known_clean_bicubic=(split != "qmul" and profile == "clean28"),
                )
                self.cache[key]["sample_split"] = split
                self.cache[key]["degradation_profile"] = "native" if split == "qmul" else profile
                if len(self.cache) > 48:
                    self.cache.pop(next(iter(self.cache)))
            return self.cache[key]

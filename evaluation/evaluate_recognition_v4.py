"""Frozen v4 evaluation: unchanged primary calibration, explicit fusion diagnostics."""

import io
import json
import zipfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from audit_restoration_leakage import exact_key
from evaluate_restoration_v3 import embeddings, extra_pairs, metrics
from input_quality import InputDegradation
from investigate_identity import load, save_grid
from prepare_recognition_v4 import BASE, OUT
from quality_metric import load_lpips, load_lpips_alex
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, notebook_definitions, runtime


# %% Convert embeddings to verification scores
def pair_scores(bank, pairs):
    return {key: (value[pairs["ia"]] * value[pairs["ib"]]).sum(1) for key, value in bank.items()}


def fuse(scores, parameters):
    return (
        sum(
            (scores[key] - parameters[key]["negative_mean"]) / parameters[key]["negative_std"]
            for key in ["arcface", "mobileface"]
        )
        / 2
    )


def fit_calibration(r, scores, labels):
    parameters = {
        key: {
            "negative_mean": float(value[labels == 0].mean()),
            "negative_std": float(value[labels == 0].std()),
        }
        for key, value in scores.items()
    }
    assert all(value["negative_std"] > 1e-6 for value in parameters.values())
    fused = fuse(scores, parameters)
    thresholds = {
        key: float(r.VerificationEval.best_threshold(value, labels))
        for key, value in {**scores, "fusion": fused}.items()
    }
    return {
        "fusion_parameters": parameters,
        "thresholds": thresholds,
        "validation_pairs": len(labels),
        "validation_auc": {
            key: float(roc_auc_score(labels, value))
            for key, value in {**scores, "fusion": fused}.items()
        },
    }


@torch.inference_mode()
# %% Freeze validation thresholds before scoring the evaluation pairs
def main():
    if (OUT / "evaluation.json").exists():
        raise ValueError("Preserve evaluation; no retuning on its test results")
    protocol = json.loads((OUT / "protocol.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    assert sha(OUT / "cache/validation28.npy") == protocol["validation28_sha256"]
    assert sha(OUT / "cache/calibration_pairs.npz") == protocol["pairs_sha256"]
    paths = {"restoration_v3": BASE}
    if selection["selected_new_checkpoint"]:
        paths["identity_v4"] = ROOT / selection["checkpoint"]
    frozen = {
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "selection_sha256": sha(OUT / "selection.json"),
        "checkpoint_sha256": {name: sha(path) for name, path in paths.items()},
        "primary_calibration": "Original512 crops, original256 positive plus256 negative pairs from held-out official-training identities; unchanged from v3.",
        "expanded_calibration": "Separate secondary pipeline: all11135 held-out-training crops, frozen4968 balanced validation pairs.",
        "fusion": "Fixed equal average of negative-validation-standardized ArcFace and MobileFace scores; separate recognition pipeline, unchanged reconstruction pixels.",
        "test_does_not_select_model_or_availability": True,
        "historical_test_reuse": True,
    }
    write_json(OUT / "frozen_evaluation.json", frozen)
    (OUT / "emb").mkdir(exist_ok=True)
    n, _ = runtime()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    r = notebook_definitions(
        "sr_part2a_recognition.ipynb", [1, 6, 7, 8, 9, 11], "v4_validation_protocol"
    )
    faces = {
        key: FrozenFace(ROOT / path).to(n.device).eval().requires_grad_(False)
        for key, path in {
            "arcface": "work/recognition/w600k_r50.onnx",
            "mobileface": "work/refinement_assets/w600k_mbf.onnx",
        }.items()
    }
    validation = np.load(OUT / "cache/validation28.npy", mmap_mode="r")
    with np.load(OUT / "cache/calibration_pairs.npz") as data:
        calibration_pairs = {
            name: {key: data[name + "_" + key] for key in ["ia", "ib", "labels"]}
            for name in ["primary", "expanded"]
        }
    calibrations = {}
    for name, path in paths.items():
        model = load(n, path)
        bank, _ = embeddings(n, model, validation, faces)
        calibrations[name] = {}
        for key, pairs in calibration_pairs.items():
            scores = pair_scores(bank, pairs)
            np.savez(
                OUT / "emb" / f"{name}_validation_{key}.npz",
                labels=pairs["labels"],
                **{net + "_scores": value for net, value in scores.items()},
            )
            calibrations[name][key] = fit_calibration(r, scores, pairs["labels"])
        del model, bank
        torch.cuda.empty_cache()
        print("V4 CALIBRATION", name, json.dumps(calibrations[name]), flush=True)
    # Thresholds and fusion statistics are frozen BEFORE any evaluation score load.
    write_json(OUT / "calibration.json", calibrations)
    result = {
        "frozen_protocol": frozen,
        "calibration_sha256": sha(OUT / "calibration.json"),
        "recognition": {},
        "paired_ffhq": {},
    }
    original = np.load(ROOT / "work/qmul_study/cache/lr28.npy", mmap_mode="r")
    with np.load(ROOT / "work/qmul_study/cache/pairs.npz") as data:
        original_pairs = {key: data[key] for key in ["ia", "ib", "labels"]}
    domain = json.loads((ROOT / "work/qmul_domain/data_manifest.json").read_text())
    additional = json.loads((ROOT / "work/leakage_review_v2/additional_cohort.json").read_text())
    more = []
    hashes = []
    with zipfile.ZipFile(ROOT / domain["source_archive"]) as zipped:
        for path in additional["files"]:
            with Image.open(io.BytesIO(zipped.read(path))) as image:
                rgb = image.convert("RGB")
                hashes.append(exact_key(np.asarray(rgb)))
                more.append(np.asarray(rgb.resize((28, 28), Image.Resampling.BICUBIC)))
    more = np.stack(more)
    more_pairs = extra_pairs(additional["files"], hashes)
    result.update(
        additional_images=len(more),
        additional_identities=len(additional["identities"]),
        additional_pairs=len(more_pairs["labels"]),
    )
    rows = {
        "Actual input (28)": torch.from_numpy(more[np.linspace(0, len(more) - 1, 6, dtype=int)])
        .permute(0, 3, 1, 2)
        .float()
        / 255
    }
    for name, path in paths.items():
        result["recognition"][name] = {}
        model = None if name == "restoration_v3" else load(n, path)
        for dataset, images, pairs in [
            ("prior_9913_images", original, original_pairs),
            ("additional_identity_disjoint", more, more_pairs),
        ]:
            if name == "restoration_v3":
                source = ROOT / "work/restoration_v3/emb" / f"restoration_v3_{dataset}.npz"
                with np.load(source) as data:
                    assert np.array_equal(data["labels"], pairs["labels"])
                    scores = {key: data[key + "_scores"] for key in faces}
                result.setdefault("reused_v3_pair_score_sha256", {})[dataset] = sha(source)
            else:
                bank, examples = embeddings(n, model, images, faces)
                scores = pair_scores(bank, pairs)
                if dataset == "additional_identity_disjoint":
                    rows[name] = examples
                del bank
            result["recognition"][name][dataset] = {}
            raw = {
                "labels": pairs["labels"],
                **{key + "_scores": value for key, value in scores.items()},
            }
            if "anchor_id" in pairs:
                raw["anchor_id"] = pairs["anchor_id"]
            for calibration, settings in calibrations[name].items():
                fused = fuse(scores, settings["fusion_parameters"])
                result["recognition"][name][dataset][calibration] = {
                    key: metrics(value, pairs["labels"], settings["thresholds"][key])
                    for key, value in {**scores, "fusion": fused}.items()
                }
                raw[calibration + "_fusion_scores"] = fused
            np.savez(OUT / "emb" / f"{name}_{dataset}.npz", **raw)
        print("V4 RECOGNITION", name, json.dumps(result["recognition"][name]), flush=True)
        del model
        torch.cuda.empty_cache()
    del faces
    torch.cuda.empty_cache()
    models = {name: load(n, path) for name, path in paths.items()}
    base_examples = (
        models["restoration_v3"](rows["Actual input (28)"].to(n.device)).clamp(0, 1).cpu()
    )
    save_grid(
        {
            **dict(list(rows.items())[:1]),
            "restoration_v3": base_examples,
            **dict(list(rows.items())[1:]),
        },
        OUT / "qmul_comparison.png",
        "Same additional QMUL crops | fusion changes no pixels",
    )
    lpips = load_lpips(n.device)
    alex = load_lpips_alex(n.device)
    hr = np.load(ROOT / "work/data/ffhq128_u8.npy", mmap_mode="r")
    with np.load(ROOT / "work/full/split_ids.npz") as data:
        test = data["test"][:512]
    for profile in ["standard28", "clean28"]:
        dataset = n.FixedEvalSet(hr, test, InputDegradation(n, profile), 112, 2000)
        scores = {
            name: {key: [] for key in ["psnr_y", "ssim_y", "lpips_squeeze", "lpips_alex"]}
            for name in models
        }
        rows = None
        for low, target in dataset.batches(32):
            low, target = low.to(n.device), target.to(n.device)
            views = {name: model(low).clamp(0, 1) for name, model in models.items()}
            for name, sr in views.items():
                scores[name]["psnr_y"].extend(n.Metrics.psnr_y(sr, target, 4).cpu().tolist())
                scores[name]["ssim_y"].extend(n.Metrics.ssim_y(sr.cpu(), target.cpu(), 4).tolist())
                scores[name]["lpips_squeeze"].extend(
                    lpips(sr, target, normalize=True).flatten().cpu().tolist()
                )
                scores[name]["lpips_alex"].extend(
                    alex(sr, target, normalize=True).flatten().cpu().tolist()
                )
            if rows is None:
                rows = {
                    "Clean original": target.cpu(),
                    "Actual input (28)": low.cpu(),
                    **{name: value.cpu() for name, value in views.items()},
                }
        result["paired_ffhq"][profile] = {
            name: {key: float(np.mean(value)) for key, value in values.items()}
            for name, values in scores.items()
        }
        write_json(OUT / (profile + "_per_image.json"), scores)
        save_grid(
            rows,
            OUT / (profile + "_comparison.png"),
            "Same512 held-out FFHQ test images | " + profile,
        )
        print("V4 PAIRED", profile, json.dumps(result["paired_ffhq"][profile]), flush=True)
    result["primary_target_over70"] = bool(
        selection["selected_new_checkpoint"]
        and result["recognition"]["identity_v4"]["prior_9913_images"]["primary"]["mobileface"][
            "accuracy"
        ]
        > 0.70
    )
    result["fusion_over70"] = {
        name: values["prior_9913_images"]["primary"]["fusion"]["accuracy"] > 0.70
        for name, values in result["recognition"].items()
    }
    write_json(OUT / "evaluation.json", result)
    print("V4 EVALUATION COMPLETE primary70:", result["primary_target_over70"], flush=True)


if __name__ == "__main__":
    main()

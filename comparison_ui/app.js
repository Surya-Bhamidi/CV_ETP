"use strict";
const $ = (id) => document.getElementById(id);
// The viewer keeps one selected input while the user switches between models.
const state = {
  ready: false,
  busy: false,
  result: null,
  source: "samples",
  layout: "slider",
  compare: "input",
  split: 50,
  zoom: 1,
  panX: 0,
  panY: 0,
  upload: null,
  reference: null,
  sample: null,
  model: "identity_v4",
  models: [],
  lastRequest: null,
  sampleSplit: "test",
  profile: "standard28",
  galleryFocus: "all",
  inference: "single",
  cleanConsistencyAvailable: false,
};
const icon = (id) => `<svg aria-hidden="true"><use href="#${id}"/></svg>`;
const sizeText = (size) => `${size[0]} × ${size[1]}`;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("The local server returned an unexpected response.");
  }
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "This image could not be processed.",
    );
  return body;
}
function showError(message) {
  $("error-message").textContent = message;
  $("error-message").hidden = false;
}
function clearError() {
  $("error-message").hidden = true;
  $("error-message").textContent = "";
}
function updateInferenceControl() {
  const available =
    state.cleanConsistencyAvailable &&
    state.model === "identity_v4" &&
    state.profile === "clean28" &&
    ((state.source === "samples" && state.sampleSplit !== "qmul") ||
      (state.source === "upload" && $("input-mode").value === "simulate"));
  if (!available) state.inference = "single";
  $("inference-mode").value = state.inference;
  $("inference-mode").disabled = state.busy || !state.ready || !available;
  $("inference-help").textContent = available
    ? "Experimental: one neural pass plus three input-consistency corrections. Small measured gains on clean bicubic inputs; inspect against Original and Single-pass v4."
    : "Clean-input correction requires our model and downsampling-only paired/simulated input. Native or noisy inputs retain single-pass inference.";
}
function updateButtons() {
  updateInferenceControl();
  $("reconstruct").disabled = !state.ready || state.busy || !state.upload;
  document
    .querySelectorAll(".sample-button")
    .forEach((button) => (button.disabled = state.busy));
  $("input-mode").disabled = state.busy;
  $("drop-zone").disabled = state.busy;
  $("model-choice").disabled = state.busy || !state.ready;
  $("quality-profile").disabled = state.busy || !state.ready;
  $("sample-split").disabled = state.busy || !state.ready;
  $("validation-cases").disabled =
    state.busy || !state.ready || !state.casesAvailable;
}
function setBusy(busy, label = "Reconstructing your image…") {
  state.busy = busy;
  $("loading-overlay").hidden = !busy;
  $("loading-label").textContent = label;
  $("loading-hint").textContent = state.ready
    ? "Running your fully trained model locally."
    : "The comparison will appear here.";
  $("reconstruct").querySelector("span").textContent = busy
    ? "Reconstructing…"
    : "Reconstruct image";
  updateButtons();
}
function setSource(source) {
  state.source = source;
  for (const name of ["samples", "upload"]) {
    const active = source === name;
    $(`${name}-tab`).classList.toggle("active", active);
    $(`${name}-tab`).setAttribute("aria-selected", String(active));
    $(`${name}-panel`).hidden = !active;
  }
  clearError();
  updateMode();
}
function setSplit(value) {
  state.split = Math.min(100, Math.max(0, Number(value)));
  $("before-layer").style.clipPath = `inset(0 ${100 - state.split}% 0 0)`;
  $("compare-divider").style.left = `${state.split}%`;
  $("split-range").value = state.split;
  $("split-range").setAttribute(
    "aria-valuetext",
    `${Math.round(state.split)} percent comparison image, ${Math.round(100 - state.split)} percent reconstructed image`,
  );
}
function updateTransform() {
  $("image-stage").style.setProperty(
    "--image-transform",
    `translate(${state.panX}px,${state.panY}px) scale(${state.zoom})`,
  );
  $("image-stage").classList.toggle("zoomed", state.zoom > 1);
  $("zoom-level").textContent =
    state.zoom === 1 ? "Fit" : `${Math.round(state.zoom * 100)}%`;
  $("zoom-out").disabled = state.zoom <= 1;
  $("zoom-in").disabled = state.zoom >= 4;
  $("drag-hint").textContent =
    state.zoom > 1 ? "Drag the image to pan" : "Drag the divider to compare";
}
function resetView() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  updateTransform();
}
function zoomBy(delta) {
  state.zoom = Math.min(4, Math.max(1, state.zoom + delta));
  if (state.zoom === 1) {
    state.panX = 0;
    state.panY = 0;
  }
  updateTransform();
}
// Choose the two images for the slider or side-by-side view.
function renderComparison() {
  if (!state.result) return;
  const result = state.result;
  if (state.compare === "reference" && !result.reference)
    state.compare = "input";
  if (state.compare === "previous" && !result.previous) state.compare = "input";
  const labels = {
    input: "LOW-RES INPUT",
    bicubic: "BICUBIC ×4",
    reference: "CLEAN ORIGINAL",
    previous: result.previous_label || "ORIGINAL CUSTOM SR",
  };
  const image =
    state.compare === "reference" ? result.reference : result[state.compare];
  $("before-image").src = image;
  $("side-before-image").src = image;
  for (const name of ["before-image", "side-before-image"]) {
    $(name).classList.toggle("pixelated", state.compare === "input");
    $(name).alt = labels[state.compare];
  }
  $("before-tag").textContent = labels[state.compare];
  $("side-before-tag").textContent = labels[state.compare];
  $("slider-view").hidden = state.layout !== "slider";
  $("side-view").hidden = state.layout !== "side";
  document.querySelectorAll("[data-compare]").forEach((button) => {
    const active = button.dataset.compare === state.compare;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll("[data-layout]").forEach((button) => {
    const active = button.dataset.layout === state.layout;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.compare === "reference")
    $("result-note").textContent =
      `Clean reference view: the model received the ${sizeText(result.input_size)} low-res input, not this original image. ${result.note || ""}`;
  else
    $("result-note").textContent =
      result.note ||
      (result.reference
        ? "Switch to Original to inspect how closely the reconstruction matches the clean face."
        : "This upload has no clean reference. Compare visually, or add a matching original to measure PSNR and SSIM.");
  if (state.galleryFocus === "cases" && state.source === "samples")
    $("result-note").textContent +=
      " Selected after validation measurement: illustrative case study, not an unbiased test example.";
  setSplit(state.split);
}
function metric(name, number, unit, difference, digits) {
  $(name + "-value").replaceChildren(
    document.createTextNode(number.toFixed(digits)),
  );
  if (unit) {
    const small = document.createElement("small");
    small.textContent = ` ${unit}`;
    $(name + "-value").append(small);
  }
  const caption = $(name + "-caption");
  caption.classList.remove("positive", "negative");
  caption.classList.add(difference >= 0 ? "positive" : "negative");
  caption.textContent = `${difference >= 0 ? "+" : ""}${difference.toFixed(digits)}${unit ? " " + unit : ""} vs bicubic`;
}
function showResult(result) {
  state.result = result;
  $("empty-state").hidden = true;
  $("after-image").src = result.sr;
  $("side-after-image").src = result.sr;
  $("input-resolution").innerHTML =
    `${sizeText(result.input_size)}<small> px</small>`;
  $("output-resolution").innerHTML =
    `${sizeText(result.output_size)}<small> px</small>`;
  $("original-option").disabled = !result.reference;
  $("original-option").title = result.reference
    ? "Compare against the clean original"
    : "A clean original is needed";
  $("previous-option").disabled = !result.previous;
  $("previous-option").textContent =
    result.inference_mode === "clean_consistency"
      ? "Single-pass v4"
      : "Our model";
  $("previous-option").title =
    "Compare with single-pass v4 on identical input pixels";
  $("previous-option").hidden = !result.previous;
  if (result.model_info) {
    $("model-description").textContent = result.model_info.description;
    const external = result.model_info.kind === "external_pretrained";
    $("epoch-count").textContent = external
      ? "externally pretrained"
      : result.model_info.finetune_epochs
        ? `30 + ${result.model_info.finetune_epochs} epochs`
        : `${result.model_info.epochs} epochs`;
    $("selected-model-label").textContent = external
      ? result.model_id === "codeformer"
        ? "CodeFormer pretrained model"
        : "GFPGAN pretrained model"
      : "Your trained SR model";
    const warning =
      result.model_info.warning ||
      (result.model_info.experimental
        ? "Experimental refinement: image structure and CCTV checks regressed."
        : "");
    $("model-warning").textContent = warning;
    $("model-warning").hidden = !warning;
    $("after-tag-text").textContent = external ? "RESTORATION ×4" : "SR ×4";
    $("side-after-tag-text").textContent = external
      ? "RESTORATION ×4"
      : "SR ×4";
    $("after-image").alt =
      `Reconstructed image from ${result.model_info.label}`;
  }
  if (result.metrics) {
    metric(
      "psnr",
      result.metrics.sr.psnr,
      "dB",
      result.metrics.sr.psnr - result.metrics.bicubic.psnr,
      2,
    );
    metric(
      "ssim",
      result.metrics.sr.ssim,
      "",
      result.metrics.sr.ssim - result.metrics.bicubic.ssim,
      4,
    );
  } else {
    $("psnr-value").innerHTML = "—<small> dB</small>";
    $("ssim-value").textContent = "—";
    for (const name of ["psnr", "ssim"]) {
      $(name + "-caption").textContent = "A clean original is needed";
      $(name + "-caption").classList.remove("positive", "negative");
    }
  }
  $("result-name").textContent = result.name;
  $("input-guidance").textContent = result.input_guidance || "";
  $("runtime-label").textContent =
    `${result.processing_ms < 1000 ? Math.round(result.processing_ms) + " ms" : (result.processing_ms / 1000).toFixed(2) + " s"} selected-model inference · processed locally`;
  $("result-note").textContent =
    result.note ||
    (result.reference
      ? "Switch to Original to inspect how closely the reconstruction matches the clean face."
      : "This upload has no clean reference. Compare visually, or add a matching original to measure PSNR and SSIM.");
  $("download").href = result.sr;
  $("download").download = result.filename;
  $("download").classList.remove("disabled");
  $("download").setAttribute("aria-disabled", "false");
  resetView();
  setSplit(50);
  renderComparison();
}
// All three models use the same API. The server returns the input, prediction and available comparisons.
async function reconstruct(payload, name) {
  if (state.busy || !state.ready) return;
  clearError();
  setBusy(true, `Reconstructing ${name || "your image"}…`);
  try {
    const result = await api("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        model: state.model,
        inference: state.inference,
      }),
    });
    state.lastRequest = { ...payload };
    showResult(result);
  } catch (error) {
    showError(error.message);
    if (state.result?.model_id) {
      state.model = state.result.model_id;
      $("model-choice").value = state.model;
    }
  } finally {
    setBusy(false);
  }
}
async function chooseSample(sample) {
  if (state.busy) return;
  state.sample = sample.id;
  document.querySelectorAll(".sample-button").forEach((button) => {
    const selected = Number(button.dataset.id) === sample.id;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  $("selected-name").textContent = sample.name;
  $("selected-detail").textContent =
    `${sizeText(sample.input_size)} input → ${sizeText(sample.input_size.map((value) => value * 4))} output`;
  await reconstruct(
    {
      sample_id: sample.id,
      sample_split: state.sampleSplit,
      profile: state.profile,
    },
    sample.name.toLowerCase(),
  );
}
// A gallery entry identifies its split, index and degradation profile so its input can be reproduced.
async function loadGallery(index = 0) {
  $("gallery-label").textContent =
    state.galleryFocus === "cases"
      ? "SELECTED VALIDATION CASE STUDIES"
      : state.sampleSplit === "qmul"
        ? "REAL SURVEILLANCE CROPS"
        : "PAIRED FFHQ FACES";
  $("gallery-help").textContent =
    state.galleryFocus === "cases"
      ? "Selected after validation measurement: two v4 face-feature cosine strengths and two heavy-model perceptual strengths. These are not training images or unbiased test examples."
      : state.sampleSplit === "qmul"
        ? "Actual camera-derived crops at native resolution. No matching clean high-resolution original is available."
        : "FFHQ examples excluded from custom training; validation is used for model selection.";
  updateMode();
  let response;
  setBusy(true, "Loading paired examples...");
  try {
    response = await api(
      `/api/samples?split=${encodeURIComponent(state.sampleSplit)}&profile=${encodeURIComponent(state.profile)}&focus=${state.sampleSplit === "qmul" ? "all" : state.galleryFocus}`,
    );
  } finally {
    setBusy(false);
  }
  $("sample-grid").replaceChildren();
  $("sample-count").textContent = `${response.samples.length} examples`;
  for (const sample of response.samples) {
    const button = document.createElement("button");
    button.className = "sample-button";
    button.dataset.id = sample.id;
    button.setAttribute("aria-label", `Reconstruct ${sample.name}`);
    button.setAttribute("aria-pressed", "false");
    if (sample.case_study)
      button.title = sample.case_study + " (selected validation illustration)";
    const img = document.createElement("img");
    img.src = sample.thumbnail;
    img.alt = sample.name;
    button.append(img);
    button.addEventListener("click", () => chooseSample(sample));
    $("sample-grid").append(button);
  }
  if (response.samples.length)
    await chooseSample(
      response.samples.find((sample) => sample.id === index) ||
        response.samples[0],
    );
}
function readFile(file) {
  return new Promise((resolve, reject) => {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      reject(new Error("Choose a PNG, JPEG, or WebP image."));
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      reject(new Error("Choose an image smaller than 8 MB."));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("This image could not be opened."));
    reader.readAsDataURL(file);
  });
}
async function uploadFile(file, reference = false) {
  if (!file || state.busy) return;
  clearError();
  try {
    const url = await readFile(file);
    if (reference) {
      state.reference = { url, name: file.name };
      $("reference-name").textContent = file.name;
      $("reference-selection").hidden = false;
      return;
    }
    const dimensions = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve([img.naturalWidth, img.naturalHeight]);
      img.onerror = () => reject(new Error("This image could not be opened."));
      img.src = url;
    });
    state.upload = { url, name: file.name };
    $("uploaded-thumb").src = url;
    $("upload-name").textContent = file.name;
    $("upload-dimensions").textContent = sizeText(dimensions) + " px";
    $("upload-preview").hidden = false;
    updateButtons();
  } catch (error) {
    showError(error.message);
  }
}
function updateMode() {
  updateInferenceControl();
  const simulate = $("input-mode").value === "simulate";
  $("reference-controls").hidden = simulate;
  $("quality-controls").hidden =
    (state.source === "upload" && !simulate) ||
    (state.source === "samples" && state.sampleSplit === "qmul");
  $("mode-help").textContent = simulate
    ? "Creates a 112 × 112 center-fitted original and a degraded 28 × 28 input. Choose a cropped face."
    : "Use a cropped face. Large inputs are fitted within 256 px before processing.";
}

$("samples-tab").addEventListener("click", () => setSource("samples"));
$("upload-tab").addEventListener("click", () => setSource("upload"));
$("drop-zone").addEventListener("click", () => $("image-file").click());
$("image-file").addEventListener("change", (event) =>
  uploadFile(event.target.files[0]),
);
$("reference-file").addEventListener("change", (event) =>
  uploadFile(event.target.files[0], true),
);
$("add-reference").addEventListener("click", () => $("reference-file").click());
$("remove-reference").addEventListener("click", () => {
  state.reference = null;
  $("reference-file").value = "";
  $("reference-selection").hidden = true;
});
$("remove-upload").addEventListener("click", () => {
  state.upload = null;
  $("image-file").value = "";
  $("upload-preview").hidden = true;
  updateButtons();
});
$("input-mode").addEventListener("change", updateMode);
$("validation-cases").addEventListener("click", async () => {
  if (state.busy) return;
  state.galleryFocus = "cases";
  state.sampleSplit = "validation";
  state.profile = "standard28";
  $("sample-split").value = "validation";
  $("quality-profile").value = "standard28";
  try {
    await loadGallery();
  } catch (error) {
    showError(error.message);
  }
});
$("sample-split").addEventListener("change", async (event) => {
  if (state.busy) return;
  state.sampleSplit = event.target.value;
  if (state.galleryFocus === "cases") state.galleryFocus = "all";
  try {
    await loadGallery(state.sample ?? 0);
  } catch (error) {
    showError(error.message);
  }
});
$("quality-profile").addEventListener("change", async (event) => {
  if (state.busy) return;
  state.profile = event.target.value;
  if (state.galleryFocus === "cases") state.galleryFocus = "all";
  if (state.source === "samples") {
    try {
      await loadGallery(state.sample ?? 0);
    } catch (error) {
      showError(error.message);
    }
  } else if (state.lastRequest?.mode === "simulate")
    await reconstruct(
      { ...state.lastRequest, profile: state.profile },
      "your image",
    );
});
$("model-choice").addEventListener("change", async (event) => {
  state.model = event.target.value;
  if (state.lastRequest)
    await reconstruct(state.lastRequest, "the selected image");
});
$("inference-mode").addEventListener("change", async (event) => {
  state.inference = event.target.value;
  if (state.lastRequest)
    await reconstruct(state.lastRequest, "the selected image");
});
for (const eventName of ["dragenter", "dragover"])
  $("drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("drop-zone").classList.add("drag-over");
  });
for (const eventName of ["dragleave", "drop"])
  $("drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("drop-zone").classList.remove("drag-over");
  });
$("drop-zone").addEventListener("drop", (event) =>
  uploadFile(event.dataTransfer.files[0]),
);
$("reconstruct").addEventListener("click", () => {
  if (state.upload)
    reconstruct(
      {
        image: state.upload.url,
        name: state.upload.name,
        mode: $("input-mode").value,
        profile: state.profile,
        reference: state.reference?.url || null,
      },
      "your image",
    );
});
document.querySelectorAll("[data-layout]").forEach((button) =>
  button.addEventListener("click", () => {
    state.layout = button.dataset.layout;
    resetView();
    renderComparison();
  }),
);
document.querySelectorAll("[data-compare]").forEach((button) =>
  button.addEventListener("click", () => {
    state.compare = button.dataset.compare;
    renderComparison();
  }),
);
$("split-range").addEventListener("input", (event) =>
  setSplit(event.target.value),
);
$("zoom-in").addEventListener("click", () => zoomBy(0.5));
$("zoom-out").addEventListener("click", () => zoomBy(-0.5));
$("reset-view").addEventListener("click", resetView);
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("viewer-card").requestFullscreen();
  } catch {
    showError("Full-screen mode is not available in this browser.");
  }
});
let dragging = null;
$("divider-handle").addEventListener("pointerdown", (event) => {
  if (!state.result) return;
  event.preventDefault();
  event.stopPropagation();
  dragging = { type: "split" };
  $("divider-handle").setPointerCapture(event.pointerId);
});
$("image-stage").addEventListener("pointerdown", (event) => {
  if (
    state.busy ||
    !state.result ||
    state.zoom <= 1 ||
    event.target.closest("#divider-handle")
  )
    return;
  event.preventDefault();
  dragging = {
    type: "pan",
    x: event.clientX,
    y: event.clientY,
    px: state.panX,
    py: state.panY,
  };
  $("image-stage").setPointerCapture(event.pointerId);
  $("image-stage").classList.add("panning");
});
window.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  if (dragging.type === "split") {
    const box = $("image-stage").getBoundingClientRect();
    setSplit((100 * (event.clientX - box.left)) / box.width);
  } else {
    const bounds = $("image-stage").getBoundingClientRect();
    const limitX = (bounds.width * (state.zoom - 1)) / 2;
    const limitY = (bounds.height * (state.zoom - 1)) / 2;
    state.panX = Math.max(
      -limitX,
      Math.min(limitX, dragging.px + event.clientX - dragging.x),
    );
    state.panY = Math.max(
      -limitY,
      Math.min(limitY, dragging.py + event.clientY - dragging.y),
    );
    updateTransform();
  }
});
function stopDrag() {
  dragging = null;
  $("image-stage").classList.remove("panning");
}
window.addEventListener("pointerup", stopDrag);
window.addEventListener("pointercancel", stopDrag);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") resetView();
});

// Read checkpoint and gallery availability before enabling the controls.
async function start() {
  $("model-status").classList.add("loading");
  for (let attempt = 0; attempt < 180; attempt++) {
    try {
      const status = await api("/api/status");
      if (status.state === "error") {
        showError(status.error);
        $("status-text").textContent = "Model unavailable";
        $("model-status").className = "model-status error";
        $("loading-overlay").hidden = true;
        return;
      }
      if (status.state === "ready") {
        state.ready = true;
        $("status-text").textContent =
          status.hardware === "CPU" ? "Model ready · CPU" : "Model ready · GPU";
        $("model-status").classList.remove("loading");
        $("epoch-count").textContent = status.epochs;
        state.models = status.models || [];
        state.cleanConsistencyAvailable = Boolean(
          status.clean_consistency_available,
        );
        state.casesAvailable = status.default_gallery?.focus === "cases";
        const query = new URLSearchParams(location.search);
        // A direct launch uses the labeled demo; explicit split/focus/quality links retain their experiment.
        if (
          !query.has("split") &&
          !query.has("focus") &&
          !query.has("profile") &&
          status.default_gallery
        ) {
          state.sampleSplit = status.default_gallery.split;
          state.profile = status.default_gallery.profile;
          state.galleryFocus = status.default_gallery.focus;
        }
        if (["test", "validation", "qmul"].includes(query.get("split")))
          state.sampleSplit = query.get("split");
        const availableSplits = status.available_sample_splits || [
          "test",
          "validation",
        ];
        if (!availableSplits.includes(state.sampleSplit))
          state.sampleSplit = availableSplits[0] || "test";
        for (const option of $("sample-split").options)
          option.disabled = !availableSplits.includes(option.value);
        if (state.sampleSplit === "qmul" && !status.qmul_examples_available)
          state.sampleSplit = "test";
        $("sample-split").querySelector('option[value="qmul"]').disabled =
          !status.qmul_examples_available;
        if (["standard28", "mild28", "clean28"].includes(query.get("profile")))
          state.profile = query.get("profile");
        if (state.casesAvailable && query.get("focus") === "cases") {
          state.galleryFocus = "cases";
          state.sampleSplit = "validation";
          state.profile = "standard28";
        }
        $("sample-split").value = state.sampleSplit;
        $("quality-profile").value = state.profile;
        if (
          ["input", "bicubic", "reference", "previous"].includes(
            query.get("compare"),
          )
        )
          state.compare = query.get("compare");
        let requested =
          query.get("model") || status.default_model || "identity_v4";
        state.model = state.models.some((model) => model.id === requested)
          ? requested
          : status.default_model;
        if (query.get("inference") === "clean_consistency")
          state.inference = "clean_consistency";
        $("model-choice").replaceChildren(
          ...state.models.map((model) => {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.label;
            return option;
          }),
        );
        $("model-choice").value = state.model;
        $("model-description").textContent =
          state.models.find((model) => model.id === state.model)?.description ||
          "";
        $("loading-overlay").hidden = true;
        updateButtons();
        if (status.samples_available) {
          await loadGallery(Number(query.get("sample") || 0));
        } else {
          $("sample-grid").replaceChildren();
          $("sample-count").textContent = "Not available";
          $("sample-description").hidden = true;
          setSource("upload");
        }
        return;
      }
    } catch (error) {
      if (attempt === 179) {
        showError(
          "The local server could not be reached. Restart the comparison app.",
        );
        $("loading-overlay").hidden = true;
        return;
      }
    }
    await wait(1000);
  }
  showError(
    "The model is taking longer than expected to load. Check the server log and refresh.",
  );
  $("loading-overlay").hidden = true;
}
updateTransform();
start();

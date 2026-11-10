"""Browser smoke checks for the local comparison UI (system Microsoft Edge)."""
import base64
import io
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "work/ui"
URL = "http://127.0.0.1:8765"


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checks = {}
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#download').getAttribute('aria-disabled') === 'false'", timeout=120000)
        expect(page.locator(".sample-button")).to_have_count(12)
        expect(page.locator("#input-resolution")).to_contain_text("28")
        expect(page.locator("#output-resolution")).to_contain_text("112")
        checks["real_example_rendered"] = True
        if page.locator("#model-choice option[value='detail']").count():
            page.locator("#model-choice").select_option("standard")
            expect(page.locator("#model-description")).to_contain_text("pixel L1", timeout=120000)
            standard_url = page.locator("#after-image").get_attribute("src")
            page.locator("#model-choice").select_option("detail")
            expect(page.locator("#model-description")).to_contain_text("fine-tune epochs", timeout=120000)
            expect(page.locator("#previous-option")).to_be_enabled()
            expect(page.locator("#model-warning")).to_be_visible()
            assert page.locator("#after-image").get_attribute("src") != standard_url
            page.locator("#previous-option").click()
            previous_url = page.locator("#before-image").get_attribute("src")
            import numpy as np
            a = np.array(Image.open(io.BytesIO(base64.b64decode(standard_url.split(",", 1)[1]))))
            b = np.array(Image.open(io.BytesIO(base64.b64decode(previous_url.split(",", 1)[1]))))
            assert np.abs(a.astype(float) - b.astype(float)).max() <= 1
            page.locator("[data-compare='input']").click()
            checks["model_selector_and_previous_sr_use_distinct_real_checkpoints"] = True
        page.locator("#split-range").evaluate("el => {el.value=25;el.dispatchEvent(new Event('input',{bubbles:true}));}")
        assert "75%" in page.locator("#before-layer").get_attribute("style")
        handle = page.locator("#divider-handle").bounding_box()
        stage = page.locator("#image-stage").bounding_box()
        page.mouse.move(handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2)
        page.mouse.down()
        page.mouse.move(stage["x"] + stage["width"] * .7, stage["y"] + stage["height"] / 2, steps=8)
        page.mouse.up()
        assert 68 < float(page.locator("#split-range").input_value()) < 72
        checks["slider_range_and_pointer_drag"] = True
        page.locator("[data-compare='reference']").click()
        expect(page.locator("#before-tag")).to_have_text("CLEAN ORIGINAL")
        assert page.locator("#before-image").evaluate("el=>el.naturalWidth") == 112
        page.locator("[data-compare='bicubic']").click()
        expect(page.locator("#before-tag")).to_have_text("BICUBIC ×4")
        checks["original_and_bicubic_comparison"] = True
        page.locator("[data-layout='side']").click()
        expect(page.locator("#side-view")).to_be_visible()
        expect(page.locator("#slider-view")).to_be_hidden()
        page.locator("#zoom-in").click()
        expect(page.locator("#zoom-level")).to_have_text("150%")
        page.locator("#reset-view").click()
        expect(page.locator("#zoom-level")).to_have_text("Fit")
        page.locator("[data-layout='slider']").click()
        checks["side_by_side_and_zoom"] = True
        with page.expect_download() as event:
            page.locator("#download").click()
        download = event.value
        download.save_as(ARTIFACTS / "downloaded_sr.png")
        with Image.open(ARTIFACTS / "downloaded_sr.png") as image:
            assert image.size == (112, 112)
        checks["download_is_output_resolution"] = True
        pair = page.evaluate("async()=>{const r=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:0})});return r.json();}")
        for key, name in [("input", "uploaded_input.png"), ("reference", "uploaded_original.png")]:
            (ARTIFACTS / name).write_bytes(base64.b64decode(pair[key].split(",", 1)[1]))
        page.locator("#upload-tab").click()
        page.locator("#image-file").set_input_files(ARTIFACTS / "uploaded_input.png")
        page.locator("#reference-file").set_input_files(ARTIFACTS / "uploaded_original.png")
        expect(page.locator("#reconstruct")).to_be_enabled()
        page.locator("#reconstruct").click()
        expect(page.locator("#result-name")).to_have_text("uploaded_input.png", timeout=120000)
        expect(page.locator("#loading-overlay")).to_be_hidden()
        expect(page.locator("#original-option")).to_be_enabled()
        assert float(page.locator("#psnr-value").inner_text().split()[0]) > 0
        checks["upload_with_reference_and_real_metrics"] = True
        page.locator("#remove-reference").click()
        page.locator("#reconstruct").click()
        expect(page.locator("#original-option")).to_be_disabled(timeout=120000)
        expect(page.locator("#psnr-caption")).to_have_text("A clean original is needed")
        checks["unpaired_upload_has_no_quality_score"] = True
        page.locator("#image-file").set_input_files(ARTIFACTS / "uploaded_original.png")
        page.locator("#input-mode").select_option("simulate")
        page.locator("#reconstruct").click()
        expect(page.locator("#result-name")).to_have_text("uploaded_original.png", timeout=120000)
        expect(page.locator("#loading-overlay")).to_be_hidden()
        expect(page.locator("#original-option")).to_be_enabled()
        expect(page.locator("#result-note")).to_contain_text("Simulation")
        checks["clear_photo_simulation"] = True
        page.locator("#samples-tab").click()
        page.locator(".sample-button").nth(3).click()
        expect(page.locator("#result-name")).to_have_text("Test face 04", timeout=120000)
        expect(page.locator("#loading-overlay")).to_be_hidden()
        page.screenshot(path=str(ARTIFACTS / "desktop.png"), full_page=True)
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto(URL, wait_until="networkidle")
        mobile.wait_for_function("document.querySelector('#download').getAttribute('aria-disabled') === 'false'", timeout=120000)
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile horizontal overflow"
        mobile.locator("[data-layout='side']").click()
        expect(mobile.locator("#side-view")).to_be_visible()
        mobile.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
        checks["mobile_layout_no_horizontal_overflow"] = True
        assert not errors, errors
        checks["no_browser_javascript_errors"] = True
        browser.close()
    (ARTIFACTS / "browser_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()

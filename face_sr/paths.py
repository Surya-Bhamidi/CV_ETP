"""Project files are resolved from this package independently of the shell directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "comparison_ui"
APP_ID = "cv-etp-compare-v1"
V4_REVIEW = ROOT / "work/recognition_v4/review.json"
VISUAL_CASES = ROOT / "work/visual_tradeoff/case_studies.json"
CLEAN_CONSISTENCY_REVIEW = ROOT / "work/restoration_passes_v1/clean_only_followup/review.json"

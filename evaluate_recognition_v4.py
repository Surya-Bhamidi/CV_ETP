"""Compatibility entry point for the final v4 workflow."""

from evaluation.evaluate_recognition_v4 import main, pair_scores, fuse, fit_calibration

if __name__ == "__main__":
    main()

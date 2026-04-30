#!/usr/bin/env python3
"""Run the shared lane perception pipeline on one saved image."""

from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_challenge.part_a.lane_pipeline import LanePipelineConfig, detect_lane_geometry, draw_detection_overlay, draw_roi_guides, make_full_frame_gray_panel


IMAGE_PATH = REPO_ROOT / "racetrack_images" / "lane_3" / "image17.png"
CONFIG_PATH = REPO_ROOT / "config" / "part_a" / "lane_follower.yaml"

DILATE_ITERATIONS = 1


def load_lane_detector_params(config_path=CONFIG_PATH):
    with config_path.open() as f:
        data = yaml.safe_load(f)
    return data["lane_detector"]["ros__parameters"]


def make_config():
    params = load_lane_detector_params()
    return LanePipelineConfig(
        white_lower=np.array([params["white_lower_h"], params["white_lower_s"], params["white_lower_v"]]),
        white_upper=np.array([params["white_upper_h"], params["white_upper_s"], params["white_upper_v"]]),
        roi_top_pct=params["roi_top_pct"],
        left_roi_top_pct=params["left_roi_top_pct"],
        left_roi_bottom_pct=params["left_roi_bottom_pct"],
        right_roi_top_pct=params.get("right_roi_top_pct", 0.0),
        right_roi_bottom_pct=params.get("right_roi_bottom_pct", 0.0),
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        hough_threshold=params["hough_threshold"],
        hough_min_line_length=params["hough_min_line_length"],
        hough_max_line_gap=params["hough_max_line_gap"],
        min_angle_deg=params["min_angle_deg"],
        max_angle_deg=params["max_angle_deg"],
        lane_width_px=params["lane_width_px"],
        center_offset_px=params.get("center_offset_px", 0.0),
        dilate_iterations=DILATE_ITERATIONS,
    )


def add_label(img, label):
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(labeled, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def build_montage(original, mask_panel, edges_panel, hough_panel):
    top = np.hstack([
        add_label(original, "1. input / optional crop"),
        add_label(mask_panel, "2. HSV white mask"),
    ])
    bottom = np.hstack([
        add_label(edges_panel, "3. Canny edges"),
        add_label(hough_panel, "4. Hough lines + centerline"),
    ])
    return np.vstack([top, bottom])


def main():
    image_path = IMAGE_PATH
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    config = make_config()
    result = detect_lane_geometry(img, config)

    original_panel = img.copy()
    draw_roi_guides(original_panel, config, result.roi_y)

    mask_panel = make_full_frame_gray_panel(result.mask, img.shape, result.roi_y)
    edges_panel = make_full_frame_gray_panel(result.edges, img.shape, result.roi_y)
    hough_panel = draw_detection_overlay(original_panel, result)

    montage = build_montage(original_panel, mask_panel, edges_panel, hough_panel)

    print(f"Image: {image_path}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Crop: top {config.roi_top_pct}")

    cv2.imshow("lane perception pipeline", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

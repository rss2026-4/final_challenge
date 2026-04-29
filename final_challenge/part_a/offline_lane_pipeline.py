#!/usr/bin/env python3
"""Run the lane perception demo pipeline on one saved image.

This is intentionally not a ROS node. It demonstrates:
crop -> HSV white filtering -> Canny edges -> Hough line segments.
"""

from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = REPO_ROOT / "racetrack_images" / "lane_3" / "image1.png"

# Same values as config/part_a/lane_follower.yaml.
USE_CROP = True
ROI_TOP_PCT = 0.46
LEFT_ROI_TOP_PCT = 0.30
LEFT_ROI_BOTTOM_PCT = -0.25

WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 60, 255])

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 30

MIN_ANGLE_DEG = 10.0
MAX_ANGLE_DEG = 85.0
LANE_WIDTH_PX = 150.0
DILATE_ITERATIONS = 1


def add_label(img, label):
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_full_frame_gray_panel(gray_roi, full_shape, roi_y):
    h, w = full_shape[:2]
    panel = np.zeros((h, w), dtype=np.uint8)
    panel[roi_y:, :] = gray_roi
    return cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)


def fit_line(pts):
    if len(pts) < 4:
        return None

    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    return np.polyfit(ys, xs, 1)


def fit_centerline(left_line, right_line):
    if left_line is not None and right_line is not None:
        return (np.asarray(left_line) + np.asarray(right_line)) / 2.0
    if left_line is not None:
        return np.asarray([left_line[0], left_line[1] + LANE_WIDTH_PX / 2.0])
    if right_line is not None:
        return np.asarray([right_line[0], right_line[1] - LANE_WIDTH_PX / 2.0])
    return None


def draw_line_model(img, line_coeffs, y_top, y_bottom, color, thickness):
    if line_coeffs is None:
        return

    x_top = int(np.polyval(line_coeffs, y_top))
    x_bottom = int(np.polyval(line_coeffs, y_bottom))
    cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)


def draw_hough_and_centerline(img, lines, roi_y, min_angle_deg, max_angle_deg):
    overlay = img.copy()
    h, w = overlay.shape[:2]
    left_pts = []
    right_pts = []

    if lines is None:
        return overlay

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < min_angle_deg or angle > max_angle_deg:
            continue

        y1_global = y1 + roi_y
        y2_global = y2 + roi_y
        mid_x = (x1 + x2) / 2.0

        if mid_x < w / 2.0:
            color = (255, 0, 0)
            left_pts.extend([(x1, y1_global), (x2, y2_global)])
        else:
            color = (0, 0, 255)
            right_pts.extend([(x1, y1_global), (x2, y2_global)])

        cv2.line(
            overlay,
            (int(x1), int(y1_global)),
            (int(x2), int(y2_global)),
            color,
            2,
        )

    left_line = fit_line(left_pts)
    right_line = fit_line(right_pts)
    center_line = fit_centerline(left_line, right_line)

    draw_line_model(overlay, left_line, roi_y, h - 1, (255, 0, 0), 3)
    draw_line_model(overlay, right_line, roi_y, h - 1, (0, 0, 255), 3)
    draw_line_model(overlay, center_line, roi_y, h - 1, (255, 0, 255), 4)

    return overlay


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

    h, w = img.shape[:2]
    roi_y = int(h * ROI_TOP_PCT) if USE_CROP else 0
    roi_y = int(np.clip(roi_y, 0, h - 1))
    roi = img[roi_y:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    roi_h, roi_w = mask.shape[:2]
    if LEFT_ROI_TOP_PCT > 0 or LEFT_ROI_BOTTOM_PCT > 0:
        x_top = int(LEFT_ROI_TOP_PCT * roi_w)
        x_bot = int(LEFT_ROI_BOTTOM_PCT * roi_w)
        pts = np.array([[0, 0], [x_top, 0],
                        [x_bot, roi_h - 1], [0, roi_h - 1]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 0)

    if DILATE_ITERATIONS > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)

    edges = cv2.Canny(mask, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    original_panel = img.copy()
    if roi_y > 0:
        cv2.line(original_panel, (0, roi_y), (w, roi_y), (255, 255, 0), 3)
    if LEFT_ROI_TOP_PCT > 0 or LEFT_ROI_BOTTOM_PCT > 0:
        x_top = int(LEFT_ROI_TOP_PCT * w)
        x_bot = int(LEFT_ROI_BOTTOM_PCT * w)
        cv2.line(original_panel, (x_top, roi_y), (x_bot, h - 1),
                 (255, 255, 0), 3)

    mask_panel = make_full_frame_gray_panel(mask, img.shape, roi_y)
    edges_panel = make_full_frame_gray_panel(edges, img.shape, roi_y)
    hough_panel = draw_hough_and_centerline(
        original_panel,
        lines,
        roi_y,
        MIN_ANGLE_DEG,
        MAX_ANGLE_DEG,
    )

    montage = build_montage(original_panel, mask_panel, edges_panel, hough_panel)

    print(f"Image: {image_path}")
    print(f"Crop: {'top ' + str(ROI_TOP_PCT) if USE_CROP else 'off'}")

    cv2.imshow("lane perception pipeline", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

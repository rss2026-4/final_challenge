#!/usr/bin/env python3
"""Shared lane perception helpers for online and offline pipelines."""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class LanePipelineConfig:
    white_lower: np.ndarray = field(default_factory=lambda: np.array([0, 0, 200], dtype=np.uint8))
    white_upper: np.ndarray = field(default_factory=lambda: np.array([180, 60, 255], dtype=np.uint8))
    roi_top_pct: float = 0.5
    roi_bottom_pct: float = 0.0
    left_roi_top_pct: float = 0.0
    left_roi_bottom_pct: float = 0.0
    right_roi_top_pct: float = 0.0
    right_roi_bottom_pct: float = 0.0
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 30
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 30
    min_angle_deg: float = 20.0
    max_angle_deg: float = 85.0
    lane_width_px: float = 150.0
    center_offset_px: float = 0.0
    dilate_iterations: int = 1
    line_support_bin_width_px: float = 40.0

    def __post_init__(self):
        self.white_lower = np.asarray(self.white_lower, dtype=np.uint8)
        self.white_upper = np.asarray(self.white_upper, dtype=np.uint8)


@dataclass
class LanePipelineResult:
    roi_y: int
    mask: np.ndarray
    edges: np.ndarray
    raw_lines: Optional[np.ndarray]
    all_lines: list
    left_line: Optional[np.ndarray]
    right_line: Optional[np.ndarray]
    center_line: Optional[np.ndarray]


def detect_lane_geometry(img, config):
    """Detect lane-line geometry from a BGR image."""
    h, w = img.shape[:2]
    roi_y = int(np.clip(h * config.roi_top_pct, 0, h - 1))
    roi = img[roi_y:, :]

    mask = build_white_mask(roi, config)
    edges = cv2.Canny(mask, config.canny_low, config.canny_high)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=config.hough_threshold, minLineLength=config.hough_min_line_length, maxLineGap=config.hough_max_line_gap)

    left_pts, right_pts, all_lines = collect_lane_points(raw_lines, roi_y, h, w, config.min_angle_deg, config.max_angle_deg, config.line_support_bin_width_px)
    left_line = fit_line(left_pts)
    right_line = fit_line(right_pts)
    center_line = fit_centerline(left_line, right_line, config.lane_width_px, config.center_offset_px)

    return LanePipelineResult(
        roi_y=roi_y,
        mask=mask,
        edges=edges,
        raw_lines=raw_lines,
        all_lines=all_lines,
        left_line=left_line,
        right_line=right_line,
        center_line=center_line,
    )


def build_white_mask(roi, config):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, config.white_lower, config.white_upper)
    apply_left_roi_mask(mask, config)
    apply_right_roi_mask(mask, config)
    apply_bottom_roi_mask(mask, config)

    if config.dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=config.dilate_iterations)

    return mask


def apply_left_roi_mask(mask, config):
    if config.left_roi_top_pct <= 0 and config.left_roi_bottom_pct <= 0:
        return

    roi_h, roi_w = mask.shape[:2]
    x_top = int(config.left_roi_top_pct * roi_w)
    x_bot = int(config.left_roi_bottom_pct * roi_w)
    pts = np.array([[0, 0], [x_top, 0], [x_bot, roi_h - 1], [0, roi_h - 1]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 0)


def apply_right_roi_mask(mask, config):
    if config.right_roi_top_pct <= 0 and config.right_roi_bottom_pct <= 0:
        return

    roi_h, roi_w = mask.shape[:2]
    x_top = int((1.0 - config.right_roi_top_pct) * roi_w)
    x_bot = int((1.0 - config.right_roi_bottom_pct) * roi_w)
    pts = np.array([[x_top, 0], [roi_w - 1, 0], [roi_w - 1, roi_h - 1], [x_bot, roi_h - 1]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 0)


def apply_bottom_roi_mask(mask, config):
    if config.roi_bottom_pct <= 0:
        return

    roi_h, roi_w = mask.shape[:2]
    y_cutoff = int((1.0 - config.roi_bottom_pct) * roi_h)
    mask[y_cutoff:, :] = 0


def collect_lane_points(
    raw_lines,
    roi_y,
    image_height,
    image_width,
    min_angle_deg,
    max_angle_deg,
    bin_width_px,
):
    left_bins = {}
    right_bins = {}
    all_lines = []

    if raw_lines is None:
        return [], [], all_lines

    for seg in raw_lines:
        x1, y1, x2, y2 = seg[0]
        if y1 == y2:
            continue

        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < min_angle_deg or angle > max_angle_deg:
            continue

        y1_global = y1 + roi_y
        y2_global = y2 + roi_y
        line_coeffs = fit_line([(x1, y1_global), (x2, y2_global)])
        if line_coeffs is None:
            continue

        x_bottom = float(np.polyval(line_coeffs, image_height - 1))
        if not np.isfinite(x_bottom):
            continue

        side = "left" if x_bottom < image_width / 2.0 else "right"
        all_lines.append((x1, y1_global, x2, y2_global, side))

        bins = left_bins if side == "left" else right_bins
        bin_key = int(round(x_bottom / max(1.0, bin_width_px)))
        bin_data = bins.setdefault(bin_key, {"points": [], "length": 0.0, "angles": []})
        bin_data["points"].extend([(x1, y1_global), (x2, y2_global)])
        seg_len = float(np.hypot(x2 - x1, y2 - y1))
        bin_data["length"] += seg_len
        bin_data["angles"].append((angle, seg_len))

    return (most_vertical_bin_points(left_bins), most_vertical_bin_points(right_bins), all_lines)


def most_vertical_bin_points(bins):
    if not bins:
        return []

    def avg_angle(bin_data):
        total_len = bin_data["length"]
        if total_len < 1e-6:
            return 0.0
        return sum(a * l for a, l in bin_data["angles"]) / total_len

    best = max(bins.values(), key=avg_angle)
    return best["points"]


def fit_line(pts):
    """Fit x = a*y + b via least squares. Returns [a, b]."""
    if len(pts) < 2:
        return None

    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    try:
        return np.polyfit(ys, xs, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None


def fit_centerline(left_line, right_line, lane_width_px, center_offset_px=0.0):
    """Return centerline coefficients for x = a*y + b.

    center_offset_px shifts the centerline horizontally in pixels: negative
    biases left, positive biases right.
    """
    if left_line is not None and right_line is not None:
        center = (np.asarray(left_line) + np.asarray(right_line)) / 2.0
    elif left_line is not None:
        center = np.asarray([left_line[0], left_line[1] + lane_width_px / 2.0])
    elif right_line is not None:
        center = np.asarray([right_line[0], right_line[1] - lane_width_px / 2.0])
    else:
        return None

    center[1] += center_offset_px
    return center


def choose_lookahead(
    center_line,
    image_shape,
    homography,
    lookahead_distance_m,
    lookahead_samples,
    lookahead_row_pct,
    roi_y,
):
    if center_line is None:
        return None

    h, w = image_shape[:2]
    result = choose_distance_matched_lookahead(center_line, h, w, roi_y, homography, lookahead_distance_m, lookahead_samples)
    if result is not None:
        return result

    return choose_fixed_row_lookahead(center_line, h, homography, lookahead_row_pct)


def choose_distance_matched_lookahead(
    center_line,
    h,
    w,
    roi_y,
    homography,
    lookahead_distance_m,
    lookahead_samples,
):
    best = None

    for y in np.linspace(h - 1, roi_y, max(2, int(lookahead_samples))):
        x = float(np.polyval(center_line, y))
        if not np.isfinite(x) or x < 0.0 or x >= w:
            continue

        gx, gy = homography.transform_uv_to_xy(x, y)
        if not np.isfinite(gx) or not np.isfinite(gy) or gx <= 0.05:
            continue

        distance_error = abs(np.hypot(gx, gy) - lookahead_distance_m)
        candidate = (distance_error, x, y, gx, gy)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return None

    _, x, y, gx, gy = best
    return x, y, gx, gy


def choose_fixed_row_lookahead(center_line, h, homography, lookahead_row_pct):
    y = int(h * lookahead_row_pct)
    x = float(np.polyval(center_line, y))
    gx, gy = homography.transform_uv_to_xy(x, y)
    if not np.isfinite(gx) or not np.isfinite(gy) or gx <= 0.05:
        return None
    return x, y, gx, gy


def make_full_frame_gray_panel(gray_roi, full_shape, roi_y):
    h, w = full_shape[:2]
    panel = np.zeros((h, w), dtype=np.uint8)
    panel[roi_y:, :] = gray_roi
    return cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)


def draw_line_model(img, line_coeffs, y_top, y_bottom, color, thickness):
    if line_coeffs is None:
        return

    x_top = int(np.polyval(line_coeffs, y_top))
    x_bottom = int(np.polyval(line_coeffs, y_bottom))
    cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)


def draw_roi_guides(img, config, roi_y):
    h, w = img.shape[:2]
    cv2.line(img, (0, roi_y), (w, roi_y), (255, 255, 0), 3)

    if config.roi_bottom_pct > 0:
        roi_h = h - roi_y
        y_cutoff = roi_y + int((1.0 - config.roi_bottom_pct) * roi_h)
        cv2.line(img, (0, y_cutoff), (w, y_cutoff), (255, 255, 0), 3)

    if config.left_roi_top_pct > 0 or config.left_roi_bottom_pct > 0:
        x_top = int(config.left_roi_top_pct * w)
        x_bot = int(config.left_roi_bottom_pct * w)
        cv2.line(img, (x_top, roi_y), (x_bot, h - 1), (255, 255, 0), 3)

    if config.right_roi_top_pct > 0 or config.right_roi_bottom_pct > 0:
        x_top = int((1.0 - config.right_roi_top_pct) * w)
        x_bot = int((1.0 - config.right_roi_bottom_pct) * w)
        cv2.line(img, (x_top, roi_y), (x_bot, h - 1), (255, 255, 0), 3)


def draw_detection_overlay(
    img,
    result,
    config=None,
    lookahead_uv=None,
    raw_line_thickness=2,
):
    overlay = img.copy()
    h, w = overlay.shape[:2]

    for x1, y1, x2, y2, side in result.all_lines:
        color = (255, 0, 0) if side == "left" else (0, 0, 255)
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, raw_line_thickness)

    draw_line_model(overlay, result.left_line, result.roi_y, h - 1, (255, 0, 0), 3)
    draw_line_model(overlay, result.right_line, result.roi_y, h - 1, (0, 0, 255), 3)
    draw_line_model(overlay, result.center_line, result.roi_y, h - 1, (255, 0, 255), 4)

    if lookahead_uv is not None:
        u, v = int(lookahead_uv[0]), int(lookahead_uv[1])
        cv2.circle(overlay, (u, v), 10, (0, 255, 255), -1)

    if config is not None:
        draw_roi_guides(overlay, config, result.roi_y)

    return overlay

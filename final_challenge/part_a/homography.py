#!/usr/bin/env python3
"""Generate the lane detector homography matrix file."""

from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "config" / "part_a" / "homography_matrix.txt"


def load_homography_matrix(path):
    matrix = np.loadtxt(path, dtype=np.float64)
    return validate_homography_matrix(matrix)


def save_homography_matrix(path, matrix):
    matrix = validate_homography_matrix(matrix)
    header = "Image-to-ground-plane homography matrix. Rows are loaded with numpy.loadtxt."
    np.savetxt(path, matrix, fmt="%.12e", header=header)


def validate_homography_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homography matrix contains non-finite values")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Homography matrix is singular")
    if abs(matrix[2, 2]) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return matrix


class Homography:
    """Pixel-to-ground-plane homography transform.

    Car frame convention: x forward, y left.
    """

    def __init__(self, pts_image=None, pts_ground=None, matrix=None):
        if matrix is not None:
            self.H = validate_homography_matrix(matrix)
            return
        if pts_image is not None and pts_ground is not None:
            self.H = self.compute_from_points(pts_image, pts_ground)
            return
        if pts_image is not None and pts_ground is None:
            self.H = validate_homography_matrix(pts_image)
            return
        raise ValueError("Provide either a matrix or image/ground point pairs")

    @classmethod
    def from_file(cls, path):
        return cls(matrix=load_homography_matrix(path))

    @staticmethod
    def compute_from_points(pts_image, pts_ground):
        """Compute the homography from >=4 point correspondences.

        Args:
            pts_image:  Nx2 list of [u, v] pixel coordinates.
            pts_ground: Nx2 list of [x, y] ground-plane coordinates in meters.
        """
        pts_img = np.float32(pts_image).reshape(-1, 1, 2)
        pts_gnd = np.float32(pts_ground).reshape(-1, 1, 2)
        h_matrix, _ = cv2.findHomography(pts_img, pts_gnd)
        return validate_homography_matrix(h_matrix)

    def transform_uv_to_xy(self, u, v):
        """Convert pixel (u, v) to ground-plane (x, y) in meters."""
        pt = np.array([[u], [v], [1.0]])
        xy = self.H @ pt
        xy /= xy[2, 0]
        return float(xy[0, 0]), float(xy[1, 0])

    def transform_xy_to_uv(self, x, y):
        """Convert ground-plane (x, y) back to pixel (u, v)."""
        h_inv = np.linalg.inv(self.H)
        pt = np.array([[x], [y], [1.0]])
        uv = h_inv @ pt
        uv /= uv[2, 0]
        return float(uv[0, 0]), float(uv[1, 0])

# TODO: Replace these with the final checkerboard calibration correspondences.
# These are the previous Lab 4 defaults used directly by lane_detector.py.
METERS_PER_INCH = 0.0254
PTS_IMAGE_PLANE = [
    [511.0, 198.0],
    [580.0, 251.0],
    [287.0, 196.0],
    [163.0, 205.0],
    [341.0, 185.0],
    [148.0, 313.0],
]
PTS_GROUND_PLANE = [
    [46.0 * METERS_PER_INCH, -24.0 * METERS_PER_INCH],
    [23.5 * METERS_PER_INCH, -16.0 * METERS_PER_INCH],
    [48.5 * METERS_PER_INCH, 6.5 * METERS_PER_INCH],
    [43.0 * METERS_PER_INCH, 22.0 * METERS_PER_INCH],
    [63.0 * METERS_PER_INCH, -1.5 * METERS_PER_INCH],
    [14.0 * METERS_PER_INCH, 10.0 * METERS_PER_INCH],
]


def main():
    homography = Homography(PTS_IMAGE_PLANE, PTS_GROUND_PLANE)
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_homography_matrix(DEFAULT_OUTPUT_PATH, homography.H)

    print(f"Wrote homography matrix to {DEFAULT_OUTPUT_PATH}")
    print(np.array2string(homography.H, precision=12, separator=", "))


if __name__ == "__main__":
    main()

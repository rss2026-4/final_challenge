import numpy as np
import cv2


class Homography:
    """Pixel-to-ground-plane homography transform.

    Car frame convention: x forward, y left.
    """

    def __init__(self, pts_image, pts_ground):
        """Compute the homography from >=4 point correspondences.

        Args:
            pts_image:  Nx2 list of [u, v] pixel coordinates.
            pts_ground: Nx2 list of [x, y] ground-plane coordinates in meters.
        """
        pts_img = np.float32(pts_image).reshape(-1, 1, 2)
        pts_gnd = np.float32(pts_ground).reshape(-1, 1, 2)
        self.H, _ = cv2.findHomography(pts_img, pts_gnd)

    def transform_uv_to_xy(self, u, v):
        """Convert pixel (u, v) to ground-plane (x, y) in meters."""
        pt = np.array([[u], [v], [1.0]])
        xy = self.H @ pt
        xy /= xy[2, 0]
        return float(xy[0, 0]), float(xy[1, 0])

    def transform_xy_to_uv(self, x, y):
        """Convert ground-plane (x, y) back to pixel (u, v)."""
        H_inv = np.linalg.inv(self.H)
        pt = np.array([[x], [y], [1.0]])
        uv = H_inv @ pt
        uv /= uv[2, 0]
        return float(uv[0, 0]), float(uv[1, 0])

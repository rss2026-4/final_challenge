#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

from final_challenge.part_a.homography_utils import Homography

# TODO: Calibrate these for the race track camera setup.
# These are Lab 4 defaults — they WILL need updating for Johnson Track.
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


class LaneDetector(Node):
    def __init__(self):
        super().__init__("lane_detector")

        self.declare_parameter("camera_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("white_lower_h", 0)
        self.declare_parameter("white_lower_s", 0)
        self.declare_parameter("white_lower_v", 200)
        self.declare_parameter("white_upper_h", 180)
        self.declare_parameter("white_upper_s", 60)
        self.declare_parameter("white_upper_v", 255)
        self.declare_parameter("roi_top_pct", 0.5)
        self.declare_parameter("left_roi_top_pct", 0.0)
        self.declare_parameter("left_roi_bottom_pct", 0.0)
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)
        self.declare_parameter("hough_threshold", 30)
        self.declare_parameter("hough_min_line_length", 30)
        self.declare_parameter("hough_max_line_gap", 30)
        self.declare_parameter("min_angle_deg", 20.0)
        self.declare_parameter("max_angle_deg", 85.0)
        self.declare_parameter("lookahead_row_pct", 0.65)
        self.declare_parameter("lane_width_px", 150.0)

        camera_topic = self.get_parameter("camera_topic").value
        self.white_lower = np.array([
            self.get_parameter("white_lower_h").value,
            self.get_parameter("white_lower_s").value,
            self.get_parameter("white_lower_v").value,
        ])
        self.white_upper = np.array([
            self.get_parameter("white_upper_h").value,
            self.get_parameter("white_upper_s").value,
            self.get_parameter("white_upper_v").value,
        ])
        self.roi_top_pct = self.get_parameter("roi_top_pct").value
        self.left_roi_top_pct = self.get_parameter("left_roi_top_pct").value
        self.left_roi_bottom_pct = self.get_parameter("left_roi_bottom_pct").value
        self.canny_low = self.get_parameter("canny_low").value
        self.canny_high = self.get_parameter("canny_high").value
        self.hough_threshold = self.get_parameter("hough_threshold").value
        self.hough_min_length = self.get_parameter("hough_min_line_length").value
        self.hough_max_gap = self.get_parameter("hough_max_line_gap").value
        self.min_angle_deg = self.get_parameter("min_angle_deg").value
        self.max_angle_deg = self.get_parameter("max_angle_deg").value
        self.lookahead_row_pct = self.get_parameter("lookahead_row_pct").value
        self.lane_width_px = self.get_parameter("lane_width_px").value

        self.homography = Homography(PTS_IMAGE_PLANE, PTS_GROUND_PLANE)
        self.bridge = CvBridge()

        self.left_line = None
        self.right_line = None
        self.lookahead_uv = None
        self.all_lines = []

        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 5)
        self.lookahead_pub = self.create_publisher(
            PointStamped, "/lane/lookahead_point", 1)
        self.marker_pub = self.create_publisher(
            Marker, "/lane/marker", 1)
        self.debug_pub = self.create_publisher(
            Image, "/lane/debug_img", 1)

        self.get_logger().info("LaneDetector initialized")

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        result = self.detect_lanes(img)

        if result is not None:
            gx, gy = result
            pt_msg = PointStamped()
            pt_msg.header.stamp = self.get_clock().now().to_msg()
            pt_msg.header.frame_id = "base_link"
            pt_msg.point.x = gx
            pt_msg.point.y = gy
            self.lookahead_pub.publish(pt_msg)
            self._publish_marker(gx, gy)

        if self.debug_pub.get_subscription_count() > 0:
            self._publish_debug_image(img)

    # ------------------------------------------------------------------
    # Core detection pipeline
    # ------------------------------------------------------------------

    def detect_lanes(self, img):
        h, w = img.shape[:2]

        # 1 — Crop to road surface ROI
        roi_y = int(h * self.roi_top_pct)
        roi = img[roi_y:, :]

        # 2 — White mask in HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 2b — Mask out left slanted ROI
        roi_h, roi_w = mask.shape[:2]
        if self.left_roi_top_pct > 0 or self.left_roi_bottom_pct > 0:
            x_top = int(self.left_roi_top_pct * roi_w)
            x_bot = int(self.left_roi_bottom_pct * roi_w)
            pts = np.array([[0, 0], [x_top, 0],
                            [x_bot, roi_h - 1], [0, roi_h - 1]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)

        # 3 — Edge detection
        edges = cv2.Canny(mask, self.canny_low, self.canny_high)

        # 4 — Hough line segments
        raw_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )
        if raw_lines is None:
            self.left_line = self.right_line = None
            self.all_lines = []
            self.lookahead_uv = None
            return None

        # 5 — Filter by angle and classify left / right
        left_pts, right_pts = [], []
        debug_lines = []

        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            y1 += roi_y
            y2 += roi_y

            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < self.min_angle_deg or angle > self.max_angle_deg:
                continue

            mid_x = (x1 + x2) / 2.0
            side = "left" if mid_x < w / 2.0 else "right"
            debug_lines.append((x1, y1, x2, y2, side))

            bucket = left_pts if side == "left" else right_pts
            bucket.extend([(x1, y1), (x2, y2)])

        self.all_lines = debug_lines

        # 6 — Least-squares line fit per side: x = a*y + b
        left_line = self._fit_line(left_pts) if len(left_pts) >= 4 else None
        right_line = self._fit_line(right_pts) if len(right_pts) >= 4 else None
        self.left_line = left_line
        self.right_line = right_line

        # 7 — Midpoint at the lookahead row
        lookahead_y = int(h * self.lookahead_row_pct)

        if left_line is not None and right_line is not None:
            x_mid = (np.polyval(left_line, lookahead_y) +
                     np.polyval(right_line, lookahead_y)) / 2.0
        elif left_line is not None:
            x_mid = np.polyval(left_line, lookahead_y) + self.lane_width_px / 2.0
        elif right_line is not None:
            x_mid = np.polyval(right_line, lookahead_y) - self.lane_width_px / 2.0
        else:
            self.lookahead_uv = None
            return None

        self.lookahead_uv = (x_mid, lookahead_y)

        # 8 — Homography → ground-plane metres in the car frame
        gx, gy = self.homography.transform_uv_to_xy(x_mid, lookahead_y)
        if gx <= 0.05:
            return None

        return gx, gy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_line(pts):
        """Fit x = a*y + b via least squares.  Returns [a, b]."""
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        try:
            return np.polyfit(ys, xs, 1)
        except (np.RankWarning, np.linalg.LinAlgError):
            return None

    def _publish_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.a = 1.0
        m.color.g = 1.0
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.orientation.w = 1.0
        self.marker_pub.publish(m)

    def _publish_debug_image(self, img):
        debug = img.copy()
        h, w = debug.shape[:2]
        roi_y = int(h * self.roi_top_pct)

        for x1, y1, x2, y2, side in self.all_lines:
            color = (255, 0, 0) if side == "left" else (0, 0, 255)
            cv2.line(debug, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        for line_coeffs in (self.left_line, self.right_line):
            if line_coeffs is not None:
                x_bot = int(np.polyval(line_coeffs, h - 1))
                x_top = int(np.polyval(line_coeffs, roi_y))
                cv2.line(debug, (x_bot, h - 1), (x_top, roi_y), (0, 255, 0), 2)

        if self.lookahead_uv is not None:
            u, v = int(self.lookahead_uv[0]), int(self.lookahead_uv[1])
            cv2.circle(debug, (u, v), 10, (0, 255, 255), -1)
            cv2.line(debug, (0, v), (w, v), (0, 255, 255), 1)

        cv2.line(debug, (0, roi_y), (w, roi_y), (255, 255, 0), 3)

        if self.left_roi_top_pct > 0 or self.left_roi_bottom_pct > 0:
            x_top = int(self.left_roi_top_pct * w)
            x_bot = int(self.left_roi_bottom_pct * w)
            cv2.line(debug, (x_top, roi_y), (x_bot, h - 1), (255, 255, 0), 3)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from final_challenge.part_a.homography import Homography
from final_challenge.part_a.lane_pipeline import LanePipelineConfig, choose_lookahead, detect_lane_geometry, draw_detection_overlay


def default_homography_matrix_path():
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("final_challenge"))
        / "config"
        / "part_a"
        / "homography_matrix.txt"
    )


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
        self.declare_parameter("roi_bottom_pct", 0.0)
        self.declare_parameter("left_roi_top_pct", 0.0)
        self.declare_parameter("left_roi_bottom_pct", 0.0)
        self.declare_parameter("right_roi_top_pct", 0.0)
        self.declare_parameter("right_roi_bottom_pct", 0.0)
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)
        self.declare_parameter("hough_threshold", 30)
        self.declare_parameter("hough_min_line_length", 30)
        self.declare_parameter("hough_max_line_gap", 30)
        self.declare_parameter("min_angle_deg", 20.0)
        self.declare_parameter("max_angle_deg", 85.0)
        self.declare_parameter("lookahead_distance_m", 1.0)
        self.declare_parameter("lookahead_samples", 80)
        self.declare_parameter("lookahead_row_pct", 0.65)
        self.declare_parameter("lane_width_px", 150.0)
        self.declare_parameter("center_offset_px", 0.0)
        self.declare_parameter("homography_matrix_path", "")

        camera_topic = self.get_parameter("camera_topic").value
        self.pipeline_config = LanePipelineConfig(
            white_lower=np.array([self.get_parameter("white_lower_h").value, self.get_parameter("white_lower_s").value, self.get_parameter("white_lower_v").value]),
            white_upper=np.array([self.get_parameter("white_upper_h").value, self.get_parameter("white_upper_s").value, self.get_parameter("white_upper_v").value]),
            roi_top_pct=self.get_parameter("roi_top_pct").value,
            roi_bottom_pct=self.get_parameter("roi_bottom_pct").value,
            left_roi_top_pct=self.get_parameter("left_roi_top_pct").value,
            left_roi_bottom_pct=self.get_parameter("left_roi_bottom_pct").value,
            right_roi_top_pct=self.get_parameter("right_roi_top_pct").value,
            right_roi_bottom_pct=self.get_parameter("right_roi_bottom_pct").value,
            canny_low=self.get_parameter("canny_low").value,
            canny_high=self.get_parameter("canny_high").value,
            hough_threshold=self.get_parameter("hough_threshold").value,
            hough_min_line_length=self.get_parameter("hough_min_line_length").value,
            hough_max_line_gap=self.get_parameter("hough_max_line_gap").value,
            min_angle_deg=self.get_parameter("min_angle_deg").value,
            max_angle_deg=self.get_parameter("max_angle_deg").value,
            lane_width_px=self.get_parameter("lane_width_px").value,
            center_offset_px=self.get_parameter("center_offset_px").value,
        )
        self.lookahead_distance_m = self.get_parameter("lookahead_distance_m").value
        self.lookahead_samples = max(2, int(self.get_parameter("lookahead_samples").value))
        self.lookahead_row_pct = self.get_parameter("lookahead_row_pct").value

        homography_path = self.get_parameter("homography_matrix_path").value
        if not homography_path:
            homography_path = str(default_homography_matrix_path())
        self.homography = Homography.from_file(homography_path)
        self.bridge = CvBridge()

        self.last_detection = None
        self.lookahead_uv = None
        self.image_shape = None

        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 5)
        self.lookahead_pub = self.create_publisher(PointStamped, "/lane/lookahead_point", 1)
        self.cross_track_error_pub = self.create_publisher(Float32, "/lane/cross_track_error", 1)
        self.marker_array_pub = self.create_publisher(MarkerArray, "/lane/markers", 1)
        self.debug_pub = self.create_publisher(Image, "/lane/debug_img", 1)

        self.get_logger().info(
            f"LaneDetector initialized with homography {homography_path}"
        )

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
            self._publish_cross_track_error(img.shape)
            self._publish_lane_markers(gx, gy)
        else:
            self._clear_lane_markers()

        if self.debug_pub.get_subscription_count() > 0:
            self._publish_debug_image(img)

    def detect_lanes(self, img):
        self.image_shape = img.shape
        self.last_detection = detect_lane_geometry(img, self.pipeline_config)
        lookahead = choose_lookahead(self.last_detection.center_line, img.shape, self.homography, self.lookahead_distance_m, self.lookahead_samples, self.lookahead_row_pct, self.last_detection.roi_y)
        if lookahead is None:
            self.lookahead_uv = None
            return None

        x_mid, lookahead_y, gx, gy = lookahead
        self.lookahead_uv = (x_mid, lookahead_y)
        return gx, gy

    def _compute_cross_track_error(self, image_shape):
        """Signed y of the centerline point closest to the car origin (0,0)."""
        if self.last_detection is None or self.last_detection.center_line is None:
            return None

        h, w = image_shape[:2]
        roi_y = self.last_detection.roi_y
        center_line = self.last_detection.center_line

        best_dist_sq = float("inf")
        best_y = None
        for row in np.linspace(h - 1, roi_y, 20):
            u = float(np.polyval(center_line, row))
            if not np.isfinite(u) or u < 0.0 or u >= w:
                continue
            x_base, y_base = self.homography.transform_uv_to_xy(u, row)
            if not np.isfinite(x_base) or not np.isfinite(y_base):
                continue
            d2 = x_base * x_base + y_base * y_base
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_y = y_base

        return best_y

    def _publish_cross_track_error(self, image_shape):
        cte = self._compute_cross_track_error(image_shape)
        if cte is None:
            return
        msg = Float32()
        msg.data = float(cte)
        self.cross_track_error_pub.publish(msg)

    def _publish_lane_markers(self, lookahead_x, lookahead_y):
        if self.last_detection is None or self.image_shape is None:
            return

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker(stamp))

        marker_id = 1
        for name, line, color in (
            ("left_lane", self.last_detection.left_line, (0.0, 0.2, 1.0, 1.0)),
            ("right_lane", self.last_detection.right_line, (1.0, 0.1, 0.0, 1.0)),
            ("centerline", self.last_detection.center_line, (1.0, 0.0, 1.0, 1.0)),
        ):
            marker = self._make_line_marker(marker_id, name, line, color, stamp)
            marker_id += 1
            if marker is not None:
                markers.markers.append(marker)

        markers.markers.append(self._make_sphere_marker(marker_id, "lookahead", lookahead_x, lookahead_y, (1.0, 1.0, 0.0, 1.0), stamp))
        self.marker_array_pub.publish(markers)

    def _clear_lane_markers(self):
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker(self.get_clock().now().to_msg()))
        self.marker_array_pub.publish(markers)

    @staticmethod
    def _delete_all_marker(stamp):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    def _make_line_marker(self, marker_id, name, line, color, stamp):
        points = self._sample_line_in_base_link(line)
        if len(points) < 2:
            return None

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "lane_geometry"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.035
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.pose.orientation.w = 1.0
        marker.text = name
        marker.points = points
        return marker

    def _make_sphere_marker(self, marker_id, name, x, y, color, stamp):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "lane_geometry"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.18
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.04
        marker.pose.orientation.w = 1.0
        marker.text = name
        return marker

    def _sample_line_in_base_link(self, line):
        if line is None:
            return []

        h, w = self.image_shape[:2]
        roi_y = self.last_detection.roi_y
        points = []

        for y in np.linspace(h - 1, roi_y, 20):
            u = float(np.polyval(line, y))
            if not np.isfinite(u) or u < 0.0 or u >= w:
                continue

            x_base, y_base = self.homography.transform_uv_to_xy(u, y)
            if not np.isfinite(x_base) or not np.isfinite(y_base):
                continue

            point = Point()
            point.x = float(x_base)
            point.y = float(y_base)
            point.z = 0.02
            points.append(point)

        return points

    def _publish_debug_image(self, img):
        if self.last_detection is None:
            return

        debug = draw_detection_overlay(img, self.last_detection, config=self.pipeline_config, lookahead_uv=self.lookahead_uv, raw_line_thickness=1)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

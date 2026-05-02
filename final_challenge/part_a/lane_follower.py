#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32


class LaneFollower(Node):
    WHEELBASE = 0.325
    MAX_STEER = 0.34

    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("speed", 2.0)
        self.declare_parameter("wheelbase", self.WHEELBASE)
        self.declare_parameter("max_steer", self.MAX_STEER)
        self.declare_parameter("steering_ema_alpha", 1.0)
        self.declare_parameter("max_steering_delta", 0.05)

        drive_topic = self.get_parameter("drive_topic").value
        self.speed = self.get_parameter("speed").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.max_steer = self.get_parameter("max_steer").value
        self.steering_ema_alpha = self._clamp(self.get_parameter("steering_ema_alpha").value, 0.0, 1.0)
        self.max_steering_delta = self.get_parameter("max_steering_delta").value
        self.filtered_steering = None

        self.lookahead_sub = self.create_subscription(PointStamped, "/lane/lookahead_point", self.lookahead_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.steering_angle_pub = self.create_publisher(Float32, "/lane/steering_angle", 1)
        self.filtered_steering_angle_pub = self.create_publisher(Float32, "/lane/filtered_steering_angle", 1)

        self.get_logger().info(
            f"LaneFollower ready  speed={self.speed}  drive={drive_topic}  "
            f"steering_ema_alpha={self.steering_ema_alpha}")

    def lookahead_callback(self, msg):
        gx = msg.point.x
        gy = msg.point.y

        L = math.hypot(gx, gy)
        if L < 0.01:
            return

        alpha = math.atan2(gy, gx)
        steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), L)
        steering = max(-self.max_steer, min(self.max_steer, steering))
        filtered_steering = self._filter_steering(steering)

        self._publish_float(self.steering_angle_pub, steering)
        self._publish_float(self.filtered_steering_angle_pub, filtered_steering)

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.steering_angle = filtered_steering
        drive.drive.speed = self.speed
        self.drive_pub.publish(drive)

    def _filter_steering(self, steering):
        if self.filtered_steering is None:
            self.filtered_steering = steering
            return steering

        delta = steering - self.filtered_steering
        if abs(delta) > self.max_steering_delta:
            steering = self.filtered_steering + math.copysign(self.max_steering_delta, delta)

        alpha = self.steering_ema_alpha
        self.filtered_steering = alpha * steering + (1.0 - alpha) * self.filtered_steering
        return self.filtered_steering

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _publish_float(publisher, value):
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

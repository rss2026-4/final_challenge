#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped


class LaneFollower(Node):
    WHEELBASE = 0.325
    MAX_STEER = 0.34

    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("speed", 2.0)
        self.declare_parameter("wheelbase", self.WHEELBASE)
        self.declare_parameter("max_steer", self.MAX_STEER)

        drive_topic = self.get_parameter("drive_topic").value
        self.speed = self.get_parameter("speed").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.max_steer = self.get_parameter("max_steer").value

        self.lookahead_sub = self.create_subscription(
            PointStamped, "/lane/lookahead_point", self.lookahead_callback, 1)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)

        self.get_logger().info(
            f"LaneFollower ready  speed={self.speed}  drive={drive_topic}")

    def lookahead_callback(self, msg):
        gx = msg.point.x
        gy = msg.point.y

        L = math.hypot(gx, gy)
        if L < 0.01:
            return

        alpha = math.atan2(gy, gx)
        steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), L)
        steering = max(-self.max_steer, min(self.max_steer, steering))

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.steering_angle = steering
        drive.drive.speed = self.speed
        self.drive_pub.publish(drive)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vis_msgs.msg import ObjectLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from visual_servoing.computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        self.declare_parameter("object_px", "/relative_object_px")
        self.declare_parameter("object_debug", "/object_debug_img")
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")

        self.OBJECT_PX = self.get_parameter("object_px").get_parameter_value().string_value
        self.OBJECT_DEBUG = self.get_parameter("object_debug").get_parameter_value().string_value
        self.IMAGE_TOPIC = self.get_parameter("image_topic").get_parameter_value().string_value

        # Subscribe to ZED camera RGB frames
        self.parking_meter = self.create_publisher(ObjectLocationPixel, self.OBJECT_PX, 10)
        self.debug_pub = self.create_publisher(Image, self.OBJECT_DEBUG, 10)
        self.image_sub = self.create_subscription(Image, self.IMAGE_TOPIC, self.image_callback, 5)

        self.bridge = CvBridge()  # Converts between ROS images and OpenCV Images
        self.get_logger().info("Parking Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.
        

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        if self.LineFollower:
            image[0:200, :] = 0
            image[260:, :] = 0

        (x1, y1), (x2, y2) = cd_color_segmentation(image, None)

        if x2 - x1 > 0 and y2 - y1 > 0:
            object_px = ObjectLocationPixel()
            object_px.label = "Parking Meter"
            object_px.u = float((x1 + x2) / 2.0)
            if self.LineFollower: 
                object_px.v = float((y1 + y2) / 2.0)
            else:
                object_px.v = float(y2)
            self.parking_meter.publish(object_px)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (int(object_px.u), int(object_px.v)), 5, (0, 0, 255), -1)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
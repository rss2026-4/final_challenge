import rclpy
from rclpy.node import Node
import numpy as np

import cv2

from vis_msgs.msg import ObjectLocation, ParkingError, ObjectLocationPixel
from state_msgs.msg import State
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class TrafficLight(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic",  "/vesc/high_level/input/navigation")
        self.declare_parameter("error_topic",  "/parking_error")
        self.declare_parameter("object_topic",  "/closest_obj")
        self.declare_parameter("state_topic", "/state")

        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.ERROR_TOPIC = self.get_parameter("error_topic").get_parameter_value().string_value
        self.OBJECT_TOPIC = self.get_parameter("object_topic").get_parameter_value().string_value
        self.STATE_TOPIC = self.get_parameter("state_topic").get_parameter_value().string_value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, self.ERROR_TOPIC, 10)
        self.state_pub = self.create_publisher(State, self.STATE_TOPIC, 10)
        self.object_sub = self.create_subscription(ObjectLocation, self.OBJECT_TOPIC, self.relative_parking_callback, 1)
        self.state_sub = self.create_subscription(State, self.STATE_TOPIC, self.state_callback, 1)

        self.parking_distance_min = 0.5
        self.parking_distance_max = 1  # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.robot_offset = 0.25 # offset depending on how far the wheels are - apparently part of ts pure pursuit controller lol
        self.backing_up = False

        self.drive_cmd = AckermannDriveStamped()
        self.create_timer(1 / 20, self.timer_callback)

        self.get_logger().info("Parking Controller Initialized - testing for changes")

        self.disabled = True
        self.traffic_light = False

        #updated variables
        self.traffic_light_pub = self.create_publisher(ObjectLocationPixel, "/relative_traffic_px", 10)
        self.debug_pub = self.create_publisher(Image, "/traffic_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()


        self.declare_parameter('annotated_image_topic', "/yolo/color_image")

        self.ANNOTATED_IMAGE_TOPIC = self.get_parameter('annotated_image_topic').get_parameter_value().string_value

        self.pub = self.create_publisher(Image, self.ANNOTATED_IMAGE_TOPIC, 10)
    
    def state_callback(self, msg):
        self.get_logger().info("are we parking", {msg.current_state})
        if msg.current_state == State.PARKING_STOP:
            self.disabled = False
        else:
            self.disabled = True

    def image_callback(self, image_msg):
        self.get_logger().info("in image callback")
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        (x1, y1), (x2, y2) = self.cd_color_segmentation(image, None)

        if x2 - x1 > 0 and y2 - y1 > 0:
            traffic_px = ObjectLocationPixel()
            traffic_px.u = float((x1 + x2) / 2.0)
            traffic_px.v = float(y2)
            self.traffic_light_pub.publish(traffic_px)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (int(traffic_px.u), int(traffic_px.v)), 5, (0, 0, 255), -1)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def relative_parking_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)
        

        if self.parking_distance_min< distance < self.parking_distance_max:
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
        else:
            pass

        #################################

        self.drive_cmd = drive_cmd
        self.error_publisher()



    def timer_callback(self):
        pass
        # self.drive_pub.publish(self.drive_cmd)

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################
        
        
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        # error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)
        error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2) - self.parking_distance


        #################################

        self.error_pub.publish(error_msg)

    def image_print(img):
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cd_color_segmentation(self,img, template):
    # convert to input image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # binary mask for orange cone values

        # used for tests
        lower = np.array([0, 0, 255])
        upper = np.array([25, 120, 255])

        # used for robot
        # lower = np.array([5, 80, 150])
        # upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # list of all detected shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            self.traffic_light = False
            self.get_logger().info("NO")
            return ((0, 0), (0, 0))
        else:
            self.traffic_light = True
            self.get_logger().info("R E D")
            # find the largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            bounding_box = ((x, y), (x + w, y + h))
            self.draw_detections(img,bounding_box)

            return bounding_box

    
    def draw_detections(self, bgr_image: np.ndarray,box):
        top_left = box[0]
        bot_right = box[1]
        out_image = bgr_image.copy()
        out_image = cv2.rectangle(out_image, top_left, bot_right,2)
        return out_image

    


def main(args=None):

    
    rclpy.init(args=args)
    pc = TrafficLight()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()






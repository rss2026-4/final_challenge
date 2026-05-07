import rclpy
from rclpy.node import Node
import numpy as np

import cv2

from vis_msgs.msg import ObjectLocation, ParkingError, ObjectLocationPixel, Bounding
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

        self.declare_parameter("drive_topic",  "/vesc/high_level/input/nav_0")
        self.declare_parameter("error_topic",  "/parking_error")
        self.declare_parameter("object_topic",  "/closest_obj")
        self.declare_parameter("state_topic", "/state")
        self.declare_parameter("yolo_topic", "/bounding_box")
        self.declare_parameter("hsv_topic", "/hsv_image")
        self.declare_parameter("yolo_img_topic", "/yolo_image")
        self.declare_parameter("cropped_topic", "/cropped_image")

        

        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.ERROR_TOPIC = self.get_parameter("error_topic").get_parameter_value().string_value
        self.OBJECT_TOPIC = self.get_parameter("object_topic").get_parameter_value().string_value
        self.STATE_TOPIC = self.get_parameter("state_topic").get_parameter_value().string_value
        self.YOLO_TOPIC = self.get_parameter("yolo_topic").get_parameter_value().string_value
        self.HSV_TOPIC = self.get_parameter("hsv_topic").get_parameter_value().string_value
        self.YOLO_IMG_TOPIC = self.get_parameter("yolo_img_topic").get_parameter_value().string_value
        self.CROPPED_TOPIC = self.get_parameter("cropped_topic").get_parameter_value().string_value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, self.ERROR_TOPIC, 10)
        self.state_pub = self.create_publisher(State, self.STATE_TOPIC, 10)
        self.hsv_pub = self.create_publisher(Image, self.HSV_TOPIC, 10)
        self.yolo_pub = self.create_publisher(Image, self.YOLO_IMG_TOPIC, 10)
        self.cropped_pub = self.create_publisher(Image, self.CROPPED_TOPIC, 10)
        self.object_sub = self.create_subscription(ObjectLocation, self.OBJECT_TOPIC, self.relative_parking_callback, 1)
        self.state_sub = self.create_subscription(State, self.STATE_TOPIC, self.state_callback, 1)
        self.yolo_sub = self.create_subscription(Bounding, self.YOLO_TOPIC, self.yolo_bounding, 1)


        self.parking_distance_min = 0.5
        self.parking_distance_max = 2.0       
        self.relative_x = 0
        self.relative_y = 0
        self.robot_offset = 0.25 # offset depending on how far the wheels are - apparently part of ts pure pursuit controller lol
        self.backing_up = False

        self.x_bottom_right = None
        self.y_bottom_right = None
        self.x_top_left = None
        self.y_top_left = None
        self.new_bottom_y = None
        self.w = None

        self.side = True

        self.drive_cmd = AckermannDriveStamped()
        self.create_timer(1 / 20, self.timer_callback)

        self.get_logger().info("Parking Controller Initialized - testing for changes")

        self.disabled = True
        self.red = None
        self.stopped = False

        #updated variables
        self.traffic_light_pub = self.create_publisher(ObjectLocationPixel, "/relative_traffic_px", 10)
        self.debug_pub = self.create_publisher(Image, "/traffic_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()
    
    def state_callback(self, msg):
        #checks to see the state, ie if we detected the traffic light
        if msg.current_state == State.TRAFFIC_STOP:
            self.disabled = False
        else:
            self.disabled = True

    def yolo_bounding(self, bounding):
        if self.disabled:
            return
        #these are the yolo bounding box corners 
        self.x_bottom_right = int(bounding.x_bottom_right)
        self.y_bottom_right = int(bounding.y_bottom_right)
        self.x_top_left = int(bounding.x_top_left)
        self.y_top_left = int(bounding.y_top_left)


    def image_callback(self, image_msg):
        if self.disabled:
            return
            
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        #bounding box of the color segmentation
        cropped = self.traffic_image(image)
        (x1, y1), (x2, y2) = self.cd_color_segmentation(cropped, None)
        new_image = None

        if x2 - x1 > 0 and y2 - y1 > 0:
            # self.get_logger().info('not in here')
            traffic_px = ObjectLocationPixel()
            traffic_px.u = float((x1 + x2) / 2.0)
            traffic_px.v = float((y1+y2) / 2.0)
            yolo_image_raw = cv2.rectangle(image, (self.x_top_left+x1, self.y_top_left+y1), (self.x_top_left+x2, self.y_top_left+y2), (0, 255, 0), 2)
            # final_image = self.bridge.cv2_to_imgmsg(new_image, "bgr8")
            yolo_image_final = self.bridge.cv2_to_imgmsg(yolo_image_raw,'bgr8')
            self.yolo_pub.publish(yolo_image_final)
            # self.hsv_pub.publish(final_image)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def relative_parking_callback(self, msg):
        if self.disabled:
            return
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "base_link"

        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        self.get_logger().info(f'distance from traffic light {distance}')

        if (distance < self.parking_distance_max) and (self.red == True):
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.get_logger().info('within distance, will stop')
            self.drive_cmd = drive_cmd
            self.drive_pub.publish(self.drive_cmd)
            self.stopped = True
            # self.error_publisher()

        # Transition back to forward when light turns green
        # if (self.red != True) or ((self.stopped == True) and distance > 8):
        #     new_state = State()
        #     new_state.current_state = State.PATH_FOLLOWING_FORWARD
        #     self.get_logger().info("Green light means go forward :D ")
        #     state_pub.publish(new_state)


        #################################

        

    def traffic_image(self,old_image):
        if self.disabled:
            return
        if self.x_bottom_right is not None:
            delta_y = int(abs(self.y_top_left - self.y_bottom_right)/3)
            self.new_bottom_y = self.y_top_left + delta_y
            cropped_raw = old_image[self.y_top_left:self.new_bottom_y, self.x_top_left:self.x_bottom_right]
            # self.get_logger().info(f'y_s {self.y_top_left}, {self.y_bottom_right}')
            # self.get_logger().info(f'x_s {self.x_top_left}, {self.x_bottom_right}')
            cropped_final = self.bridge.cv2_to_imgmsg(cropped_raw,'bgr8')
            # self.get_logger().info(f'cropped {type(cropped_final)}')
            self.cropped_pub.publish(cropped_final)
            return cropped_raw
        else:
            return old_image

    
    def cd_color_segmentation(self,img, template):
        if self.disabled:
            return
    # convert to input image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # binary mask for orange cone values

        # used for tests
        #I think this is BGR
        #1,7,250

        lower = np.array([0, 0, 200])
        upper = np.array([50,50,255])
        if self.side == True:
            lower = np.array([0, 160, 0])
            upper = np.array([120,255,255])
            

        # used for robot
        # lower = np.array([5, 80, 150])
        # upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # list of all detected shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            self.red = False
            self.get_logger().info('we can drive')
            return ((0, 0), (0, 0))
        self.get_logger().info('red light detected')
        self.red = True
        # find the largest contour
        
        if self.x_bottom_right is not None:
            # self.get_logger().info(f'contouring')
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            bounding_box = ((x, y), (x + w, y + h))
            if contours is None:
                self.get_logger().info(f'returning zeros')
                return ((0, 0), (0, 0))

            return bounding_box
        return ((0,0), (0,0))


    def timer_callback(self):

        pass
        # self.drive_pub.publish(self.drive_cmd)

    def error_publisher(self):
        pass
        """
        # Publish the error between the car and the cone. We will view this
        # with rqt_plot to plot the success of the controller
        # """
        # error_msg = ParkingError()

        # #################################
        
        
        # # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        # error_msg.x_error = self.relative_x
        # error_msg.y_error = self.relative_y
        # # error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)
        # error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2) - self.parking_distance


        # #################################

        # self.error_pub.publish(error_msg)

    def image_print(img):
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

   

def main(args=None):

    rclpy.init(args=args)
    pc = TrafficLight()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
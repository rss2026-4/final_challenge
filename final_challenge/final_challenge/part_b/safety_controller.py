#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
# from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

# from safety_controller.visualization_tools import VisualizationTools

class SafetyController(Node):

    def __init__(self):
        super().__init__('safety_controller')

        # DECLARE PARAMETERS
        self.declare_parameter('safe_topic', '/vesc/low_level/input/safety')
        self.declare_parameter('drive_topic', '/vesc/low_level/ackermann_cmd')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('desired_distance', 1.0)

        # GET PARAMETERS
        self.SAFE_TOPIC = self.get_parameter('safe_topic').get_parameter_value().string_value
        self.SCAN_TOPIC  = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        # CREATE PUBLISHERS/SUBSCRIBERS

        # drive_sub msg will be one behind laser_sub msg
        self.drive_pub    = self.create_publisher(AckermannDriveStamped, self.SAFE_TOPIC, 10)
        self.drive_sub    = self.create_subscription(AckermannDriveStamped, self.DRIVE_TOPIC, self.drive_callback, 10) 
        self.laser_sub    = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.laser_callback, 10)
        # self.line_pub     = self.create_publisher(Marker, '/wall', 1)
        # self.safety_pub   = self.create_publisher(Bool, '/safety', 1)

        self.current_drive = AckermannDriveStamped()

        # VARIABLES/CONSTANTS
        # ANGLE_INCREMENT =  0.047576 # rads -> about 2.7 degs s

    def drive_callback(self, msg):
        self.current_drive = msg

    def laser_callback(self, msg):

        # create velocity vector and use that to determine collissions

        current_drive = self.current_drive
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = 0.0 # radians
        drive_msg.drive.steering_angle_velocity = 0.0 # radians/s
        drive_msg.drive.acceleration = 0.0 # m/s^2
        drive_msg.drive.jerk = 0.0 # m/s^3
        
        # if object detected
        if self.object_detection(msg, current_drive):
            self.get_logger().info("Stopping!!! obstacle")
            drive_msg.drive.speed = 0.0 # m/s
            self.drive_pub.publish(drive_msg)

        # if self.object_detection(msg, current_drive):
        #     self.get_logger().info("Stopping!!! obstacle")
        #     msg = Bool()
        #     msg.data = True
        #     # self.safety_pub.publish(msg)
        # else:
        #     msg = Bool()
        #     msg.data = False
        #     # self.safety_pub.publish(msg)


        # else:
        #     drive_msg.drive.speed = 2.0 # m/s


    def object_detection(self, laser_msg, current_drive):
        """
        takes in laserscan data and returns true if object detected, false if not
        """
        # safety params
        halfway_indx = len(laser_msg.ranges)//2
        angle_range = 10 # num indexes away to have a certain angle range; angle_range*2.7 degs = degree range 
        start = halfway_indx-angle_range
        end = halfway_indx+angle_range
        safety_range = 0.25

        
        # x = [0.0]
        # y = [0.0]
        filtered_data = laser_msg.ranges[start:end]
        # x.append(laser_msg.ranges[halfway_indx]*np.cos(0))
        # y.append(laser_msg.ranges[halfway_indx]*np.sin(0))


        num_danger = 0
        for point in filtered_data:
            if num_danger >= 3:
                return True
            
            if point < safety_range:
                num_danger += 1
    
        # VisualizationTools.plot_line(x, y, self.line_pub, frame="/laser")
        return False


def main():
    rclpy.init()
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
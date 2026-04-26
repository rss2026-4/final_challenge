import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ObjectLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped


class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic",  "/vesc/low_level/input/navigation")
        self.declare_parameter("error_topic",  "/parking_error")
        self.declare_parameter("object_topic",  "/relative_parking")

        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.ERROR_TOPIC = self.get_parameter("error_topic").get_parameter_value().string_value
        self.OBJECT_TOPIC = self.get_parameter("object_topic").get_parameter_value().string_value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, self.ERROR_TOPIC, 10)
        self.object_pub = self.create_subscription(ObjectLocation, self.OBJECT_TOPIC, self.relative_parking_callback, 1)

        self.parking_distance = 0.5  # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.robot_offset = 0.25 # offset depending on how far the wheels are - apparently part of ts pure pursuit controller lol
        self.backing_up = False

        self.drive_cmd = AckermannDriveStamped()
        self.create_timer(1 / 20, self.timer_callback)

        self.get_logger().info("Parking Controller Initialized - testing for changes")

    def relative_parking_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd
        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        if self.backing_up:
            # CASE 1: Recovery — reverse straight until far enough away
            if self.relative_x < self.parking_distance + 0.3:
                drive_cmd.drive.speed = -0.5
                drive_cmd.drive.steering_angle = 0.0
            else:
                self.backing_up = False

        elif distance < self.parking_distance:
            # CASE 2: Close enough — stop
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0

            # Back up if we're not well-aligned
            if abs(self.relative_y) > 0.3:
                self.backing_up = True

        elif self.relative_x < 0:
            # CASE 3: Cone is behind us — back up straight
            drive_cmd.drive.speed = -0.5
            drive_cmd.drive.steering_angle = 0.0

        else:
            # CASE 4: Drive toward cone using pure pursuit
            eta = np.arctan2(self.relative_y, self.relative_x)

            # lookahead = min(distance, 0.5)   # cap lookahead so turns stay steep even at long range
            lookahead = distance
            steering_angle = np.arctan2(
                2.0 * self.robot_offset * np.sin(eta),
                lookahead
            )

            # given saftey controller problems makes sense to scale speed
            speed = min(1.5, max(0.3, distance))
            # speed = 1.5


            drive_cmd.drive.speed = float(speed)
            drive_cmd.drive.steering_angle = float(steering_angle)


        #################################

        self.drive_cmd = drive_cmd
        self.error_publisher()

    def timer_callback(self):
        self.drive_pub.publish(self.drive_cmd)

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


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
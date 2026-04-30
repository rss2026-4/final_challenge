from final_challenge.part_b.sensor_model import SensorModel
from final_challenge.part_b.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, PoseStamped

from rclpy.node import Node
import rclpy

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('initial_pose', "/initialpose")
        self.declare_parameter('odom_pub', "/pf/pose/odom")
        self.declare_parameter('pose_pub', "/pf/pose")


        self.PF_FRAME = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.ODOM_TOPIC = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.INITIAL_POSE = self.get_parameter("initial_pose").get_parameter_value().string_value
        
        self.ODOM_PUB = self.get_parameter("odom_pub").get_parameter_value().string_value
        self.POSE_PUB = self.get_parameter("pose_pub").get_parameter_value().string_value

        self.particle_filter_frame = self.PF_FRAME.lstrip('/')

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
  

        self.laser_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.ODOM_TOPIC, self.odom_callback, 1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.INITIAL_POSE, self.pose_callback, 1)

        self.odom_pub = self.create_publisher(Odometry, self.ODOM_PUB, 1)
        self.pose_pub = self.create_publisher(PoseStamped, self.POSE_PUB, 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.declare_parameter('num_particles', 200)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        # Already declared by SensorModel, just read it
        self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

        # initialies particle array with zeros, weights all particles equally
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_init = False
        self.last_odom_time = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_array_pub = self.create_publisher(PoseArray, "/particles", 1)
        self.scan_pub = self.create_publisher(LaserScan, "/pf/scan", 1)
        self.latest_scan = None

        self.get_logger().info("Particle Filter Initialized")
        self.get_logger().info("=============+READY+=============")

    def pose_callback(self, msg):
        """
        This is called when the user clicks 2d pose estimate in rviz, 
        which results in a pose publisehd to /initialpose topic. It then 
        spreads num_particles around the clicked pose with noise, 
        reset the weights so they are all equal, and enable the particle filter.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # yaw
        quat = msg.pose.pose.orientation
        theta = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')[2]

        # noiseily scatter partildes aroudn clicked pose
        center = np.array([x, y, theta])
        noise = np.random.normal(0, 0.1, size=(self.num_particles, 3))
        # noise = np.zeros((self.num_particles, 3))
        self.particles = center + noise
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_init = True

        self.get_logger().info("Particles initialized around (%.2f, %.2f, %.2f)" % (x, y, theta))
        
        
    def odom_callback(self, msg):
        """
        This is called on every new odometry message. It updates the particles 
        based on the motion model by gathering information about how the robot 
        moves and then sending that information to the motion model along 
        with the particles.
        """
        if not self.particle_init:
            return

        now = self.get_clock().now()

        # first odometry message so return without updating particles
        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        # ros time to nano seconds to seconds covnerisobn
        dt = (now - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = now

        # gather twist data to feed into motion model
        dx = msg.twist.twist.linear.x * dt
        dy = msg.twist.twist.linear.y * dt
        dtheta = msg.twist.twist.angular.z * dt
        odometry = np.array([dx, dy, dtheta])

        self.particles = self.motion_model.evaluate(self.particles, odometry)
        self.publish_tf()
            
    def laser_callback(self, msg):
        self.latest_scan = msg

        if not self.particle_init:
            return

        ranges = np.array(msg.ranges)
        indices = np.linspace(0, len(ranges) - 1, self.num_beams_per_particle, dtype=int)
        downsampled = ranges[indices]

        weights = self.sensor_model.evaluate(self.particles, downsampled)
        if weights is None:
            return
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return
        self.weights = weights / weight_sum

        # Only resample ~50% of the time to preserve diversity
        if np.random.random() < 0.5:
            indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.particles += np.random.normal(0, [0.05, 0.05, 0.02], size=self.particles.shape)
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights *= weights
            self.weights /= np.sum(self.weights)

        self.publish_tf()
        self.publish_viz()

        now = self.get_clock().now().to_msg()
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = self.particle_filter_frame
        scan.angle_min = msg.angle_min
        scan.angle_max = msg.angle_max
        scan.angle_increment = msg.angle_increment
        scan.time_increment = msg.time_increment
        scan.scan_time = msg.scan_time
        scan.range_min = msg.range_min
        scan.range_max = msg.range_max
        scan.ranges = msg.ranges
        scan.intensities = msg.intensities
        self.scan_pub.publish(scan)

    def publish_tf(self):
        w = self.weights
        mean_x = float(np.sum(w * self.particles[:, 0]))
        mean_y = float(np.sum(w * self.particles[:, 1]))
        mean_theta = float(np.arctan2(
            np.sum(w * np.sin(self.particles[:, 2])),
            np.sum(w * np.cos(self.particles[:, 2]))
        ))

        half = mean_theta * 0.5
        qz = float(np.sin(half))
        qw = float(np.cos(half))
        now = self.get_clock().now().to_msg()

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = self.particle_filter_frame
        odom_msg.pose.pose.position.x = mean_x
        odom_msg.pose.pose.position.y = mean_y
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = mean_x
        pose_msg.pose.position.y = mean_y
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = self.particle_filter_frame
        t.transform.translation.x = mean_x
        t.transform.translation.y = mean_y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def publish_viz(self):
        if self.pose_array_pub.get_subscription_count() > 0:
            now = self.get_clock().now().to_msg()
            pose_array = PoseArray()
            pose_array.header.stamp = now
            pose_array.header.frame_id = "map"
            half_thetas = self.particles[:, 2] * 0.5
            sin_h = np.sin(half_thetas)
            cos_h = np.cos(half_thetas)
            for i in range(self.num_particles):
                pose = Pose()
                pose.position.x = float(self.particles[i, 0])
                pose.position.y = float(self.particles[i, 1])
                pose.orientation.z = float(sin_h[i])
                pose.orientation.w = float(cos_h[i])
                pose_array.poses.append(pose)
            self.pose_array_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
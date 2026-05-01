import rclpy
import numpy as np
from vis_msgs.msg import ObjectLocation, ObjectLocationArray
from state_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseArray, PoseStamped

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        # declare params
        self.declare_parameter('state_topic', '/state')
        self.declare_parameter('object_detect_topic', '/object_detect')
        self.declare_parameter('closest_pub_topic', '/closest_obj')
        self.declare_parameter('traj_topic', "/trajectory/current")
        self.declare_parameter('pose_topic', "/pf/pose")

        # get params
        self.STATE_TOPIC = self.get_parameter('state_topic').get_parameter_value().string_value
        self.OBJECT_DETECT_TOPIC = self.get_parameter('object_detect_topic').get_parameter_value().string_value
        self.CLOSEST_PUB_TOPIC = self.get_parameter('closest_pub_topic').get_parameter_value().string_value
        self.TRAJ_TOPIC = self.get_parameter('traj_topic').get_parameter_value().string_value
        self.POSE_TOPIC = self.get_parameter('pose_topic').get_parameter_value().string_value

        # publishers
        self.state_pub = self.create_publisher(State, self.STATE_TOPIC, 10)
        self.closest_pub = self.create_publisher(ObjectLocation, self.CLOSEST_PUB_TOPIC, 10)
        self.traj_pub = self.create_publisher(PoseArray, self.TRAJ_TOPIC, 10)
        
        # subscribers
        self.state_sub = self.create_subscription(State, self.STATE_TOPIC, self.state_callback, 10)
        self.object_detect_sub = self.create_subscription(ObjectLocationArray, self.OBJECT_DETECT_TOPIC, self.object_detect_callback, 10)
        traj_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.traj_sub = self.create_subscription(PoseArray,self.TRAJ_TOPIC,self.traj_callback,traj_qos)
        self.pose_sub = self.create_subscription(PoseStamped, self.POSE_TOPIC, self.pose_callback, 1)

        # useful attributes
        self.park_start = 0
        self.state_dict = {
            State.PATH_PLANNING_FORWARD : "PATH_PLANNING_FORWARD",
            State.PATH_PLANNING_RETURN: "PATH_PLANNING_RETURN",
            State.PATH_FOLLOWING_FORWARD: "PATH_FOLLOWING_FORWARD",
            State.PATH_FORWARD_DONE: "PATH_FORWARD_DONE",
            State.PATH_FOLLOWING_RETURN: "PATH_FOLLOWING_RETURN",
            State.PATH_RETURN_DONE: "PATH_RETURN_DONE",
            State.TRAFFIC_STOP: "TRAFFIC_STOP",
            State.PARKING_METER: "PARKING_METER",
            State.PARKED: "PARKED",
        }

        self.state = None
        self.return_started = False
        self.forward_started = False

        self.get_logger().info("State Machine Initialized")
        self.initial_state()
        self.current_pose = None
        self.parked_pose = None

    
    def initial_state(self):
        """
        Initializes state as planning (ready to receive points for trajectory)
        """
        new_state = State()
        new_state.current_state = State.PATH_PLANNING_FORWARD

        self.state_pub.publish(new_state)
    
    def pose_callback(self, msg):
        self.current_pose = (msg.pose.position.x, msg.pose.position.y)


    def object_detect_callback(self, msg):
        if self.state == State.PARKED:
            return

        
        objects = msg.objects # list of object location msg items
        min_dist = np.inf
        closest_obj = None
        
        for object_msg in objects:
            
            obj_distance = self.get_dist((object_msg.x_pos, object_msg.y_pos))
            if obj_distance < min_dist:
                min_dist = obj_distance
                closest_obj = object_msg
        
        if closest_obj == None:
            return
        
        if closest_obj.label == "traffic light":
            if self.state != State.PARKING_METER:
                new_state = State()
                new_state.current_state = State.TRAFFIC_STOP

                self.state_pub.publish(new_state)

            self.closest_pub.publish(closest_obj)

        cooldown_dist = 5 # meters
        park_dist = self.get_dist(self.current_pose, self.parked_pose)
        if closest_obj.label == "backpack" and park_dist >= cooldown_dist:
            if self.state != State.PARKING_METER:
                new_state = State()
                new_state.current_state = State.PARKING_METER
                self.state_pub.publish(new_state)
        
            self.closest_pub.publish(closest_obj)
    
    def state_callback(self, msg):
        if self.state == msg.current_state:
            return
        else:
            self.state = msg.current_state

        self.get_logger().info(f"Current State: {self.state_dict[self.state]}")
        
        if msg.current_state == State.PARKED:
            self.park_start = self.get_clock().now().nanoseconds / 1e9 
            self.parked_pose = self.current_pose
            self.timer = self.create_timer(6, self.park_timer)
        
        if msg.current_state == State.PATH_FORWARD_DONE:
            self.get_logger().info("Forward pass completed. Planning return path.")
            new_state = State()
            new_state.current_state = State.PATH_PLANNING_RETURN
            self.state_pub.pub(new_state)

        if msg.current_state == State.PATH_RETURN_DONE:
            self.get_logger().info("Trip completed! No more actions to take")

    def get_dist(self, pose_a, pose_b=(0,0)):
        dx = pose_a[0] - pose_b[0]
        dy = pose_a[1] - pose_b[1]

        return np.sqrt(dx**2 + dy**2)

    def park_timer(self):
        self.get_logger().info("Timer started ")
        current_time = self.get_clock().now().nanoseconds / 1e9 

        if current_time - self.park_start >= 5:
            self.get_logger().info("Times up ")
            self.timer.destroy()
            self.timer = None

            state = State()
            state.current_state = State.PATH_FOLLOWING_FORWARD
            self.state_pub.publish(state)
            
    # def repark_check(self, begin_time, wait_time):
    #     current_time = self.get_clock().now().nanoseconds / 1e9 

    #     if current_time - begin_time >= wait_time:
    #         return True
    #     return False    

    def traj_callback(self, msg):
        if self.state == State.PATH_PLANNING_FORWARD and not self.forward_started:
            self.get_logger().info("Forward path created! Publishing and following trajectory.")
            new_state = State()
            new_state.current_state = State.PATH_FOLLOWING_FORWARD
            self.state_pub.publish(new_state)

            # publish trajectory
            self.forward_started = True
            self.traj_pub.publish(msg)

        if self.state == State.PATH_PLANNING_RETURN and not self.return_started:
            self.get_logger().info("Return path created! Publishing and following trajectory.")
            new_state = State()
            new_state.current_state = State.PATH_FOLLOWING_RETURN
            self.state_pub.publish(new_state)

            # publish trajectory
            self.return_started = True
            self.traj_pub.publish(msg)
        

def main() -> None:
    rclpy.init()
    node = StateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
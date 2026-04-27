import rclpy
import numpy as np
from state_msgs.msg import State
from vs_msgs.msg import ObjectLocation, ObjectLocationArray
from rclpy.node import Node

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        # declare params
        self.declare_parameter('state_topic', '/state')
        self.declare_parameter('object_detect_topic', '/object_detect')

        # get params
        self.STATE_TOPIC = self.get_parameter('state_topic').get_parameter_value().string_value
        self.OBJECT_DETECT_TOPIC = self.get_parameter('object_detect_topic').get_parameter_value().string_value
        
        # publishers
        self.state_pub = self.create_publisher(State, self.STATE_TOPIC, 10)
        
        # subscribers
        self.object_detect_sub = self.create_subscription(ObjectLocationArray, self.OBJECT_DETECT_TOPIC, self.object_detect_callback)

    
    def object_detect_callback(self, msg):
        objects = msg.objects # list of objectlocation msg items
        detections = {} # dictionary with closest detection of labels (key) with value of closest distance
        
        for object in objects:
            label = object.label
            distance = np.sqrt(object.x**2 + object.y**2)
            if label not in detections or detections[label] < distance:
                detections[label] = distance
            
        detections = sorted(list(detections.items())) # closest object has highest priority
        closest = detections[0]
        if closest[0] == "person":
            new_state = State()
            new_state.current_state = State.pedestrian_stop
            
            self.state_pub.pub(new_state)

        if closest[0] == "traffic light":
            new_state = State()
            new_state.current_state = State.traffic_stop
            
            self.state_pub.pub(new_state)

        if closest[0] == "parking meter":
            new_state = State()
            new_state.current_state = State.parking_meter
            
            self.state_pub.pub(new_state)
 
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
import rclpy
from rclpy.node import Node

class NodeName(Node):
    def __init__(self):
        super().__init__('node_name')
        # declare params

        # get params
        
        # publishers

        # subscribers
        pass

    
def main() -> None:
    rclpy.init()
    node = NodeName()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
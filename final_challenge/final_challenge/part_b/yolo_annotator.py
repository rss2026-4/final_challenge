#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import torch

from state_msgs.msg import State
from vis_msgs.msg import ObjectLocationPixel, ObjectLocationPixelArray

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from typing import List
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Bounding box coordinates in the original image:
    x1: int
    y1: int
    x2: int
    y2: int


class YoloAnnotatorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_annotator")

        # Declare and get ROS parameters
        self.declare_parameter('object_detect_topic', '/object_detect_pixels')
        self.declare_parameter('state_topic', '/state')
        self.OBJECT_DETECT_TOPIC = self.get_parameter('object_detect_topic').get_parameter_value().string_value
        self.STATE_TOPIC = self.get_parameter('state_topic').get_parameter_value().string_value

        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value()
            .string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.iou_threshold = (
            self.declare_parameter("iou_threshold", 0.7)
            .get_parameter_value()
            .double_value
        )



        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.class_color_map = self.get_class_color_map()
        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in self.class_color_map
        ]

        self.get_logger().info(f"Running {self.model_name} on device {self.device}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        if self.allowed_cls:
            self.get_logger().info(f"You've chosen to keep these class IDs: {self.allowed_cls}")
        else:
            self.get_logger().warn("No allowed classes matched the model's class list.")

        # Create publisher and subscribers
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter('annotated_image_topic', "/yolo/annotated_image")
        
        self.IMAGE_TOPIC = self.get_parameter('image_topic').get_parameter_value().string_value
        self.ANNOTATED_IMAGE_TOPIC = self.get_parameter('annotated_image_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(Image, self.IMAGE_TOPIC, self.on_image, 10)
        self.pub = self.create_publisher(Image, self.ANNOTATED_IMAGE_TOPIC, 10)
        
        self.object_detection_pub = self.create_publisher(ObjectLocationPixelArray, self.OBJECT_DETECT_TOPIC, 10)
        self.state_sub = self.create_subscription(State, self.STATE_TOPIC, self.state_callback, 10)
        self.save_img = False
        self.num_imgs = 0

    def state_callback(self, msg):
        if msg.current_state == State.PARKED:
            self.save_img = True

        else:
            self.save_img = False


    def get_class_color_map(self) -> dict[str, tuple[int, int, int]]:
        """
        Return a dictionary mapping a list of COCO class names you want to keep
        to the detection BGR colors in the annotated image. COCO class names include
        "chair", "couch", "tv", "laptop", "dining table", and many more. The list
        of available classes can be found in `self.model.names`.
        """
        # TODO: Customize this dictionary for the lab. Choose a subset of
        #       COCO class names to detect and their corresponding colors
        #       in the annotated image.
        # self.get_logger().info(" %s " % self.model.names)
        return {
            "backpack": (255, 0, 0),
            "traffic light": (0, 255, 0),
        }

    def on_image(self, msg: Image) -> None:
        # Convert ROS -> OpenCV (BGR)
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Run YOLO inference
        try:
            results = self.model(
                bgr,
                classes=self.allowed_cls,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        # Convert results to Detection List
        dets = self.results_to_detections(results[0])

        # Draw detections on BGR image
        annotated = self.draw_detections(bgr, dets)
        if self.save_img and self.num_imgs < 2 and dets != []:
            cv2.imwrite(f"parking_meter_{self.num_imgs}.jpg", annotated)
            self.num_imgs += 1
            self.save_img = False
            self.get_logger().info("Image saved to parking_meter.jpg!")



        # Publish annotated BGR image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

    def results_to_detections(self, result) -> List[Detection]:
        """
        Convert an Ultralytics result into a Detection list.

        YOLOv11 outputs:
          result.boxes.xyxy: (N, 4) tensor
          result.boxes.conf: (N,) tensor
          result.boxes.cls:  (N,) tensor
        """
        detections = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        # Convert Torch tensors -> CPU numpy
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

        # TODO: Store YOLO outputs as Detections. Iterate through xyxy_np, conf_np, and cls_np
        #       to append a Detection with all its instance variables filled in to the
        #       detections List.
        #
        # Hint: use Python's zip keyword to iterate through the three arrays in a single for loop.
        # self.get_logger().info("running results to detection")

        object_img_locs = ObjectLocationPixelArray()

        for detection in zip(xyxy_np, conf_np, cls_np):            
            points = detection[0]
            confidence = detection[1]
            class_id = int(detection[2])
            class_name = self.model.names[class_id]
            x1 = int(points[0])
            y1 = int(points[1])
            x2 = int(points[2])
            y2 = int(points[3])

            new_detection = Detection(class_id=class_id, class_name=class_name, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)
            detections.append(new_detection)

            object_img_loc = ObjectLocationPixel()
            
            object_img_loc.u = float((x1 + x2)/2.0)
            object_img_loc.v = float((y2)) # may want to change this
            object_img_loc.label = class_name

            object_img_locs.objects.append(object_img_loc)
        
        self.object_detection_pub.publish(object_img_locs)
        return detections

    def draw_detections(
        self,
        bgr_image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:

        out_image = bgr_image.copy()

        for det in detections:
            # TODO: Get the bounding box for the detection
            # self.get_logger()
            top_left = (det.x1, det.y1)
            bot_right = (det.x2, det.y2)

            # TODO: Draw the bounding box around the detection to the output image.
            #       Use the colors you specified per class in `get_class_color_map`
            #       by accessing the self.class_color_map dictionary.
            #
            out_image = cv2.rectangle(out_image, top_left, bot_right, self.class_color_map[det.class_name], 2)
            # Hint: Use cv2's `rectangle` function to draw a rectangle on the annotated image.

            # TODO: Label the box with the class name and confidence.
            out_image = cv2.putText(out_image, f"{det.class_name}, conf: {det.confidence:.2f}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_color_map[det.class_name], 2)
            # Hint: Use cv2's `putText` function to put text on the annotated image.

        return out_image


def main() -> None:
    rclpy.init()
    node = YoloAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
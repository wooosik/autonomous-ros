#!/usr/bin/env python3
"""
YOLO Detector Node
Performs object detection using YOLOv8 on camera images
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.detection_image_pub = self.create_publisher(Image, '/detection_image', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', 'cpu')  # 'cpu' or 'cuda'

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.img_size = self.get_parameter('image_size').value
        self.device = self.get_parameter('device').value

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # YOLO model (placeholder - requires ultralytics package)
        self.model = None
        self.initialize_yolo_model()

        self.get_logger().info('YOLO Detector Node initialized')

    def initialize_yolo_model(self):
        """Initialize YOLO model"""
        try:
            # This requires: pip install ultralytics
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.get_logger().info(f'YOLO model loaded: {self.model_path}')
        except ImportError:
            self.get_logger().warn(
                'ultralytics package not installed. '
                'Install with: pip install ultralytics'
            )
            self.model = None
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            self.model = None

    def image_callback(self, msg):
        """Process incoming image and detect objects"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.model is None:
                # If model not loaded, just republish image with text
                cv2.putText(
                    cv_image,
                    'YOLO model not loaded',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                detection_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.detection_image_pub.publish(detection_image)
                return

            # Run YOLO inference
            results = self.model(
                cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )

            # Parse detections
            detections = self.parse_detections(results, msg.header)

            # Publish detections
            self.detection_pub.publish(detections)

            # Draw detections on image
            annotated_image = self.draw_detections(cv_image, results)

            # Publish annotated image
            detection_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            self.detection_image_pub.publish(detection_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def parse_detections(self, results, header):
        """Convert YOLO results to ROS Detection2DArray message"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = Detection2D()

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detection.bbox.center.position.x = float((x1 + x2) / 2)
                detection.bbox.center.position.y = float((y1 + y2) / 2)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                # Class and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(box.cls[0]))
                hypothesis.hypothesis.score = float(box.conf[0])

                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

        return detection_array

    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class and confidence
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f'{class_name}: {conf:.2f}'
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return annotated_image


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Webcam Publisher Node
Publishes images from laptop camera to ROS2 topics
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class WebcamPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        # Parameters
        self.declare_parameter('camera_id', 0)  # 0 for default webcam
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        self.camera_id = self.get_parameter('camera_id').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {self.camera_id}')
            raise RuntimeError(f'Cannot open camera {self.camera_id}')

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(f'Webcam opened: {actual_width}x{actual_height} @ {actual_fps} fps')

        # Create timer to publish frames
        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Camera info (basic calibration - should be replaced with actual calibration)
        self.camera_info_msg = self.create_camera_info(actual_width, actual_height)

        self.get_logger().info('Webcam Publisher Node initialized')

    def create_camera_info(self, width, height):
        """Create a basic camera info message"""
        msg = CameraInfo()
        msg.header.frame_id = 'camera_frame'
        msg.width = width
        msg.height = height

        # Simple camera matrix (assuming no distortion)
        # Focal length approximation: f = width (for ~60 degree FOV)
        fx = fy = width
        cx = width / 2.0
        cy = height / 2.0

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        # No distortion coefficients
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification matrix (identity)
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        # Projection matrix
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return msg

    def timer_callback(self):
        """Read and publish camera frame"""
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return

        try:
            # Create timestamp
            timestamp = self.get_clock().now().to_msg()

            # Publish image
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = 'camera_frame'
            self.image_pub.publish(img_msg)

            # Publish camera info
            self.camera_info_msg.header.stamp = timestamp
            self.camera_info_pub.publish(self.camera_info_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing image: {str(e)}')

    def destroy_node(self):
        """Cleanup resources"""
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

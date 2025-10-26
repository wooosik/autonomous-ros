#!/usr/bin/env python3
"""
Camera Processor Node
Handles camera image preprocessing and enhancement
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraProcessorNode(Node):
    def __init__(self):
        super().__init__('camera_processor_node')

        # Publishers
        self.processed_image_pub = self.create_publisher(Image, '/camera/image_processed', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Parameters
        self.declare_parameter('enable_undistortion', True)
        self.declare_parameter('enable_enhancement', True)
        self.declare_parameter('brightness_factor', 1.0)
        self.declare_parameter('contrast_factor', 1.0)
        self.declare_parameter('denoise', False)

        self.enable_undistortion = self.get_parameter('enable_undistortion').value
        self.enable_enhancement = self.get_parameter('enable_enhancement').value
        self.brightness_factor = self.get_parameter('brightness_factor').value
        self.contrast_factor = self.get_parameter('contrast_factor').value
        self.denoise = self.get_parameter('denoise').value

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        self.get_logger().info('Camera Processor Node initialized')

    def camera_info_callback(self, msg):
        """Update camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Apply undistortion
            if self.enable_undistortion and self.camera_matrix is not None:
                cv_image = cv2.undistort(
                    cv_image,
                    self.camera_matrix,
                    self.dist_coeffs
                )

            # Apply image enhancement
            if self.enable_enhancement:
                cv_image = self.enhance_image(cv_image)

            # Apply denoising
            if self.denoise:
                cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)

            # Convert back to ROS Image and publish
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.processed_image_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def enhance_image(self, image):
        """Apply brightness and contrast adjustment"""
        # Convert to float for processing
        enhanced = image.astype(np.float32)

        # Apply contrast and brightness
        enhanced = enhanced * self.contrast_factor + (self.brightness_factor - 1.0) * 128

        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255)

        return enhanced.astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('perception')
    model_dir = os.path.join(pkg_dir, 'models')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='yolov8n.pt',
            description='Path to YOLO model file'
        ),

        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for detections'
        ),

        DeclareLaunchArgument(
            'device',
            default_value='cpu',
            description='Device to run inference on (cpu or cuda)'
        ),

        # Camera processor node
        Node(
            package='perception',
            executable='camera_processor_node.py',
            name='camera_processor',
            output='screen',
            parameters=[{
                'enable_undistortion': True,
                'enable_enhancement': True,
                'brightness_factor': 1.0,
                'contrast_factor': 1.0,
                'denoise': False,
            }]
        ),

        # YOLO detector node
        Node(
            package='perception',
            executable='yolo_detector_node.py',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'iou_threshold': 0.45,
                'image_size': 640,
                'device': LaunchConfiguration('device'),
            }],
            remappings=[
                ('/camera/image_raw', '/camera/image_processed'),
            ]
        ),
    ])

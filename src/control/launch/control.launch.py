from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'control_frequency',
            default_value='20.0',
            description='Control loop frequency in Hz'
        ),

        DeclareLaunchArgument(
            'lookahead_distance',
            default_value='2.0',
            description='Lookahead distance for pure pursuit controller'
        ),

        Node(
            package='control',
            executable='trajectory_tracker_node.py',
            name='trajectory_tracker',
            output='screen',
            parameters=[{
                'control_frequency': LaunchConfiguration('control_frequency'),
                'lookahead_distance': LaunchConfiguration('lookahead_distance'),
                'max_linear_velocity': 1.0,
                'max_angular_velocity': 1.0,
            }]
        ),

        Node(
            package='control',
            executable='pid_controller_node.py',
            name='pid_controller',
            output='screen',
            parameters=[{
                'velocity_kp': 1.0,
                'velocity_ki': 0.1,
                'velocity_kd': 0.05,
                'steering_kp': 1.5,
                'steering_ki': 0.0,
                'steering_kd': 0.1,
                'control_frequency': 50.0,
            }]
        ),
    ])

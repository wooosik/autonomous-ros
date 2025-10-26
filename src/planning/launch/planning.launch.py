from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'planning_frequency',
            default_value='10.0',
            description='Planning update frequency in Hz'
        ),

        Node(
            package='planning',
            executable='path_planner_node.py',
            name='path_planner',
            output='screen',
            parameters=[{
                'planning_frequency': LaunchConfiguration('planning_frequency'),
            }]
        ),
    ])

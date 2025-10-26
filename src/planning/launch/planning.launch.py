from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'planning_frequency',
            default_value='1.0',
            description='Planning update frequency in Hz'
        ),

        DeclareLaunchArgument(
            'planning_algorithm',
            default_value='astar',
            description='Path planning algorithm: astar, rrt, rrt_star'
        ),

        DeclareLaunchArgument(
            'grid_resolution',
            default_value='0.1',
            description='Grid resolution in meters'
        ),

        DeclareLaunchArgument(
            'map_width',
            default_value='20.0',
            description='Map width in meters'
        ),

        DeclareLaunchArgument(
            'map_height',
            default_value='20.0',
            description='Map height in meters'
        ),

        DeclareLaunchArgument(
            'inflation_radius',
            default_value='0.5',
            description='Obstacle inflation radius in meters'
        ),

        Node(
            package='planning',
            executable='path_planner_node.py',
            name='path_planner',
            output='screen',
            parameters=[{
                'planning_frequency': LaunchConfiguration('planning_frequency'),
                'planning_algorithm': LaunchConfiguration('planning_algorithm'),
                'grid_resolution': LaunchConfiguration('grid_resolution'),
                'map_width': LaunchConfiguration('map_width'),
                'map_height': LaunchConfiguration('map_height'),
                'inflation_radius': LaunchConfiguration('inflation_radius'),
                'allow_diagonal': True,
                'smooth_path': True,
                'replan_threshold': 0.5,
            }]
        ),
    ])

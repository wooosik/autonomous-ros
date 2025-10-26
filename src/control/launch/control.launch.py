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
            'controller_type',
            default_value='stanley',
            description='Controller type: pure_pursuit, stanley, dwa'
        ),

        DeclareLaunchArgument(
            'lookahead_distance',
            default_value='2.0',
            description='Lookahead distance for pure pursuit controller'
        ),

        # Pure Pursuit Controller
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
            }],
            condition=lambda context: context.launch_configurations['controller_type'] == 'pure_pursuit'
        ),

        # Stanley Controller
        Node(
            package='control',
            executable='stanley_controller_node.py',
            name='stanley_controller',
            output='screen',
            parameters=[{
                'control_frequency': LaunchConfiguration('control_frequency'),
                'control_gain': 2.5,
                'softening_gain': 1.0,
                'max_steer': 0.6,
                'wheelbase': 0.5,
                'max_linear_velocity': 1.0,
                'min_linear_velocity': 0.1,
                'max_angular_velocity': 1.0,
                'goal_tolerance': 0.2,
                'path_lookahead': 5,
            }],
            condition=lambda context: context.launch_configurations['controller_type'] == 'stanley'
        ),

        # DWA Planner (Local Planner + Controller)
        Node(
            package='control',
            executable='dwa_planner_node.py',
            name='dwa_planner',
            output='screen',
            parameters=[{
                'control_frequency': 10.0,
                'goal_tolerance': 0.2,
                'max_speed': 1.0,
                'robot_radius': 0.3,
                'predict_time': 2.0,
            }],
            condition=lambda context: context.launch_configurations['controller_type'] == 'dwa'
        ),

        # PID Controller (optional, can run alongside other controllers)
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

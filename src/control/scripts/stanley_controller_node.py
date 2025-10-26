#!/usr/bin/env python3
"""
Stanley Controller Node
Implements Stanley lateral control algorithm for path tracking
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
import math
import numpy as np


class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Parameters
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('control_gain', 2.5)  # k parameter in Stanley
        self.declare_parameter('softening_gain', 1.0)  # k_s parameter (velocity dependent)
        self.declare_parameter('max_steer', 0.6)  # Max steering angle in radians
        self.declare_parameter('wheelbase', 0.5)  # Wheelbase length in meters
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('min_linear_velocity', 0.1)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('path_lookahead', 5)  # Number of points to look ahead

        self.control_freq = self.get_parameter('control_frequency').value
        self.control_gain = self.get_parameter('control_gain').value
        self.softening_gain = self.get_parameter('softening_gain').value
        self.max_steer = self.get_parameter('max_steer').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.min_linear_vel = self.get_parameter('min_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_lookahead = self.get_parameter('path_lookahead').value

        # State variables
        self.current_pose = None
        self.current_path = None
        self.current_waypoint_idx = 0
        self.goal_reached = False

        # Timer for control loop
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)

        self.get_logger().info('Stanley Controller Node initialized')
        self.get_logger().info(f'  - Control gain (k): {self.control_gain}')
        self.get_logger().info(f'  - Wheelbase: {self.wheelbase}m')
        self.get_logger().info(f'  - Max steering angle: {self.max_steer:.2f} rad')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def path_callback(self, msg):
        """Receive new path to track"""
        if len(msg.poses) == 0:
            return

        self.current_path = msg
        self.current_waypoint_idx = 0
        self.goal_reached = False
        self.get_logger().info(f'New path received with {len(msg.poses)} waypoints')

    def control_loop(self):
        """Main control loop using Stanley algorithm"""
        if self.current_pose is None or self.current_path is None:
            return

        if len(self.current_path.poses) == 0:
            self.stop()
            return

        # Check if goal is reached
        goal_pose = self.current_path.poses[-1].pose
        distance_to_goal = self.distance_between_poses(self.current_pose, goal_pose)

        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info('Goal reached!')
                self.goal_reached = True
            self.stop()
            return

        # Find closest point on path
        closest_idx = self.find_closest_point()
        if closest_idx is None:
            self.stop()
            return

        # Update current waypoint index
        self.current_waypoint_idx = min(
            closest_idx + self.path_lookahead,
            len(self.current_path.poses) - 1
        )

        # Calculate Stanley control
        cmd_vel = self.stanley_control(closest_idx)
        self.cmd_vel_pub.publish(cmd_vel)

    def find_closest_point(self):
        """Find the closest point on the path to current position"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return None

        min_dist = float('inf')
        closest_idx = 0

        # Search from current waypoint to end
        search_start = max(0, self.current_waypoint_idx - 5)

        for i in range(search_start, len(self.current_path.poses)):
            pose = self.current_path.poses[i].pose
            dist = self.distance_between_poses(self.current_pose, pose)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def stanley_control(self, closest_idx):
        """
        Calculate control commands using Stanley algorithm

        The Stanley method consists of three components:
        1. Heading error: difference between vehicle heading and path heading
        2. Cross-track error: perpendicular distance from vehicle to path
        3. Steering angle: combination of both errors
        """
        cmd_vel = Twist()

        if closest_idx >= len(self.current_path.poses):
            return cmd_vel

        # Get closest point on path
        closest_pose = self.current_path.poses[closest_idx].pose

        # Get current heading
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Calculate path heading at closest point
        if closest_idx < len(self.current_path.poses) - 1:
            next_pose = self.current_path.poses[closest_idx + 1].pose
            path_yaw = math.atan2(
                next_pose.position.y - closest_pose.position.y,
                next_pose.position.x - closest_pose.position.x
            )
        else:
            path_yaw = self.get_yaw_from_quaternion(closest_pose.orientation)

        # 1. Heading error
        heading_error = self.normalize_angle(path_yaw - current_yaw)

        # 2. Cross-track error
        # Calculate vector from closest point to vehicle
        dx = self.current_pose.position.x - closest_pose.position.x
        dy = self.current_pose.position.y - closest_pose.position.y

        # Calculate cross-track error (perpendicular distance)
        # Positive if vehicle is to the left of path, negative if to the right
        cross_track_error = -dx * math.sin(path_yaw) + dy * math.cos(path_yaw)

        # Calculate desired velocity (reduce speed for sharp turns)
        turn_sharpness = abs(heading_error) / math.pi
        desired_velocity = self.max_linear_vel * (1.0 - 0.5 * turn_sharpness)
        desired_velocity = max(self.min_linear_vel, desired_velocity)

        # 3. Stanley steering control
        # arctan term for cross-track error correction
        velocity = max(desired_velocity, 0.1)  # Avoid division by zero
        cross_track_term = math.atan2(
            self.control_gain * cross_track_error,
            self.softening_gain + velocity
        )

        # Total steering angle
        steering_angle = heading_error + cross_track_term

        # Limit steering angle
        steering_angle = max(min(steering_angle, self.max_steer), -self.max_steer)

        # Convert steering angle to angular velocity
        # For differential drive: w = v * tan(delta) / L
        # For car-like: w = v * sin(delta) / L (approximation for small angles)
        if abs(steering_angle) > 0.01:
            angular_velocity = desired_velocity * math.tan(steering_angle) / self.wheelbase
        else:
            angular_velocity = 0.0

        # Limit angular velocity
        angular_velocity = max(
            min(angular_velocity, self.max_angular_vel),
            -self.max_angular_vel
        )

        # Set command velocities
        cmd_vel.linear.x = desired_velocity
        cmd_vel.angular.z = angular_velocity

        # Debug logging (throttled)
        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Log every ~1 second
            self.get_logger().debug(
                f'Stanley Control - '
                f'Heading Error: {heading_error:.3f}rad, '
                f'Cross-track Error: {cross_track_error:.3f}m, '
                f'Steering: {steering_angle:.3f}rad, '
                f'V: {desired_velocity:.2f}m/s, '
                f'W: {angular_velocity:.2f}rad/s'
            )

        return cmd_vel

    def distance_between_poses(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx * dx + dy * dy)

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def stop(self):
        """Send stop command"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

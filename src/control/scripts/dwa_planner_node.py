#!/usr/bin/env python3
"""
DWA (Dynamic Window Approach) Local Planner Node
Implements local path planning with dynamic obstacle avoidance
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from typing import List, Tuple


class DWAConfig:
    """Configuration for DWA algorithm"""

    def __init__(self):
        # Robot specifications
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 1.0  # [rad/s]
        self.max_accel = 0.5  # [m/s^2]
        self.max_delta_yaw_rate = 1.0  # [rad/s^2]

        # Velocity resolution
        self.v_resolution = 0.1  # [m/s]
        self.yaw_rate_resolution = 0.2  # [rad/s]

        # Prediction time and resolution
        self.predict_time = 2.0  # [s]
        self.dt = 0.1  # [s]

        # Cost function weights
        self.heading_cost_gain = 1.0
        self.clearance_cost_gain = 2.0
        self.velocity_cost_gain = 0.5

        # Robot radius
        self.robot_radius = 0.3  # [m]

        # Obstacle distance threshold
        self.obstacle_threshold = 2.0  # [m]


class DWAPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_planner_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.predicted_path_pub = self.create_publisher(Path, '/dwa_predicted_path', 10)

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

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Parameters
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('predict_time', 2.0)

        control_freq = self.get_parameter('control_frequency').value
        goal_tolerance = self.get_parameter('goal_tolerance').value

        # DWA configuration
        self.config = DWAConfig()
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.robot_radius = self.get_parameter('robot_radius').value
        self.config.predict_time = self.get_parameter('predict_time').value
        self.goal_tolerance = goal_tolerance

        # State variables
        self.current_pose = None
        self.current_velocity = Twist()
        self.current_path = None
        self.obstacles = []
        self.goal_reached = False

        # Timer for control loop
        self.timer = self.create_timer(1.0 / control_freq, self.control_loop)

        self.get_logger().info('DWA Planner Node initialized')
        self.get_logger().info(f'  - Max speed: {self.config.max_speed}m/s')
        self.get_logger().info(f'  - Robot radius: {self.config.robot_radius}m')
        self.get_logger().info(f'  - Prediction time: {self.config.predict_time}s')

    def odom_callback(self, msg):
        """Update current pose and velocity from odometry"""
        self.current_pose = msg.pose.pose
        self.current_velocity.linear.x = msg.twist.twist.linear.x
        self.current_velocity.angular.z = msg.twist.twist.angular.z

    def path_callback(self, msg):
        """Receive new path to track"""
        if len(msg.poses) == 0:
            return

        self.current_path = msg
        self.goal_reached = False
        self.get_logger().info(f'New path received with {len(msg.poses)} waypoints')

    def laser_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        if self.current_pose is None:
            return

        # Convert laser scan to obstacle points in robot frame
        obstacles = []
        angle = msg.angle_min

        for distance in msg.ranges:
            if msg.range_min < distance < msg.range_max:
                # Convert to Cartesian coordinates (robot frame)
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                obstacles.append([x, y])

            angle += msg.angle_increment

        self.obstacles = np.array(obstacles) if obstacles else np.array([])

    def control_loop(self):
        """Main control loop using DWA algorithm"""
        if self.current_pose is None or self.current_path is None:
            return

        if len(self.current_path.poses) == 0:
            self.stop()
            return

        # Check if goal is reached
        goal_pose = self.current_path.poses[-1].pose
        distance_to_goal = self.distance_to_pose(goal_pose)

        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info('Goal reached!')
                self.goal_reached = True
            self.stop()
            return

        # Get target point from path (lookahead)
        target_pose = self.get_lookahead_point()
        if target_pose is None:
            self.stop()
            return

        # Run DWA algorithm
        best_v, best_w, best_trajectory = self.dwa_control(target_pose)

        # Publish command velocity
        cmd_vel = Twist()
        cmd_vel.linear.x = best_v
        cmd_vel.angular.z = best_w
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish predicted trajectory for visualization
        if best_trajectory is not None:
            self.publish_predicted_path(best_trajectory)

    def get_lookahead_point(self):
        """Get lookahead point from path"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return None

        # Find closest point
        min_dist = float('inf')
        closest_idx = 0

        for i, pose_stamped in enumerate(self.current_path.poses):
            dist = self.distance_to_pose(pose_stamped.pose)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Lookahead
        lookahead_idx = min(closest_idx + 10, len(self.current_path.poses) - 1)
        return self.current_path.poses[lookahead_idx].pose

    def dwa_control(self, target_pose):
        """
        DWA algorithm implementation

        Returns:
            best_v, best_w, best_trajectory
        """
        # Current state
        current_v = self.current_velocity.linear.x
        current_w = self.current_velocity.angular.z

        # Calculate dynamic window
        dw = self.calculate_dynamic_window(current_v, current_w)

        # Evaluate all velocity combinations in dynamic window
        best_cost = float('inf')
        best_v = 0.0
        best_w = 0.0
        best_trajectory = None

        # Sample velocities
        for v in np.arange(dw[0], dw[1], self.config.v_resolution):
            for w in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):
                # Predict trajectory
                trajectory = self.predict_trajectory(v, w)

                # Calculate cost
                heading_cost = self.calc_heading_cost(trajectory, target_pose)
                clearance_cost = self.calc_clearance_cost(trajectory)
                velocity_cost = self.calc_velocity_cost(v)

                total_cost = (
                    self.config.heading_cost_gain * heading_cost +
                    self.config.clearance_cost_gain * clearance_cost +
                    self.config.velocity_cost_gain * velocity_cost
                )

                # Update best trajectory
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_w = w
                    best_trajectory = trajectory

        return best_v, best_w, best_trajectory

    def calculate_dynamic_window(self, current_v, current_w):
        """
        Calculate dynamic window based on current velocity and robot constraints

        Returns:
            [v_min, v_max, w_min, w_max]
        """
        # Dynamic window based on current velocity and acceleration limits
        v_min = max(
            self.config.min_speed,
            current_v - self.config.max_accel * self.config.dt
        )
        v_max = min(
            self.config.max_speed,
            current_v + self.config.max_accel * self.config.dt
        )

        w_min = max(
            -self.config.max_yaw_rate,
            current_w - self.config.max_delta_yaw_rate * self.config.dt
        )
        w_max = min(
            self.config.max_yaw_rate,
            current_w + self.config.max_delta_yaw_rate * self.config.dt
        )

        return [v_min, v_max, w_min, w_max]

    def predict_trajectory(self, v, w):
        """
        Predict robot trajectory given velocity commands

        Returns:
            trajectory: List of (x, y, yaw) positions
        """
        trajectory = []

        # Current state (robot frame)
        x = 0.0
        y = 0.0
        yaw = 0.0

        time = 0.0
        while time <= self.config.predict_time:
            # Append current state
            trajectory.append([x, y, yaw])

            # Update state
            x += v * math.cos(yaw) * self.config.dt
            y += v * math.sin(yaw) * self.config.dt
            yaw += w * self.config.dt

            time += self.config.dt

        return np.array(trajectory)

    def calc_heading_cost(self, trajectory, target_pose):
        """Calculate heading cost (alignment with goal)"""
        if len(trajectory) == 0:
            return float('inf')

        # Get final position in trajectory
        final_x = trajectory[-1, 0]
        final_y = trajectory[-1, 1]

        # Calculate target position in robot frame
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y

        # Transform to robot frame
        target_x = dx * math.cos(-current_yaw) - dy * math.sin(-current_yaw)
        target_y = dx * math.sin(-current_yaw) + dy * math.cos(-current_yaw)

        # Calculate angle to target
        angle_to_target = math.atan2(target_y - final_y, target_x - final_x)

        # Cost is angle difference
        cost = abs(angle_to_target)

        return cost

    def calc_clearance_cost(self, trajectory):
        """Calculate clearance cost (distance to obstacles)"""
        if len(self.obstacles) == 0:
            return 0.0

        min_distance = float('inf')

        # Check all points in trajectory
        for point in trajectory:
            x, y = point[0], point[1]

            # Calculate distance to all obstacles
            if len(self.obstacles) > 0:
                distances = np.sqrt(
                    (self.obstacles[:, 0] - x)**2 +
                    (self.obstacles[:, 1] - y)**2
                )
                min_dist_to_obstacle = np.min(distances)

                # Check collision
                if min_dist_to_obstacle < self.config.robot_radius:
                    return float('inf')  # Collision

                min_distance = min(min_distance, min_dist_to_obstacle)

        # Cost is inverse of clearance
        if min_distance < self.config.obstacle_threshold:
            cost = 1.0 / (min_distance + 0.1)
        else:
            cost = 0.0

        return cost

    def calc_velocity_cost(self, v):
        """Calculate velocity cost (prefer higher speeds)"""
        # Normalize velocity to [0, 1]
        normalized_v = v / self.config.max_speed

        # Cost is inverse (prefer higher speeds)
        cost = 1.0 - normalized_v

        return cost

    def publish_predicted_path(self, trajectory):
        """Publish predicted trajectory for visualization"""
        if trajectory is None or len(trajectory) == 0:
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'base_link'

        for point in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0

            # Set orientation
            yaw = point[2]
            pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)

            path.poses.append(pose_stamped)

        self.predicted_path_pub.publish(path)

    def distance_to_pose(self, pose):
        """Calculate distance to a pose"""
        dx = pose.position.x - self.current_pose.position.x
        dy = pose.position.y - self.current_pose.position.y
        return math.sqrt(dx * dx + dy * dy)

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop(self):
        """Send stop command"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = DWAPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

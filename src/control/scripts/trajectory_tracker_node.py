#!/usr/bin/env python3
"""
Trajectory Tracker Node
Tracks a planned path using Pure Pursuit or Stanley controller
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
import math


class TrajectoryTrackerNode(Node):
    def __init__(self):
        super().__init__('trajectory_tracker_node')

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
        self.declare_parameter('lookahead_distance', 2.0)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 1.0)

        self.control_freq = self.get_parameter('control_frequency').value
        self.lookahead_dist = self.get_parameter('lookahead_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value

        # State variables
        self.current_pose = None
        self.current_path = None
        self.current_waypoint_idx = 0

        # Timer for control loop
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)

        self.get_logger().info('Trajectory Tracker Node initialized')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def path_callback(self, msg):
        """Receive new path to track"""
        self.current_path = msg
        self.current_waypoint_idx = 0
        self.get_logger().info(f'New path received with {len(msg.poses)} waypoints')

    def control_loop(self):
        """Main control loop - Pure Pursuit algorithm"""
        if self.current_pose is None or self.current_path is None:
            return

        if len(self.current_path.poses) == 0:
            self.stop()
            return

        # Find lookahead point
        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            self.stop()
            return

        # Calculate control commands
        cmd_vel = self.pure_pursuit_control(lookahead_point)
        self.cmd_vel_pub.publish(cmd_vel)

    def find_lookahead_point(self):
        """Find the lookahead point on the path"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return None

        # Simple implementation: find closest point beyond lookahead distance
        min_dist = float('inf')
        lookahead_point = None

        for i in range(self.current_waypoint_idx, len(self.current_path.poses)):
            pose = self.current_path.poses[i].pose
            dist = self.distance_to_point(pose.position)

            if dist >= self.lookahead_dist:
                lookahead_point = pose
                self.current_waypoint_idx = i
                break

        return lookahead_point if lookahead_point else self.current_path.poses[-1].pose

    def pure_pursuit_control(self, target_pose):
        """Calculate control commands using Pure Pursuit"""
        cmd_vel = Twist()

        # Calculate distance and angle to target
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y

        # Calculate heading error
        target_heading = math.atan2(dy, dx)
        current_heading = self.get_yaw_from_quaternion(self.current_pose.orientation)
        heading_error = target_heading - current_heading

        # Normalize angle to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Calculate angular velocity
        cmd_vel.angular.z = max(min(2.0 * heading_error, self.max_angular_vel), -self.max_angular_vel)

        # Calculate linear velocity (reduce speed when turning)
        cmd_vel.linear.x = self.max_linear_vel * (1.0 - abs(heading_error) / math.pi)

        return cmd_vel

    def distance_to_point(self, point):
        """Calculate distance from current pose to a point"""
        dx = point.x - self.current_pose.position.x
        dy = point.y - self.current_pose.position.y
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
    node = TrajectoryTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

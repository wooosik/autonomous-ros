#!/usr/bin/env python3
"""
Path Planner Node
Implements path planning algorithms for autonomous navigation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Parameters
        self.declare_parameter('planning_frequency', 10.0)
        self.planning_freq = self.get_parameter('planning_frequency').value

        # State variables
        self.current_pose = None
        self.goal_pose = None

        # Timer for path planning
        self.timer = self.create_timer(1.0 / self.planning_freq, self.plan_path)

        self.get_logger().info('Path Planner Node initialized')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """Receive new goal pose"""
        self.goal_pose = msg.pose
        self.get_logger().info(f'New goal received: x={msg.pose.position.x}, y={msg.pose.position.y}')

    def plan_path(self):
        """Main path planning logic"""
        if self.current_pose is None or self.goal_pose is None:
            return

        # TODO: Implement path planning algorithm (A*, RRT, etc.)
        # For now, create a simple straight-line path
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        # Add waypoints (simplified example)
        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose = self.current_pose

        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose = self.goal_pose

        path.poses = [start_pose, goal_pose]

        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

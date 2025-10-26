#!/usr/bin/env python3
"""
Path Planner Node
Implements path planning algorithms for autonomous navigation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.dirname(__file__))

from algorithms.a_star import AStarPlanner, smooth_path
from algorithms.rrt import RRTPlanner, RRTStarPlanner
from costmap.costmap_generator import CostmapGenerator


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # Parameters
        self.declare_parameter('planning_frequency', 1.0)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('map_width', 20.0)
        self.declare_parameter('map_height', 20.0)
        self.declare_parameter('inflation_radius', 0.5)
        self.declare_parameter('allow_diagonal', True)
        self.declare_parameter('smooth_path', True)
        self.declare_parameter('replan_threshold', 0.5)
        self.declare_parameter('planning_algorithm', 'astar')  # 'astar', 'rrt', 'rrt_star'

        self.planning_freq = self.get_parameter('planning_frequency').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.allow_diagonal = self.get_parameter('allow_diagonal').value
        self.smooth_path_enabled = self.get_parameter('smooth_path').value
        self.replan_threshold = self.get_parameter('replan_threshold').value
        self.planning_algorithm = self.get_parameter('planning_algorithm').value

        # Initialize planners based on selected algorithm
        self.planners = {
            'astar': AStarPlanner(
                grid_resolution=self.grid_resolution,
                allow_diagonal=self.allow_diagonal
            ),
            'rrt': RRTPlanner(
                step_size=0.5,
                max_iterations=5000,
                goal_sample_rate=0.1,
                goal_tolerance=0.5,
                grid_resolution=self.grid_resolution
            ),
            'rrt_star': RRTStarPlanner(
                step_size=0.5,
                max_iterations=5000,
                goal_sample_rate=0.1,
                goal_tolerance=0.5,
                grid_resolution=self.grid_resolution,
                search_radius=1.0
            )
        }

        if self.planning_algorithm not in self.planners:
            self.get_logger().warn(
                f'Unknown planning algorithm: {self.planning_algorithm}, using A*'
            )
            self.planning_algorithm = 'astar'

        self.planner = self.planners[self.planning_algorithm]

        # Initialize costmap
        self.costmap_generator = CostmapGenerator(
            width=self.map_width,
            height=self.map_height,
            resolution=self.grid_resolution,
            origin=(-self.map_width / 2, -self.map_height / 2),
            inflation_radius=self.inflation_radius
        )

        # Add some example obstacles (will be replaced with real obstacle detection)
        self._initialize_example_obstacles()

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', 10)

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

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # State variables
        self.current_pose = None
        self.goal_pose = None
        self.current_path = None
        self.last_goal_position = None

        # Timer for path planning
        self.timer = self.create_timer(1.0 / self.planning_freq, self.plan_path)

        # Timer for costmap publishing
        self.costmap_timer = self.create_timer(1.0, self.publish_costmap)

        self.get_logger().info('Path Planner Node initialized')
        self.get_logger().info(f'  - Grid resolution: {self.grid_resolution}m')
        self.get_logger().info(f'  - Map size: {self.map_width}x{self.map_height}m')
        self.get_logger().info(f'  - Inflation radius: {self.inflation_radius}m')
        self.get_logger().info(f'  - Planning algorithm: {self.planning_algorithm.upper()}')

    def _initialize_example_obstacles(self):
        """Initialize some example static obstacles"""
        # Add some walls or obstacles for testing
        # Rectangle obstacle
        self.costmap_generator.add_rectangle_obstacle(
            x_min=-2.0, y_min=-1.0, x_max=-1.0, y_max=1.0, static=True
        )

        # Circular obstacle
        self.costmap_generator.add_obstacle(x=2.0, y=2.0, radius=0.5, static=True)

        # Update costmap with inflation
        self.costmap_generator.update_costmap()

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """Receive new goal pose"""
        self.goal_pose = msg.pose
        self.get_logger().info(f'New goal received: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}')

        # Clear last goal position to force replanning
        self.last_goal_position = None

    def laser_callback(self, msg):
        """Process laser scan data to update dynamic obstacles"""
        if self.current_pose is None:
            return

        # Clear dynamic obstacles
        self.costmap_generator.clear_dynamic_obstacles()

        # Convert laser scan to obstacles
        angle = msg.angle_min
        for i, distance in enumerate(msg.ranges):
            if msg.range_min < distance < msg.range_max:
                # Calculate obstacle position in world frame
                current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
                obstacle_x = self.current_pose.position.x + distance * np.cos(angle + current_yaw)
                obstacle_y = self.current_pose.position.y + distance * np.sin(angle + current_yaw)

                # Add to dynamic layer
                self.costmap_generator.add_obstacle(
                    x=obstacle_x,
                    y=obstacle_y,
                    radius=0.1,
                    static=False
                )

            angle += msg.angle_increment

        # Update costmap
        self.costmap_generator.update_costmap()

    def plan_path(self):
        """Main path planning logic using A* algorithm"""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Check if we need to replan
        current_goal = (self.goal_pose.position.x, self.goal_pose.position.y)

        if self.last_goal_position is not None:
            # Check if goal has changed significantly
            goal_distance = np.sqrt(
                (current_goal[0] - self.last_goal_position[0])**2 +
                (current_goal[1] - self.last_goal_position[1])**2
            )

            if goal_distance < self.replan_threshold and self.current_path is not None:
                # Goal hasn't changed much, no need to replan
                return

        self.last_goal_position = current_goal

        # Extract start and goal positions
        start = (self.current_pose.position.x, self.current_pose.position.y)
        goal = (self.goal_pose.position.x, self.goal_pose.position.y)

        self.get_logger().info(f'Planning path from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f})')

        # Get binary costmap for planning
        costmap = self.costmap_generator.get_binary_costmap()

        # Plan path using A*
        waypoints = self.planner.plan(
            start=start,
            goal=goal,
            costmap=costmap,
            origin=self.costmap_generator.origin
        )

        if waypoints is None:
            self.get_logger().warn('Failed to find path to goal')
            return

        # Smooth path if enabled
        if self.smooth_path_enabled and len(waypoints) > 2:
            waypoints = smooth_path(waypoints, weight_data=0.5, weight_smooth=0.3)

        # Convert waypoints to Path message
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        for waypoint in waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = waypoint[0]
            pose_stamped.pose.position.y = waypoint[1]
            pose_stamped.pose.position.z = 0.0

            # Set orientation towards next waypoint
            idx = waypoints.index(waypoint)
            if idx < len(waypoints) - 1:
                next_waypoint = waypoints[idx + 1]
                yaw = np.arctan2(
                    next_waypoint[1] - waypoint[1],
                    next_waypoint[0] - waypoint[0]
                )
                pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)
            else:
                # Last waypoint uses goal orientation
                pose_stamped.pose.orientation = self.goal_pose.orientation

            path.poses.append(pose_stamped)

        self.current_path = path
        self.path_pub.publish(path)
        self.get_logger().info(f'Published path with {len(path.poses)} waypoints')

    def publish_costmap(self):
        """Publish costmap as OccupancyGrid for visualization"""
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = 'map'

        costmap_msg.info.resolution = self.grid_resolution
        costmap_msg.info.width = self.costmap_generator.grid_width
        costmap_msg.info.height = self.costmap_generator.grid_height
        costmap_msg.info.origin.position.x = self.costmap_generator.origin[0]
        costmap_msg.info.origin.position.y = self.costmap_generator.origin[1]
        costmap_msg.info.origin.position.z = 0.0
        costmap_msg.info.origin.orientation.w = 1.0

        # Convert costmap to OccupancyGrid format (0-100, -1 for unknown)
        costmap = self.costmap_generator.get_costmap()
        # Scale from 0-255 to 0-100
        occupancy_data = (costmap * 100 / 255).astype(np.int8)
        costmap_msg.data = occupancy_data.flatten().tolist()

        self.costmap_pub.publish(costmap_msg)

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q


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

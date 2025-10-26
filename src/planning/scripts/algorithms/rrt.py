#!/usr/bin/env python3
"""
RRT (Rapidly-exploring Random Tree) Path Planning Algorithm
Sampling-based path planning for complex environments
"""

import numpy as np
import math
from typing import List, Tuple, Optional


class Node:
    """Node class for RRT tree"""

    def __init__(self, position: Tuple[float, float]):
        self.position = position  # (x, y)
        self.parent = None
        self.cost = 0.0


class RRTPlanner:
    """RRT path planning algorithm implementation"""

    def __init__(
        self,
        step_size: float = 0.5,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        goal_tolerance: float = 0.5,
        grid_resolution: float = 0.1
    ):
        """
        Initialize RRT planner

        Args:
            step_size: Maximum distance to extend tree in each iteration
            max_iterations: Maximum number of iterations
            goal_sample_rate: Probability of sampling goal (0.0-1.0)
            goal_tolerance: Distance to goal to consider path found
            grid_resolution: Resolution for collision checking
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.grid_resolution = grid_resolution

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        costmap: np.ndarray,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using RRT algorithm

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            costmap: 2D numpy array representing obstacles (0=free, 1=occupied)
            origin: Origin of the costmap in world coordinates

        Returns:
            List of waypoints (x, y) in world coordinates, or None if no path found
        """
        # Check if start and goal are valid
        if not self.is_valid_world_position(start, costmap, origin):
            print(f"Start position {start} is invalid or occupied")
            return None

        if not self.is_valid_world_position(goal, costmap, origin):
            print(f"Goal position {goal} is invalid or occupied")
            return None

        # Initialize tree with start node
        start_node = Node(start)
        tree = [start_node]

        # Store costmap and origin
        self.costmap = costmap
        self.origin = origin

        # Main RRT loop
        for i in range(self.max_iterations):
            # Sample random point (or goal with probability)
            if np.random.random() < self.goal_sample_rate:
                sample_point = goal
            else:
                sample_point = self.sample_random_point(costmap, origin)

            # Find nearest node in tree
            nearest_node = self.find_nearest_node(tree, sample_point)

            # Extend tree towards sample point
            new_node = self.extend_tree(nearest_node, sample_point)

            # Check if new node is valid (collision-free)
            if new_node is None:
                continue

            if not self.is_collision_free(nearest_node.position, new_node.position):
                continue

            # Add new node to tree
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + self.distance(
                nearest_node.position, new_node.position
            )
            tree.append(new_node)

            # Check if goal is reached
            if self.distance(new_node.position, goal) <= self.goal_tolerance:
                print(f"Path found in {i+1} iterations")
                # Construct path from goal to start
                path = self.extract_path(new_node)
                return path

        print(f"Failed to find path after {self.max_iterations} iterations")
        return None

    def sample_random_point(
        self,
        costmap: np.ndarray,
        origin: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Sample random point in the map

        Args:
            costmap: Costmap array
            origin: Origin of the costmap

        Returns:
            Random point (x, y) in world coordinates
        """
        rows, cols = costmap.shape

        # Random grid position
        rand_row = np.random.randint(0, rows)
        rand_col = np.random.randint(0, cols)

        # Convert to world coordinates
        x = rand_col * self.grid_resolution + origin[0]
        y = rand_row * self.grid_resolution + origin[1]

        return (x, y)

    def find_nearest_node(
        self,
        tree: List[Node],
        point: Tuple[float, float]
    ) -> Node:
        """
        Find nearest node in tree to given point

        Args:
            tree: List of nodes in the tree
            point: Target point

        Returns:
            Nearest node
        """
        distances = [self.distance(node.position, point) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    def extend_tree(
        self,
        from_node: Node,
        to_point: Tuple[float, float]
    ) -> Optional[Node]:
        """
        Extend tree from node towards point

        Args:
            from_node: Node to extend from
            to_point: Point to extend towards

        Returns:
            New node, or None if invalid
        """
        # Calculate direction
        dx = to_point[0] - from_node.position[0]
        dy = to_point[1] - from_node.position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return None

        # Normalize direction
        dx /= distance
        dy /= distance

        # Calculate new position (limit to step_size)
        step = min(self.step_size, distance)
        new_x = from_node.position[0] + dx * step
        new_y = from_node.position[1] + dy * step

        # Check if new position is valid
        if not self.is_valid_world_position((new_x, new_y), self.costmap, self.origin):
            return None

        new_node = Node((new_x, new_y))
        return new_node

    def is_collision_free(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> bool:
        """
        Check if path between two positions is collision-free

        Args:
            pos1: Start position
            pos2: End position

        Returns:
            True if collision-free, False otherwise
        """
        # Check multiple points along the line
        num_checks = int(self.distance(pos1, pos2) / (self.grid_resolution * 0.5)) + 1

        for i in range(num_checks + 1):
            t = i / max(num_checks, 1)
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])

            if not self.is_valid_world_position((x, y), self.costmap, self.origin):
                return False

        return True

    def is_valid_world_position(
        self,
        pos: Tuple[float, float],
        costmap: np.ndarray,
        origin: Tuple[float, float]
    ) -> bool:
        """
        Check if world position is valid (within bounds and not occupied)

        Args:
            pos: Position in world coordinates
            costmap: Costmap array
            origin: Origin of the costmap

        Returns:
            True if valid, False otherwise
        """
        # Convert to grid coordinates
        grid_x = int((pos[0] - origin[0]) / self.grid_resolution)
        grid_y = int((pos[1] - origin[1]) / self.grid_resolution)

        rows, cols = costmap.shape

        # Check bounds
        if grid_y < 0 or grid_y >= rows or grid_x < 0 or grid_x >= cols:
            return False

        # Check if occupied
        if costmap[grid_y, grid_x] > 0:
            return False

        return True

    def extract_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """
        Extract path from goal node to start node

        Args:
            goal_node: Goal node

        Returns:
            List of waypoints from start to goal
        """
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        # Reverse to get start to goal
        path.reverse()
        return path

    def distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)


class RRTStarPlanner(RRTPlanner):
    """RRT* (RRT-Star) - Asymptotically optimal version of RRT"""

    def __init__(
        self,
        step_size: float = 0.5,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        goal_tolerance: float = 0.5,
        grid_resolution: float = 0.1,
        search_radius: float = 1.0
    ):
        """
        Initialize RRT* planner

        Args:
            search_radius: Radius for rewiring tree
        """
        super().__init__(
            step_size, max_iterations, goal_sample_rate,
            goal_tolerance, grid_resolution
        )
        self.search_radius = search_radius

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        costmap: np.ndarray,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using RRT* algorithm

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            costmap: 2D numpy array representing obstacles (0=free, 1=occupied)
            origin: Origin of the costmap in world coordinates

        Returns:
            List of waypoints (x, y) in world coordinates, or None if no path found
        """
        # Check if start and goal are valid
        if not self.is_valid_world_position(start, costmap, origin):
            print(f"Start position {start} is invalid or occupied")
            return None

        if not self.is_valid_world_position(goal, costmap, origin):
            print(f"Goal position {goal} is invalid or occupied")
            return None

        # Initialize tree with start node
        start_node = Node(start)
        tree = [start_node]

        # Store costmap and origin
        self.costmap = costmap
        self.origin = origin

        goal_node = None

        # Main RRT* loop
        for i in range(self.max_iterations):
            # Sample random point
            if np.random.random() < self.goal_sample_rate:
                sample_point = goal
            else:
                sample_point = self.sample_random_point(costmap, origin)

            # Find nearest node
            nearest_node = self.find_nearest_node(tree, sample_point)

            # Extend tree
            new_node = self.extend_tree(nearest_node, sample_point)

            if new_node is None:
                continue

            if not self.is_collision_free(nearest_node.position, new_node.position):
                continue

            # Find near nodes (RRT* modification)
            near_nodes = self.find_near_nodes(tree, new_node)

            # Choose best parent from near nodes
            best_parent = self.choose_parent(near_nodes, nearest_node, new_node)

            if best_parent is None:
                continue

            # Add new node to tree
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + self.distance(
                best_parent.position, new_node.position
            )
            tree.append(new_node)

            # Rewire tree (RRT* modification)
            self.rewire(tree, new_node, near_nodes)

            # Check if goal is reached
            if self.distance(new_node.position, goal) <= self.goal_tolerance:
                if goal_node is None or new_node.cost < goal_node.cost:
                    goal_node = new_node

        if goal_node is not None:
            print(f"Path found with cost {goal_node.cost:.2f}")
            path = self.extract_path(goal_node)
            return path

        print(f"Failed to find path after {self.max_iterations} iterations")
        return None

    def find_near_nodes(self, tree: List[Node], node: Node) -> List[Node]:
        """Find nodes near the given node"""
        near_nodes = []
        for tree_node in tree:
            if self.distance(tree_node.position, node.position) <= self.search_radius:
                near_nodes.append(tree_node)
        return near_nodes

    def choose_parent(
        self,
        near_nodes: List[Node],
        default_parent: Node,
        new_node: Node
    ) -> Optional[Node]:
        """Choose best parent from near nodes"""
        if not near_nodes:
            return default_parent

        min_cost = float('inf')
        best_parent = None

        for near_node in near_nodes:
            # Calculate cost through this node
            edge_cost = self.distance(near_node.position, new_node.position)
            total_cost = near_node.cost + edge_cost

            # Check if collision-free and lower cost
            if (total_cost < min_cost and
                self.is_collision_free(near_node.position, new_node.position)):
                min_cost = total_cost
                best_parent = near_node

        return best_parent if best_parent is not None else default_parent

    def rewire(self, tree: List[Node], new_node: Node, near_nodes: List[Node]):
        """Rewire tree to reduce costs"""
        for near_node in near_nodes:
            # Calculate cost through new node
            edge_cost = self.distance(new_node.position, near_node.position)
            new_cost = new_node.cost + edge_cost

            # If lower cost, rewire
            if (new_cost < near_node.cost and
                self.is_collision_free(new_node.position, near_node.position)):
                near_node.parent = new_node
                near_node.cost = new_cost

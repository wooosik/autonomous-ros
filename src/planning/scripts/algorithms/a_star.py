#!/usr/bin/env python3
"""
A* Path Planning Algorithm
Grid-based path planning using A* algorithm
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional


class Node:
    """Node class for A* algorithm"""

    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


class AStarPlanner:
    """A* path planning algorithm implementation"""

    def __init__(self, grid_resolution: float = 0.1, allow_diagonal: bool = True):
        """
        Initialize A* planner

        Args:
            grid_resolution: Resolution of the grid in meters
            allow_diagonal: Allow diagonal movement
        """
        self.grid_resolution = grid_resolution
        self.allow_diagonal = allow_diagonal

        # 8-connected grid (with diagonals) or 4-connected grid
        if allow_diagonal:
            self.directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            # Cost for diagonal movement is sqrt(2) * resolution
            self.move_costs = [
                1.414, 1.0, 1.414,
                1.0,        1.0,
                1.414, 1.0, 1.414
            ]
        else:
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            self.move_costs = [1.0, 1.0, 1.0, 1.0]

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        costmap: np.ndarray,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A* algorithm

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            costmap: 2D numpy array representing obstacles (0=free, 1=occupied)
            origin: Origin of the costmap in world coordinates

        Returns:
            List of waypoints (x, y) in world coordinates, or None if no path found
        """
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start, origin)
        goal_grid = self.world_to_grid(goal, origin)

        # Check if start and goal are valid
        if not self.is_valid_position(start_grid, costmap):
            print(f"Start position {start_grid} is invalid or occupied")
            return None

        if not self.is_valid_position(goal_grid, costmap):
            print(f"Goal position {goal_grid} is invalid or occupied")
            return None

        # Initialize start and goal nodes
        start_node = Node(start_grid)
        goal_node = Node(goal_grid)

        # Initialize open and closed lists
        open_list = []
        closed_set = set()

        # Add start node to open list
        heapq.heappush(open_list, (start_node.f, start_node))

        # Main A* loop
        while open_list:
            # Get node with lowest f value
            _, current_node = heapq.heappop(open_list)

            # Add to closed set
            closed_set.add(current_node.position)

            # Check if we reached the goal
            if current_node == goal_node:
                path = self.reconstruct_path(current_node)
                # Convert path back to world coordinates
                world_path = [self.grid_to_world(pos, origin) for pos in path]
                return world_path

            # Explore neighbors
            for i, direction in enumerate(self.directions):
                neighbor_pos = (
                    current_node.position[0] + direction[0],
                    current_node.position[1] + direction[1]
                )

                # Check if neighbor is valid
                if not self.is_valid_position(neighbor_pos, costmap):
                    continue

                # Skip if in closed set
                if neighbor_pos in closed_set:
                    continue

                # Create neighbor node
                neighbor_node = Node(neighbor_pos, current_node)

                # Calculate costs
                neighbor_node.g = current_node.g + self.move_costs[i]
                neighbor_node.h = self.heuristic(neighbor_pos, goal_grid)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # Check if neighbor is already in open list with lower cost
                in_open_list = False
                for _, open_node in open_list:
                    if open_node == neighbor_node and neighbor_node.g >= open_node.g:
                        in_open_list = True
                        break

                if not in_open_list:
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))

        # No path found
        print("No path found from start to goal")
        return None

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate heuristic cost (Euclidean distance)

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Heuristic cost
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.allow_diagonal:
            # Diagonal distance (Chebyshev + Euclidean)
            return min(dx, dy) * 1.414 + abs(dx - dy)
        else:
            # Manhattan distance
            return dx + dy

    def is_valid_position(self, pos: Tuple[int, int], costmap: np.ndarray) -> bool:
        """
        Check if position is valid (within bounds and not occupied)

        Args:
            pos: Position to check (grid coordinates)
            costmap: Costmap array

        Returns:
            True if valid, False otherwise
        """
        rows, cols = costmap.shape

        # Check bounds
        if pos[0] < 0 or pos[0] >= rows or pos[1] < 0 or pos[1] >= cols:
            return False

        # Check if occupied (assuming costmap values: 0=free, >0=occupied)
        if costmap[pos[0], pos[1]] > 0:
            return False

        return True

    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal node to start node

        Args:
            node: Goal node

        Returns:
            List of positions from start to goal
        """
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        # Reverse path to get start to goal
        path.reverse()
        return path

    def world_to_grid(
        self,
        world_pos: Tuple[float, float],
        origin: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates

        Args:
            world_pos: Position in world coordinates (x, y)
            origin: Origin of the grid in world coordinates

        Returns:
            Position in grid coordinates (row, col)
        """
        x = int((world_pos[0] - origin[0]) / self.grid_resolution)
        y = int((world_pos[1] - origin[1]) / self.grid_resolution)
        return (y, x)  # Note: row is y, col is x

    def grid_to_world(
        self,
        grid_pos: Tuple[int, int],
        origin: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates

        Args:
            grid_pos: Position in grid coordinates (row, col)
            origin: Origin of the grid in world coordinates

        Returns:
            Position in world coordinates (x, y)
        """
        x = grid_pos[1] * self.grid_resolution + origin[0]
        y = grid_pos[0] * self.grid_resolution + origin[1]
        return (x, y)


def smooth_path(
    path: List[Tuple[float, float]],
    weight_data: float = 0.5,
    weight_smooth: float = 0.3,
    tolerance: float = 0.000001
) -> List[Tuple[float, float]]:
    """
    Smooth path using gradient descent

    Args:
        path: Original path
        weight_data: Weight for data term (stay close to original path)
        weight_smooth: Weight for smoothness term
        tolerance: Convergence tolerance

    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path

    # Convert to numpy array
    new_path = np.array(path, dtype=float)

    # Gradient descent
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(2):  # x and y
                aux = new_path[i][j]

                # Update formula
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (
                    new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j]
                )

                change += abs(aux - new_path[i][j])

    return [tuple(point) for point in new_path]

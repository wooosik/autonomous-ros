#!/usr/bin/env python3
"""
Costmap Generator
Generates costmaps for path planning with obstacle inflation
"""

import numpy as np
from typing import List, Tuple
from scipy.ndimage import distance_transform_edt


class CostmapGenerator:
    """Generate and maintain costmap for path planning"""

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        inflation_radius: float = 0.5
    ):
        """
        Initialize costmap generator

        Args:
            width: Width of the map in meters
            height: Height of the map in meters
            resolution: Resolution of the grid in meters
            origin: Origin of the map in world coordinates (x, y)
            inflation_radius: Radius for obstacle inflation in meters
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        self.inflation_radius = inflation_radius

        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.inflation_cells = int(inflation_radius / resolution)

        # Initialize costmap (0 = free, 255 = occupied)
        self.costmap = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Static layer (permanent obstacles)
        self.static_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Dynamic layer (temporary obstacles)
        self.dynamic_layer = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            Grid coordinates (row, col)
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_y, grid_x)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates

        Args:
            row: Grid row
            col: Grid column

        Returns:
            World coordinates (x, y)
        """
        x = col * self.resolution + self.origin[0]
        y = row * self.resolution + self.origin[1]
        return (x, y)

    def is_valid_grid(self, row: int, col: int) -> bool:
        """Check if grid coordinates are valid"""
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width

    def add_obstacle(
        self,
        x: float,
        y: float,
        radius: float = 0.1,
        static: bool = True
    ):
        """
        Add circular obstacle to the costmap

        Args:
            x: X coordinate of obstacle center
            y: Y coordinate of obstacle center
            radius: Radius of the obstacle in meters
            static: If True, add to static layer, else dynamic layer
        """
        grid_y, grid_x = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)

        layer = self.static_layer if static else self.dynamic_layer

        # Draw circle
        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                if i * i + j * j <= grid_radius * grid_radius:
                    row = grid_y + i
                    col = grid_x + j
                    if self.is_valid_grid(row, col):
                        layer[row, col] = 255

    def add_rectangle_obstacle(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        static: bool = True
    ):
        """
        Add rectangular obstacle to the costmap

        Args:
            x_min: Minimum x coordinate
            y_min: Minimum y coordinate
            x_max: Maximum x coordinate
            y_max: Maximum y coordinate
            static: If True, add to static layer, else dynamic layer
        """
        grid_y_min, grid_x_min = self.world_to_grid(x_min, y_min)
        grid_y_max, grid_x_max = self.world_to_grid(x_max, y_max)

        # Ensure min < max
        if grid_y_min > grid_y_max:
            grid_y_min, grid_y_max = grid_y_max, grid_y_min
        if grid_x_min > grid_x_max:
            grid_x_min, grid_x_max = grid_x_max, grid_x_min

        layer = self.static_layer if static else self.dynamic_layer

        # Fill rectangle
        for row in range(grid_y_min, grid_y_max + 1):
            for col in range(grid_x_min, grid_x_max + 1):
                if self.is_valid_grid(row, col):
                    layer[row, col] = 255

    def add_line_obstacle(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float = 0.1,
        static: bool = True
    ):
        """
        Add line obstacle to the costmap

        Args:
            x1: Start x coordinate
            y1: Start y coordinate
            x2: End x coordinate
            y2: End y coordinate
            width: Width of the line in meters
            static: If True, add to static layer, else dynamic layer
        """
        grid_y1, grid_x1 = self.world_to_grid(x1, y1)
        grid_y2, grid_x2 = self.world_to_grid(x2, y2)
        grid_width = max(1, int(width / self.resolution))

        layer = self.static_layer if static else self.dynamic_layer

        # Bresenham's line algorithm with thickness
        dx = abs(grid_x2 - grid_x1)
        dy = abs(grid_y2 - grid_y1)
        sx = 1 if grid_x1 < grid_x2 else -1
        sy = 1 if grid_y1 < grid_y2 else -1
        err = dx - dy

        x, y = grid_x1, grid_y1

        while True:
            # Draw thick point
            for i in range(-grid_width, grid_width + 1):
                for j in range(-grid_width, grid_width + 1):
                    row = y + i
                    col = x + j
                    if self.is_valid_grid(row, col):
                        layer[row, col] = 255

            if x == grid_x2 and y == grid_y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def inflate_obstacles(self) -> np.ndarray:
        """
        Inflate obstacles using distance transform

        Returns:
            Inflated costmap
        """
        # Combine static and dynamic layers
        combined = np.maximum(self.static_layer, self.dynamic_layer)

        # Create binary obstacle map
        obstacles = (combined > 0).astype(np.uint8)

        # Calculate distance transform (distance to nearest obstacle)
        distances = distance_transform_edt(1 - obstacles)

        # Convert distances to costs
        inflated_costmap = np.zeros_like(combined)

        # Lethal obstacles
        inflated_costmap[obstacles == 1] = 255

        # Inflated area (cost decreases with distance)
        inflation_mask = (distances > 0) & (distances <= self.inflation_cells)
        if np.any(inflation_mask):
            # Linear decay from 254 to 1
            inflated_costmap[inflation_mask] = (
                254 * (1.0 - distances[inflation_mask] / self.inflation_cells)
            ).astype(np.uint8)

        return inflated_costmap

    def update_costmap(self):
        """Update the costmap with inflation"""
        self.costmap = self.inflate_obstacles()

    def get_costmap(self) -> np.ndarray:
        """
        Get the current costmap

        Returns:
            Costmap array (0-255, where 0 is free and 255 is occupied)
        """
        return self.costmap

    def get_binary_costmap(self, threshold: int = 253) -> np.ndarray:
        """
        Get binary costmap for path planning

        Args:
            threshold: Threshold for occupied cells

        Returns:
            Binary costmap (0 = free, 1 = occupied)
        """
        return (self.costmap > threshold).astype(np.uint8)

    def clear_dynamic_obstacles(self):
        """Clear all dynamic obstacles"""
        self.dynamic_layer.fill(0)

    def clear_all(self):
        """Clear all obstacles"""
        self.static_layer.fill(0)
        self.dynamic_layer.fill(0)
        self.costmap.fill(0)

    def get_cost_at_position(self, x: float, y: float) -> int:
        """
        Get cost at world position

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            Cost value (0-255)
        """
        grid_y, grid_x = self.world_to_grid(x, y)
        if self.is_valid_grid(grid_y, grid_x):
            return int(self.costmap[grid_y, grid_x])
        return 255  # Out of bounds is considered occupied

    def add_polygon_obstacle(
        self,
        vertices: List[Tuple[float, float]],
        static: bool = True
    ):
        """
        Add polygon obstacle using scan-line algorithm

        Args:
            vertices: List of (x, y) vertices in world coordinates
            static: If True, add to static layer, else dynamic layer
        """
        if len(vertices) < 3:
            return

        # Convert to grid coordinates
        grid_vertices = [self.world_to_grid(x, y) for x, y in vertices]

        # Get bounding box
        rows = [v[0] for v in grid_vertices]
        cols = [v[1] for v in grid_vertices]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        layer = self.static_layer if static else self.dynamic_layer

        # Scan-line fill algorithm
        for row in range(min_row, max_row + 1):
            intersections = []

            # Find intersections with polygon edges
            for i in range(len(grid_vertices)):
                v1 = grid_vertices[i]
                v2 = grid_vertices[(i + 1) % len(grid_vertices)]

                if v1[0] == v2[0]:  # Horizontal edge
                    continue

                if min(v1[0], v2[0]) <= row <= max(v1[0], v2[0]):
                    # Calculate intersection column
                    col = int(v1[1] + (row - v1[0]) * (v2[1] - v1[1]) / (v2[0] - v1[0]))
                    intersections.append(col)

            # Sort intersections and fill between pairs
            intersections.sort()
            for i in range(0, len(intersections) - 1, 2):
                for col in range(intersections[i], intersections[i + 1] + 1):
                    if self.is_valid_grid(row, col):
                        layer[row, col] = 255

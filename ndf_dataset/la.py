"""Compatibility imports for ray geometry and segmented point clouds."""

from .geometry.rays import closest_points_between_rays, calculate_rays, calculate_intersections
from .geometry.segmentation import segmented_pointcloud

__all__ = ["closest_points_between_rays", "calculate_rays", "calculate_intersections", "segmented_pointcloud"]

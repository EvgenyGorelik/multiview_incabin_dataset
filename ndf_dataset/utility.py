"""Compatibility imports; implementations live in geometry modules."""

from .geometry.bounding_box import vertices_to_3d_bbox
from .geometry.projection import get_intrinsics
from .geometry.skeleton import MHR_NAMES, MHR_CONNECTIONS, keypoints_to_skeleton_edges
from .geometry.transforms import compute_child_to_base

__all__ = ["vertices_to_3d_bbox", "get_intrinsics", "MHR_NAMES", "MHR_CONNECTIONS", "keypoints_to_skeleton_edges", "compute_child_to_base"]

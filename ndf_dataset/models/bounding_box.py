"""Oriented 3D bounding box annotations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class BoundingBox3D:
    """An oriented box; rotation uses quaternion order (x, y, z, w)."""

    bbox_label: str
    bbox_center: np.ndarray
    bbox_size: np.ndarray
    bbox_rotation: np.ndarray
    bbox_corners: np.ndarray
    obj_id: Optional[int] = None
    attr: Optional[str] = None
    action: Optional[str] = None
    velocity: Optional[np.ndarray] = None

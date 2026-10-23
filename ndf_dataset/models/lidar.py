"""Timestamped LiDAR frames."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class LidarFrame:
    timestamp: int
    path: str
    frame_id: Optional[str] = None
    points: Optional[np.ndarray] = None

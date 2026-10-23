"""Camera calibration records and image frames."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

@dataclass
class Camera:
    """Camera calibration and its timestamped image files."""

    name: str
    camera_type: str
    cam_info: Dict
    intrinsics: np.ndarray
    frame_id: Optional[str] = None
    images: List[Tuple[int, str]] = field(default_factory=list)



@dataclass
class CameraFrame:
    camera: Camera
    timestamp: Optional[int]
    path: Optional[str]
    image: Optional[np.ndarray] = None

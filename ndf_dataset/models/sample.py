"""Associated sample files and loaded samples."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .bounding_box import BoundingBox3D
from .camera import CameraFrame
from .lidar import LidarFrame

@dataclass
class SampleFiles:
    """Associated files without reading images, point clouds or annotations."""

    timestamp: int
    bboxes_3d_path: str
    cameras: Dict[str, Dict[str, CameraFrame]]
    lidar: Optional[LidarFrame] = None
    scene_id: Optional[str] = None



@dataclass
class Sample:
    timestamp: int
    bboxes_3d: List[BoundingBox3D]
    cameras: Dict[str, Dict[str, CameraFrame]]
    lidar: Optional[LidarFrame] = None
    scene_id: Optional[str] = None

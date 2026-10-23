"""Coordinate frame calibration and transforms between dataset sensors."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

from .io.calibration import load_transforms


@dataclass
class Calibration:
    transforms_child_to_parent: Dict[str, Tuple[Optional[str], np.ndarray]]
    transforms_to_base: Dict[str, np.ndarray]

    @classmethod
    def from_root(cls, root):
        path = Path(root) / 'target_transforms.json'
        return cls(*load_transforms(path)) if path.is_file() else cls({}, {})

    def get_transform(self, source_frame, target_frame):
        """Map source coordinates to target coordinates, or None if unknown."""
        source = (np.eye(4) if source_frame == 'base_link'
                  else self.transforms_to_base.get(source_frame))
        target = (np.eye(4) if target_frame == 'base_link'
                  else self.transforms_to_base.get(target_frame))
        if source is None or target is None:
            return None
        return np.linalg.inv(target) @ source

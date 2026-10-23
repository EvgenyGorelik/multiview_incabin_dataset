"""A recording scene with its own sensors, annotations and sample timeline."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .calibration import Calibration
from .indexing import DatasetIndex
from .sampling import SampleLoader


@dataclass
class Scene:
    """A lazy sample sequence restricted to a single recording.

    Cameras retain every available image assigned to the scene, including
    images that are not selected by any annotation-timestamp sample.
    Calibration and the static map are shared with the parent dataset.
    """

    name: str
    index: DatasetIndex
    calibration: Calibration
    pcd_map: Optional[np.ndarray]
    frame_id_map: Dict
    modalities: Dict
    metadata: Dict = field(default_factory=dict)
    _samples: SampleLoader = field(init=False, repr=False)

    def __post_init__(self):
        self._samples = SampleLoader(self.index, self.frame_id_map.get('lidar'), self.name)

    @property
    def timestamps(self):
        return self.index.timestamps

    @property
    def cameras(self):
        return self.index.cameras

    @property
    def lidar_scans(self):
        return self.index.lidar_scans

    @property
    def bboxes_3d(self):
        return self.index.bboxes_3d

    def get_sample_files(self, timestamp):
        return self._samples.get_files(timestamp)

    def get_sample(self, timestamp):
        return self._samples.load(timestamp)

    def __getitem__(self, index):
        return self.get_sample(self.timestamps[index])

    def __len__(self):
        return len(self.timestamps)

    def __iter__(self):
        for timestamp in self.timestamps:
            yield self.get_sample(timestamp)

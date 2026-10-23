"""Dataset access through indexed samples, sensors, calibration and a static map."""

from pathlib import Path
from typing import Optional

from .calibration import Calibration
from .geometry.projection import project_pointcloud_to_image, project_poses_to_camera
from .indexing import DatasetIndex
from .io.metadata import load_metadata
from .io.pointcloud import load_pointcloud
from .models.sample import Sample, SampleFiles
from .sampling import SampleLoader
from .scene_indexing import build_scenes
from .timestamps import nearest_timestamp


class NDFDataset:
    """Samples indexed by 3D box timestamps with nearest sensor frames."""

    def __init__(self, root: str):
        self.root = root
        self.frame_id_map = load_metadata(root, 'frame_ids.json', {})
        self.lidar_frame = self.frame_id_map.get('lidar')
        self.modalities = load_metadata(
            root, 'modalities.json', {'cameras': [], 'depth': [], 'lidar': [], 'audio': []})
        self.index = DatasetIndex.scan(root, self.frame_id_map)
        self.calibration = Calibration.from_root(root)
        self._samples = SampleLoader(self.index, self.lidar_frame)
        self.pcd_map = load_pointcloud(Path(root) / 'map' / 'dense_cleaned.ply')

        # Preserve the public collections used by existing dataset consumers.
        self.cameras = self.index.cameras
        self.lidar_scans = self.index.lidar_scans
        self.bboxes_3d = self.index.bboxes_3d
        self.timestamps = self.index.timestamps
        self.lidar_timestamps = self.index.lidar_timestamps
        self.transforms_child_to_parent = self.calibration.transforms_child_to_parent
        self.transforms_to_base = self.calibration.transforms_to_base
        self.scenes, self.scene_by_timestamp = build_scenes(
            root, self.index, self.calibration, self.pcd_map,
            self.frame_id_map, self.modalities)

    def find_nearest_lidar(self, timestamp):
        """Return the nearest scan's (timestamp, path), or None."""
        return self._samples.find_nearest_lidar(timestamp)

    def get_sample_files(self, timestamp) -> Optional[SampleFiles]:
        """Associate nearest sensor files with the nearest annotation time."""
        scene_name = self.scene_by_timestamp.get(nearest_timestamp(self.timestamps, timestamp))
        if scene_name is not None:
            return self.scenes[scene_name].get_sample_files(timestamp)
        return self._samples.get_files(timestamp)

    def get_sample(self, timestamp) -> Optional[Sample]:
        """Load boxes and sensor data at the nearest annotation time."""
        scene_name = self.scene_by_timestamp.get(nearest_timestamp(self.timestamps, timestamp))
        if scene_name is not None:
            return self.scenes[scene_name].get_sample(timestamp)
        return self._samples.load(timestamp)

    def get_transform(self, source_frame, target_frame):
        """Return a source-to-target transform, or None if unavailable."""
        return self.calibration.get_transform(source_frame, target_frame)

    def project_pointcloud_to_image(self, pts, cam_name, camera_type='color', return_valid=False):
        """Project LiDAR points using the selected dataset camera calibration."""
        camera = self.cameras.get(camera_type, {}).get(cam_name)
        if camera is None:
            raise RuntimeError('camera not found: ' + cam_name)
        transform = self.get_transform(self.lidar_frame, camera.frame_id)
        if transform is None:
            raise RuntimeError('LiDAR to camera transform is unavailable: ' + cam_name)
        uv, valid = project_pointcloud_to_image(pts, camera.cam_info, transform)
        return (uv, valid) if return_valid else uv[valid]

    def project_poses_to_image(self, poses, cam_name, key='vertices', return_valid=False, camera_type='color'):
        """Project caller-supplied legacy poses using the dataset calibration."""
        camera = self.cameras.get(camera_type, {}).get(cam_name)
        if camera is None:
            raise RuntimeError('camera not found: ' + cam_name)
        transform = self.get_transform('base_link', camera.frame_id)
        if transform is None:
            raise RuntimeError('Base to camera transform is unavailable: ' + cam_name)
        return project_poses_to_camera(poses, camera, transform, key, return_valid)

    def __getitem__(self, idx):
        return self.get_sample(self.timestamps[idx])

    def __len__(self):
        return len(self.timestamps)

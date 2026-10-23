"""Associate annotation timestamps with sensors and load sample contents."""

from .io.annotations import load_bboxes_3d
from .io.camera import load_image
from .io.pointcloud import load_pointcloud
from .models.camera import CameraFrame
from .models.lidar import LidarFrame
from .models.sample import Sample, SampleFiles
from .timestamps import nearest_timestamp


class SampleLoader:
    def __init__(self, index, lidar_frame=None, scene_id=None):
        self.index = index
        self.lidar_frame = lidar_frame
        self.scene_id = scene_id

    def find_nearest_lidar(self, timestamp):
        timestamp = nearest_timestamp(self.index.lidar_timestamps, timestamp)
        if timestamp is None:
            return None
        return timestamp, self.index.lidar_scans[timestamp]

    def get_files(self, timestamp):
        """Select the nearest annotation time and each sensor's nearest frame."""
        timestamp = nearest_timestamp(self.index.timestamps, timestamp)
        if timestamp is None:
            return None
        cameras = {}
        for kind, camera_group in self.index.cameras.items():
            cameras[kind] = {}
            for name, camera in camera_group.items():
                image_ts, path = min(
                    camera.images,
                    key=lambda item: (abs(item[0] - timestamp), item[0]),
                    default=(None, None),
                )
                cameras[kind][name] = CameraFrame(camera, image_ts, path)
        nearest_lidar = self.find_nearest_lidar(timestamp)
        lidar = (LidarFrame(*nearest_lidar, frame_id=self.lidar_frame)
                 if nearest_lidar is not None else None)
        return SampleFiles(timestamp, self.index.bboxes_3d[timestamp], cameras, lidar, self.scene_id)

    def load(self, timestamp):
        files = self.get_files(timestamp)
        if files is None:
            return None
        for kind, cameras in files.cameras.items():
            for frame in cameras.values():
                if frame.path is not None:
                    frame.image = load_image(
                        frame.path, image_type='depth' if kind == 'depth' else 'color')
        if files.lidar is not None:
            files.lidar.points = load_pointcloud(files.lidar.path)
        return Sample(files.timestamp, load_bboxes_3d(files.bboxes_3d_path),
                      files.cameras, files.lidar, files.scene_id)

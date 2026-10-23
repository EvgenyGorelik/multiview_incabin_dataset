"""Discover supported dataset files without loading sample contents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .geometry.projection import get_intrinsics
from .io.camera import load_camera_info
from .models.camera import Camera
from .timestamps import filename_timestamp


def index_timestamped_files(directory, extensions):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    entries = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            timestamp = filename_timestamp(path)
            if timestamp is not None:
                entries[timestamp] = str(path)
    return dict(sorted(entries.items()))


def index_cameras(directory, frame_ids):
    directory = Path(directory)
    cameras = {}
    if not directory.is_dir():
        return cameras
    for camera_type in sorted(directory.iterdir()):
        if not camera_type.is_dir():
            continue
        for camera_path in sorted(camera_type.iterdir()):
            info_path = camera_path / 'camera_info.yaml'
            if not camera_path.is_dir() or not info_path.is_file():
                continue
            info = load_camera_info(info_path)
            images = index_timestamped_files(
                camera_path / 'images', {'.jpg', '.jpeg', '.png', '.bmp'})
            cameras.setdefault(camera_type.name, {})[camera_path.name] = Camera(
                name=camera_path.name,
                camera_type=camera_type.name,
                cam_info=info,
                intrinsics=get_intrinsics(info),
                frame_id=frame_ids.get(camera_path.name),
                images=list(images.items()),
            )
    return cameras


@dataclass
class DatasetIndex:
    cameras: Dict[str, Dict[str, Camera]]
    lidar_scans: Dict[int, str]
    bboxes_3d: Dict[int, str]
    timestamps: List[int]
    lidar_timestamps: List[int]

    @classmethod
    def scan(cls, root, frame_ids):
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        cameras = index_cameras(root / 'cameras', frame_ids)
        lidar = index_timestamped_files(root / 'lidar', {'.pcd', '.bin'})
        boxes = index_timestamped_files(root / 'bboxes_3d', {'.json'})
        return cls(cameras, lidar, boxes, sorted(boxes), sorted(lidar))

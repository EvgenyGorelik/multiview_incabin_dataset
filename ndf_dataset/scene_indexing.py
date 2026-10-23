"""Build scene-local indexes from recording-to-sensor associations."""

from dataclasses import replace
from pathlib import Path

from .indexing import DatasetIndex
from .io.scenes import load_scene_metadata
from .scene import Scene
from .timestamps import filename_timestamp


def _camera_references(name, paths):
    """Identify camera streams independently of the original export root."""
    references = {}
    for path in paths:
        parts = Path(path).parts
        # Exported paths include cameras/<type>/<name>/images/<timestamp>.
        matches = [i for i, part in enumerate(parts) if part == 'cameras']
        kind = parts[matches[-1] + 1] if matches and len(parts) > matches[-1] + 2 else None
        timestamp = filename_timestamp(path)
        if timestamp is not None:
            references.setdefault((kind, name), set()).add(timestamp)
    return references


def build_scenes(root, index, calibration, pcd_map, frame_ids, modalities):
    metadata = load_scene_metadata(root)
    scene_indexes = {}
    reference_times = {}
    lidar_owners = {}
    for name, entry in metadata.items():
        camera_refs = {}
        for camera_name, paths in entry.get('cameras', {}).items():
            for key, timestamps in _camera_references(camera_name, paths).items():
                camera_refs.setdefault(key, set()).update(timestamps)
        lidar_refs = {filename_timestamp(path) for path in entry.get('lidar', [])}
        lidar_refs.discard(None)
        for timestamp in lidar_refs:
            if timestamp in lidar_owners and lidar_owners[timestamp] != name:
                raise ValueError(f'LiDAR timestamp {timestamp} belongs to multiple scenes')
            lidar_owners[timestamp] = name
        times = set(lidar_refs)
        for timestamps in camera_refs.values():
            times.update(timestamps)
        reference_times[name] = (min(times), max(times)) if times else None
        cameras = {}
        for kind, group in index.cameras.items():
            for camera_name, camera in group.items():
                references = camera_refs.get((kind, camera_name), set())
                references = references | camera_refs.get((None, camera_name), set())
                if (kind, camera_name) not in camera_refs and (None, camera_name) not in camera_refs:
                    continue
                images = [(ts, path) for ts, path in camera.images if ts in references]
                cameras.setdefault(kind, {})[camera_name] = replace(camera, images=images)
        lidar = {ts: path for ts, path in index.lidar_scans.items() if ts in lidar_refs}
        scene_indexes[name] = DatasetIndex(cameras, lidar, {}, [], sorted(lidar))

    owners = {}
    for timestamp, path in index.bboxes_3d.items():
        name = lidar_owners.get(timestamp)
        if name is None:
            candidates = [scene_name for scene_name, span in reference_times.items()
                          if span is not None and span[0] <= timestamp <= span[1]]
            if len(candidates) > 1:
                raise ValueError(f'Annotation timestamp {timestamp} matches multiple scenes: {candidates}')
            name = candidates[0] if candidates else None
        if name is not None:
            scene_indexes[name].bboxes_3d[timestamp] = path
            owners[timestamp] = name

    scenes = {}
    for name, scene_index in scene_indexes.items():
        scene_index.timestamps = sorted(scene_index.bboxes_3d)
        scenes[name] = Scene(name, scene_index, calibration, pcd_map,
                             frame_ids, modalities, metadata[name])
    return scenes, owners

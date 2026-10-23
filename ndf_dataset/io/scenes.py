"""Read recording-to-sensor associations from scenes.json."""

from .metadata import load_metadata


def load_scene_metadata(root):
    scenes = load_metadata(root, 'scenes.json', {})
    if not isinstance(scenes, dict):
        raise ValueError('scenes.json must map scene names to sensor associations')
    for name, entry in scenes.items():
        if not isinstance(entry, dict):
            raise ValueError(f'Scene {name!r} must contain sensor associations')
        if not isinstance(entry.get('cameras', {}), dict):
            raise ValueError(f'Scene {name!r} cameras must map names to image paths')
        if not isinstance(entry.get('lidar', []), list):
            raise ValueError(f'Scene {name!r} lidar must be a list of paths')
        for paths in entry.get('cameras', {}).values():
            if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                raise ValueError(f'Scene {name!r} camera images must be lists of paths')
        if not all(isinstance(path, str) for path in entry.get('lidar', [])):
            raise ValueError(f'Scene {name!r} lidar must contain paths')
    return scenes

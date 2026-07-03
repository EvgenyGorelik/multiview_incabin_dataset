#!/usr/bin/env python3
"""
Convert ROS2 bag data to structured directories for cameras and lidar, including metadata extraction.
"""

try:
    import rosbag2_py
    from rosbag2_py import SequentialReader
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
except ImportError:
    print("ROS2 Python libraries not found. Please source your ROS2 workspace and try again.")
    exit(1)

import yaml
import os
from pathlib import Path
from glob import glob
from typing import Dict, List, Any
import numpy as np
import xml.etree.ElementTree as ET
import json
from argparse import ArgumentParser

class TFTree:
    def __init__(self, urdf_path: str, target_frame: str):
        self.urdf_path = urdf_path
        self.target_frame = target_frame
        self.tree = self.parse_urdf()
    
    def parse_urdf(self) -> Dict[str, Any]:
        with open(self.urdf_path, 'r') as f:
            root = ET.parse(f).getroot()

        tree = {}
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = joint.find('origin')
            
            tree[child] = {
                'parent': parent,
                'joint_name': joint_name,
                'xyz': origin.get('xyz', '0 0 0'),
                'rpy': origin.get('rpy', '0 0 0'),
                'type': joint.get('type')
            }

        return tree

    def link_to_target(self, link_name: str) -> List[Dict[str, Any]]:
        path = []
        current_link = link_name
        
        while current_link != self.target_frame:
            if current_link not in self.tree:
                raise ValueError(f"Link {current_link} not found in URDF tree.")
            joint_info = self.tree[current_link]
            path.append(joint_info)
            current_link = joint_info['parent']
        
        return path

def write_transforms(file_path: str, transforms: Dict[str, Any]) -> None:
    with open(file_path, 'w') as f:
        json.dump(transforms, f, indent=4)

def get_storage_id(path: str) -> str:
    """Determine storage format from bag directory."""
    for file in glob(path + "/*"):
        ext = os.path.splitext(file)[1]
        if ext == ".db3":
            return "sqlite3"
        elif ext == ".mcap":
            return "mcap"
    return "sqlite3"


def write_pcd(filename: str, points: np.ndarray) -> None:
    """Write point cloud data to PCD file."""
    data = f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nWIDTH {points.shape[0]}\nHEIGHT 1\nPOINTS {points.shape[0]}\nDATA ascii\n"
    data += "".join(["{} {} {} {}\n".format(point[0], point[1], point[2], point[3]) for point in points])
    with open(filename, 'w') as f:
        f.write(data)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_directory_structure(base_path: str, config: Dict[str, Any]) -> None:
    """Create output directory structure for cameras and lidar."""
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    for camera in config.get('cameras', []):
        for camera_name, camera_config in camera.items():
            camera_dir = Path(base_path) / "cameras" / camera_config.get('camera_type', 'color') / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            (camera_dir / "images").mkdir(parents=True, exist_ok=True)
    
    if config.get('lidar'):
        lidar_dir = Path(base_path) / 'lidar'
        lidar_dir.mkdir(parents=True, exist_ok=True)


def convert_rosbag(input_bag_path: list, output_base_path: str, config: Dict[str, Any], meta_only: bool = False) -> None:
    """Export ROS2 bag data to structured directories."""
    assert isinstance(input_bag_path, list), "input_bag_path should be a list of bag paths."
    assert len(input_bag_path) > 0, "input_bag_path list is empty."
    
    create_directory_structure(output_base_path, config)

    lidar_topic = config.get('lidar', {}).get('topic')
    lidar_frame_id = config.get('lidar', {}).get('frame_id')


    tf_tree = TFTree(config['urdf_path'], config['target_frame'])
    target_transforms = {}
    
    initialized_fs = False

    data_association = {}  # To keep track of which camera images belong to which bag

    for bag_path in input_bag_path:
        storage_id = get_storage_id(bag_path)
        reader_storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id=storage_id
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = SequentialReader()
        reader.open(reader_storage_options, converter_options)
        
        if not initialized_fs:
            # Build topic to camera mapping
            topic_to_camera = {}
            camera_directory_mapping = {}
            camera_info_topics = {}
            camera_frame_ids = {}
            for camera in config.get('cameras', []):
                for camera_name, camera_config in camera.items():
                    topic = camera_config['topic']
                    topic_to_camera[topic] = camera_name
                    camera_info_topics[camera_config['camera_info_topic']] = camera_name
                    camera_frame_ids[camera_name] = camera_config.get('frame_id')
                    camera_directory_mapping[camera_name] = Path(output_base_path) / "cameras" / camera_config.get('camera_type', 'color') / camera_name
            initialized_fs = True

        bag_name = os.path.basename(bag_path)
        data_association[bag_name] = {
            'lidar': [],
            'cameras': {camera_name: [] for camera_name in camera_frame_ids.keys()}
        }

        print(f"Processing bag: {bag_name}")

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            
            # Handle camera images
            if topic in topic_to_camera:
                camera_name = topic_to_camera[topic]
                output_dir = camera_directory_mapping.get(camera_name) / "images"
                filename = output_dir / f"{str(timestamp).zfill(19)}.jpg"
                data_association[bag_name]['cameras'][camera_name].append(str(filename))
                
                msg = deserialize_message(data, CompressedImage)
                with open(filename, 'wb') as f:
                    f.write(msg.data)
                

                camera_frame_id = camera_frame_ids.get(camera_name)

                if camera_frame_id not in target_transforms:
                    try:
                        target_transforms[camera_frame_id] = tf_tree.link_to_target(camera_frame_id)
                    except ValueError as e:
                        print(f"Warning: {e}. Skipping transform for {camera_frame_id}.")
            
            # Handle camera info
            elif topic in camera_info_topics:
                camera_name = camera_info_topics[topic]
                output_dir = camera_directory_mapping.get(camera_name)
                filename = output_dir / "camera_info.yaml"
                
                msg = deserialize_message(data, CameraInfo)
                # Save camera info (implement serialization as needed)
                with open(filename, 'w') as f:
                    yaml.dump({
                        'image_width': msg.width,
                        'image_height': msg.height,
                        'camera_name': camera_name,
                        'camera_matrix': {
                            'rows': 3,
                            'cols': 3,
                            'data': [float(k) for k in msg.k]
                        },
                        'distortion_model': msg.distortion_model,
                        'distortion_coefficients': {
                            'rows': 1,
                            'cols': 5,
                            'data': [float(d) for d in msg.d]
                        },
                        'rectification_matrix': {
                            'rows': 3,
                            'cols': 3,
                            'data': [float(r) for r in msg.r]
                        },
                        'projection_matrix': {
                            'rows': 3,
                            'cols': 4,
                            'data': [float(p) for p in msg.p]
                        }
                    }, f, default_flow_style=False)
                camera_info_topics.pop(topic)  # Remove from mapping to avoid duplicate saves

            
            # Handle lidar
            elif topic == lidar_topic:
                output_dir = Path(output_base_path) / 'lidar'
                filename = output_dir / f"{str(timestamp).zfill(19)}.bin"
                data_association[bag_name]['lidar'].append(str(filename))
                
                msg = deserialize_message(data, PointCloud2)
                pc_array = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
                pc_array = pc_array[:, :4]  # Keep only x,y,z
                pc_array.tofile(filename)  # Save as binary PCD for efficiency
                # write_pcd(filename, pc_array)

                if lidar_frame_id not in target_transforms:
                    target_transforms[lidar_frame_id] = tf_tree.link_to_target(lidar_frame_id)

            if meta_only and lidar_frame_id in target_transforms and all(camera_frame_id in target_transforms for camera_frame_id in camera_frame_ids.values()):
                break  # Stop early if we only want metadata and have collected all transforms

    
    # Save target transforms
    transforms_file = Path(output_base_path) / "target_transforms.json"
    write_transforms(transforms_file, target_transforms)

    frame_ids_file = Path(output_base_path) / "frame_ids.json"
    with open(frame_ids_file, 'w') as f:
        frame_ids = {k: v for k, v in camera_frame_ids.items()}
        frame_ids.update({
            'lidar': lidar_frame_id
        })
        json.dump(frame_ids, f, indent=4)
    with open(Path(output_base_path) / "modalities.json", 'w') as f:
        json.dump({
            'cameras': list(camera_frame_ids.keys()),
            'lidar': [lidar_frame_id] if lidar_frame_id else []
        }, f, indent=4)

    with open(Path(output_base_path) / "data_association.json", 'w') as f:
        json.dump(data_association, f, indent=4)

    reader.close()
    print(f"Export complete. Output saved to {output_base_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export ROS2 bag data to structured directories.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML configuration file.')
    parser.add_argument('--meta-only', action='store_true', help='Only export metadata (camera info and transforms) without images or point clouds.')
    args = parser.parse_args()
    config_file = args.config
    config = load_config(config_file)
    convert_rosbag(config['input_bags'], config['output_path'], config, meta_only=args.meta_only)
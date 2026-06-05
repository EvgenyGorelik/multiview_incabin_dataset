#!/usr/bin/env python3
"""
Data loader for the NDF dataset.
"""

import os
from typing import Dict
import json
import numpy as np
from collections import defaultdict

from ndf_dataset.io import load_camera_info, load_pointcloud, load_image, load_poses, load_body_info, load_bounding_boxes, load_bboxes_3d, load_states
from ndf_dataset.transforms import rpy_xyz_to_homogeneous, project_pointcloud_to_image
from ndf_dataset.utility import get_intrinsics


class NDFDataset:
    """
    - Scan dataset directory and build lists of camera images and lidar clouds
    - Parse camera_info.yaml for intrinsics
    - Parse target_transforms.yaml to build transforms from each frame -> base_link
    - Provide method to find nearest lidar sample for a given image (by timestamp)
    """

    def __init__(self, root: str):
        self.frame_id_map = {}
        self.lidar_frame = None
        self.root = root
        self.cameras = defaultdict(lambda: defaultdict(dict))  # camera_type -> camera_name -> dict with 'images' list (tuples (ts, path)) and 'cam_info' and 'intrinsics'
        self.lidar_scans = {}    # dict of ts -> path
        self.poses = {}          # dict of ts -> path
        self.bboxes_3d = {}       # dict of ts -> path
        self.states = {}          # dict of ts -> path
        self.transforms_child_to_parent = {}  # child_frame -> 4x4 transform mapping child coords -> parent coords
        self.transforms_to_base = {}  # top-level frames (keys in yaml) -> 4x4 mapping child_frame -> base_link coords
        self._load_frame_id_map()
        self._load_modalities()
        self._scan()
        if self.lidar_scans:
            self.timestamps = sorted(set([ts for ts in self.lidar_scans.keys()]))
        else:
            self.timestamps = sorted(set([ts for cam in self.cameras.values() for ts, _ in cam["images"]]))
        self.cameras = dict(self.cameras)

    def _load_frame_id_map(self):
        fmap = os.path.join(self.root, "frame_ids.json")
        if os.path.exists(fmap):
            with open(fmap, "r") as fh:
                fm = json.load(fh)
                for folder_name, frame_id in fm.items():
                    self.frame_id_map[folder_name] = frame_id
                self.lidar_frame = fm.get("lidar")

    def _load_modalities(self):
        modalities_path = os.path.join(self.root, "modalities.json")
        if os.path.exists(modalities_path):
            with open(modalities_path, "r") as fh:
                self.modalities = json.load(fh)
        else:
            self.modalities = {"cameras": [], "depth": [], "lidar": [], "audio": []}

    def _scan(self):
        # cameras: any subdir that starts with 'camera_' or contains camera_info.yaml
        for entry in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, entry)
            if not os.path.isdir(p):
                continue
            # lidar folder
            if entry == "lidar":
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".pcd") or f.lower().endswith(".bin"):
                        ts = self._filename_timestamp(f)
                        if ts is not None:
                            self.lidar_scans[ts] = os.path.join(p, f)
                continue
            # pose folder
            if entry == "poses":
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".json"):
                        ts = self._filename_timestamp(f)
                        if ts is not None:
                            self.poses[ts] = os.path.join(p, f)
                continue

            # 3D bbox folder
            if entry == "bboxes_3d":
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".json"):
                        ts = self._filename_timestamp(f)
                        if ts is not None:
                            self.bboxes_3d[ts] = os.path.join(p, f)
                continue

            if entry == "cameras":
                for camera_type in sorted(os.listdir(p)):
                    camera_type_p = os.path.join(p, camera_type)
                    if os.path.isdir(camera_type_p):
                        # camera type folders (e.g., "color", "depth")
                        for cam_name in sorted(os.listdir(camera_type_p)):
                            cam_p = os.path.join(camera_type_p, cam_name)
                            if os.path.isdir(cam_p):
                                # camera folders
                                cam_info_f = os.path.join(cam_p, "camera_info.yaml")
                                images_dir = os.path.join(cam_p, "images")
                                # cam_name = cam_name.replace('_' + camera_type, "")  # remove camera type prefix if present

                                if os.path.exists(cam_info_f):
                                    cam_info = load_camera_info(cam_info_f)
                                    intrinsics = get_intrinsics(cam_info)
                                    images = []
                                    if os.path.isdir(images_dir):
                                        for f in sorted(os.listdir(images_dir)):
                                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                                                ts = self._filename_timestamp(f)
                                                if ts is not None:
                                                    images.append((ts, os.path.join(images_dir, f)))

                                    self.cameras[camera_type][cam_name]["cam_info"] = cam_info
                                    self.cameras[camera_type][cam_name]["intrinsics"] = intrinsics
                                    self.cameras[camera_type][cam_name]["images"] = images
                                    self.cameras[camera_type][cam_name]['frame_id'] = self.frame_id_map.get(cam_name)
            
            if entry == "segmentations":
                for cam_name in sorted(os.listdir(p)):
                    segmentation_dir = os.path.join(p, cam_name)
                    if os.path.isdir(segmentation_dir):
                        bboxes = defaultdict(list)  # ts -> bbox json path
                        segmentation = defaultdict(list)  # ts -> segmentation path
                        for f in sorted(os.listdir(segmentation_dir)):
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                                ts = self._segmentation_timestamp(f)
                                if ts is not None:
                                    segmentation[ts].append(os.path.join(segmentation_dir, f))
                            elif f.lower().endswith(".json"):
                                ts = self._segmentation_timestamp(f)
                                if ts is not None:
                                    bboxes[ts] = os.path.join(segmentation_dir, f)

                        self.cameras['color'][cam_name]["segmentations"] = dict(segmentation)
                        self.cameras['color'][cam_name]["bboxes_2d"] = dict(bboxes)
                
            if entry == "body_3d":
                for cam_name in sorted(os.listdir(p)):
                    body_dir = os.path.join(p, cam_name)
                    if os.path.isdir(body_dir):
                        body = defaultdict(list)
                        for f in sorted(os.listdir(body_dir)):
                            if f.lower().endswith(".json"):
                                file_path = os.path.join(body_dir, f)
                                ts = self._segmentation_timestamp(f)
                                if ts is not None:
                                    body[ts].append(file_path)
                    self.cameras['color'][cam_name]["body_3d"] = dict(body)

            if entry == "states":
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".json"):
                        ts = self._filename_timestamp(f)
                        if ts is not None:
                            self.states[ts] = os.path.join(p, f)

        self.cameras = dict(self.cameras)
        # parse transforms
        tt = os.path.join(self.root, "target_transforms.json")
        if os.path.exists(tt):
            with open(tt, "r") as fh:
                json_obj = json.load(fh) 
            for key, joint_list in json_obj.items():
                for j in joint_list:
                    jname = j.get("joint_name", "")
                    parent = j.get("parent")
                    xyz = [float(v) for v in str(j.get("xyz", "")).split()] if j.get("xyz") is not None else [0.0, 0.0, 0.0]
                    rpy = [float(v) for v in str(j.get("rpy", "")).split()] if j.get("rpy") is not None else [0.0, 0.0, 0.0]
                    # parse child from joint_name by splitting on "_to_"
                    parts = jname.split("_to_")
                    if len(parts) == 2:
                        parsed_parent = parts[0]
                        child = parts[1]
                    else:
                        # fallback: use parent + something
                        child = jname.replace(parent + "_to_", "") if parent else jname
                    # Build transform matrix mapping child -> parent (child coords to parent coords)
                    T = rpy_xyz_to_homogeneous(rpy, xyz)
                    # store; if duplicate child, last wins
                    self.transforms_child_to_parent[child] = (parent, T)
            # Now for each top-level key (frame group) compute transform from that frame -> base_link
            for key in json_obj.keys():
                T = self._compute_child_to_base(key)
                if T is not None:
                    self.transforms_to_base[key] = T

    def _compute_child_to_base(self, child):
        """Walk child->parent links until base_link or until can't continue.
           Compose transforms so final T maps original child coords -> base_link coords.
        """
        cur = child
        mats = []
        visited = set()
        while True:
            if cur in visited:
                return None
            visited.add(cur)
            entry = self.transforms_child_to_parent.get(cur)
            if entry is None:
                return None
            parent, T_child_to_parent = entry
            mats.append(T_child_to_parent)  # child -> parent
            if parent == "base_link":
                # compose: total = T_parent_n * ... * T_child_to_parent
                total = np.eye(4)
                for M in mats:
                    total = M @ total
                return total
            # next child is parent frame (walk upward)
            cur = parent
            if cur is None:
                return None

    def _filename_timestamp(self, filename):
        stem = os.path.splitext(filename)[0]
        try:
            return int(stem)
        except Exception:
            return None
    
    def _segmentation_timestamp(self, filename):
        stem = os.path.splitext(filename)[0]
        try:
            timestamp = int(stem.split("_")[0])
            return timestamp
        except Exception:
            return None
    
    def find_nearest_lidar(self, ts):
        """Return path to nearest lidar pcd by timestamp (or None)."""
        if not self.lidar_scans:
            return None
        if ts in self.lidar_scans:
            return ts, self.lidar_scans[ts]
        idx = np.searchsorted([t for t in self.timestamps], ts)
        cand = []
        if idx > 0:
            cand.append(self.lidar_scans[self.timestamps[idx - 1]])
        if idx < len(self.lidar_scans):
            cand.append(self.lidar_scans[self.timestamps[min(idx, len(self.lidar_scans) - 1)]])
        best = min(cand, key=lambda x: abs(x[0] - ts))
        return best[0], best[1]

    def project_poses_to_image(self, poses, cam_name, key='vertices', return_valid=False, camera_type='color'):
        """
        Project list of poses (each with 'vertices' and 'keypoints') to the specified camera image plane. 
        Returns list of projected vertices and keypoints (in pixel coordinates), and optionally validity masks if return_valid=True.
        """
        results = []
        for pose in poses:
            verts = np.array(pose[key])
            T_base_to_cam = np.linalg.inv(self.transforms_to_base[self.frame_id_map[cam_name]])
            cam_int = self.cameras[camera_type].get(cam_name, {}).get("cam_info")
            uv, valid = project_pointcloud_to_image(verts, cam_int, T_base_to_cam)
            if return_valid:
                results.append((uv, valid))
            else:
                results.append(uv[valid])
        return results

    def project_pointcloud_to_image(self, pts, cam_name, camera_type='color', return_valid=False):
        """Project a pointcloud to the specified camera image plane. Returns projected points in pixel coordinates, and optionally validity mask if return_valid=True."""
        caminfo = self.cameras[camera_type].get(cam_name, {}).get("cam_info")
        if caminfo is None:
            raise RuntimeError("camera not found: " + cam_name)
        T_lidar_to_cam = self.lidar_to_cam_transform(cam_name)
        uv, valid = project_pointcloud_to_image(pts, caminfo, T_lidar_to_cam)
        if return_valid:
            return uv, valid
        else:
            return uv[valid]


    def get_sample_files(self, timestamp) -> Dict:
        """Return dict with keys 'image_path', 'lidar_path', 'cam_info', 'segmentation_paths' for the specified timestamp."""
        # find nearest camera image
        img = {cam_name: None for cam_name in self.cameras['color']}  # cam_name -> image path
        depth = {cam_name: None for cam_name in self.cameras['depth']}  # cam_name -> depth image path
        cam_timestamps = {cam_name: None for cam_name in self.cameras['color']}  # cam_name -> timestamp
        segmentation = {cam_name: [] for cam_name in self.cameras['color']}  # cam_name -> list of segmentation paths
        bboxes = {cam_name: [] for cam_name in self.cameras['color']}  # cam_name -> list of bounding box paths
        body = {cam_name: [] for cam_name in self.cameras['color']}  # cam_name -> list of body info paths

        for cam_name, cam_data in self.cameras['color'].items():
            for ts, img_path in cam_data.get("images", []):
                if img[cam_name] is None or abs(ts - timestamp) < abs(cam_timestamps[cam_name] - timestamp):
                    img[cam_name] = img_path
                    cam_timestamps[cam_name] = ts
                    segmentation[cam_name] = self.cameras['color'][cam_name].get("segmentations", {}).get(ts, [])
                    bboxes[cam_name] = self.cameras['color'][cam_name].get("bboxes_2d", {}).get(ts, [])
                    body[cam_name] = self.cameras['color'][cam_name].get("body_3d", {}).get(ts, [])
        for cam_name, cam_data in self.cameras['depth'].items():
            for ts, depth_path in cam_data.get("images", []):
                if depth[cam_name] is None or abs(ts - timestamp) < abs(cam_timestamps[cam_name] - timestamp):
                    depth[cam_name] = depth_path
                    cam_timestamps[cam_name] = ts
        if not any(img.values()):
            return None
        
        # find nearest lidar
        lidar_ts, lidar_path = self.find_nearest_lidar(timestamp)
        pose_path = self.poses.get(lidar_ts)
        bboxes_3d_path = self.bboxes_3d.get(lidar_ts)
        state_path = self.states.get(lidar_ts)

        return {"images": img, "depth": depth, "lidar": lidar_path, "segmentations": segmentation, "bboxes_2d": bboxes, "body_3d": body, "bboxes_3d": bboxes_3d_path, "poses": pose_path, "states": state_path, "timestamps": {"lidar": lidar_ts, **cam_timestamps}}

    def get_sample(self, timestamp) -> Dict:
        """Return dict with keys 'image_path', 'lidar_path', 'cam_info', 'segmentation_paths' for the specified timestamp."""

        sample_files = self.get_sample_files(timestamp)
        if sample_files is None:
            return None

        # load data for each camera
        for cam_name in self.cameras['color']:
            if sample_files["images"][cam_name] is not None:
                sample_files["images"][cam_name] = load_image(sample_files["images"][cam_name])
                sample_files["segmentations"][cam_name] = [load_image(p, image_type="segmentation") for p in sample_files["segmentations"][cam_name]]
                sample_files["bboxes_2d"][cam_name] = load_bounding_boxes(sample_files["bboxes_2d"][cam_name])
                sample_files["body_3d"][cam_name] = [load_body_info(p) for p in sample_files["body_3d"][cam_name]]
        # load depth images for each camera
        for cam_name in self.cameras['depth']:
            if sample_files["depth"][cam_name] is not None:
                sample_files["depth"][cam_name] = load_image(sample_files["depth"][cam_name], image_type="depth")
        # find nearest lidar
        lidar_ts, lidar_path = self.find_nearest_lidar(timestamp)
        lidar_data = load_pointcloud(lidar_path) if lidar_path is not None else None
        # get pose data
        pose_data = load_poses(sample_files["poses"]) if sample_files["poses"] else None
        # get 3D bbox data
        bboxes_3d_data = load_bboxes_3d(sample_files["bboxes_3d"]) if sample_files["bboxes_3d"] else None
        # get state data
        state_data = load_states(sample_files["states"]) if sample_files["states"] else None

        return {"images": sample_files["images"], "depth": sample_files["depth"], "lidar": lidar_data, "segmentations": sample_files["segmentations"], "bboxes_2d": sample_files["bboxes_2d"], "body_3d": sample_files["body_3d"], "poses": pose_data, "bboxes_3d": bboxes_3d_data, "states": state_data, "timestamps": sample_files["timestamps"]}

    def get_transform(self, source_frame, target_frame):
        """Return 4x4 transform matrix mapping source_frame coords to target_frame coords, or None if not computable."""
        if source_frame == 'base_link':
            T_source_to_base = np.eye(4)
        else:
            T_source_to_base = self.transforms_to_base.get(source_frame)
        T_target_to_base = self.transforms_to_base.get(target_frame)
        if T_source_to_base is None or T_target_to_base is None:
            return None
        T_source_target =  np.linalg.inv(T_target_to_base) @ T_source_to_base
        return T_source_target

    def __getitem__(self, idx):
        return self.get_sample(self.timestamps[idx])
    
    def __len__(self):  
        return len(self.timestamps)

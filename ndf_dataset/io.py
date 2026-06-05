from collections import defaultdict
import numpy as np
import yaml
import json
from PIL import Image
import os

from ndf_dataset.transforms import rpy_xyz_to_homogeneous
from ndf_dataset.utility import compute_child_to_base

def load_camera_info(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        obj = yaml.safe_load(fh)
    # camera_matrix 'data' length 9 (3x3)
    cam = {}
    data = obj.get("camera_matrix", {}).get("data", [])
    if len(data) >= 9:
        fx = float(data[0]); fy = float(data[4]); cx = float(data[2]); cy = float(data[5])
        cam["fx"] = fx; cam["fy"] = fy; cam["cx"] = cx; cam["cy"] = cy
        cam["width"] = int(obj.get("image_width", 0))
        cam["height"] = int(obj.get("image_height", 0))
    else:
        # fallback: projection matrix
        proj = obj.get("projection_matrix", {}).get("data", [])
        if len(proj) >= 12:
            fx = float(proj[0]); fy = float(proj[5]); cx = float(proj[2]); cy = float(proj[6])
            cam["fx"] = fx; cam["fy"] = fy; cam["cx"] = cx; cam["cy"] = cy
    return cam

def load_pointcloud(pcd_path, intensity=False):
    """Return Nx3 numpy array of points."""
    if not os.path.exists(pcd_path):
        return None
    
    if pcd_path.lower().endswith(".bin"):
        # Assume binary format with float32 x,y,z
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        if intensity:
            return points
        else:
            return points[:, :3]  # Ignore intensity if present

    with open(pcd_path, "r") as f:
        lines = f.readlines()
    header_ended = False
    pts = []
    for line in lines:
        if not header_ended:
            if line.strip() == "DATA ascii":
                header_ended = True
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                pts.append([x, y, z])
            except ValueError:
                continue
    pts = np.array(pts, dtype=np.float32)
    return pts

def load_image(image_path, image_type="color"):
    """Load image as HxWx3 uint8 numpy array (RGB order)."""
    assert image_type in ["color", "depth", "segmentation"], "image_type must be 'color' or 'depth' or 'segmentation'"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        if image_type == "depth":
            img = img.convert("I")  # 32-bit float for depth
        elif image_type == "segmentation":
            img = img.convert("L")  # 8-bit grayscale for segmentation
        else:
            img = img.convert("RGB")  # Ensure color images are RGB
        img = np.array(img)
        return img
    return None


def load_poses(poses_path):
    if not os.path.exists(poses_path):
        return None
    with open(poses_path, "r") as fh:
        pose_data = json.load(fh)
    poses = list()
    if len(pose_data) == 0:
        return poses
    for pose_id, entry in pose_data.items():
        poses.append({
            'keypoints': np.array(entry),
            'pose_id': pose_id
        })
    return poses

def load_2d_annotations(bbox_path):
    bboxes = load_bounding_boxes(bbox_path)
    for i in range(len(bboxes)):
        bboxes[i]['seg'] = load_image(os.path.join(os.path.dirname(bbox_path), bboxes[i]['seg_fname']), image_type="segmentation")
    return bboxes

def load_body_info(body_info_path):
    if not os.path.exists(body_info_path):
        return None
    with open(body_info_path, "r") as fh:
        body_info = json.load(fh)
    result = dict()
    result['pred_vertices'] = np.array(body_info['pred_vertices'])
    result['pred_keypoints_3d'] = np.array(body_info['pred_keypoints_3d'])
    result['pred_cam_t'] = np.array(body_info['pred_cam_t'])
    return result

def load_bounding_boxes(bbox_path):
    if not os.path.exists(bbox_path):
        return list()
    with open(bbox_path, "r") as fh:
        bbox_data = json.load(fh)
    return bbox_data

def load_bboxes_3d(bbox_path):
    if not os.path.exists(bbox_path):
        return list()
    with open(bbox_path, "r") as fh:
        bbox_data = json.load(fh)
    results = []
    for entry in bbox_data:
        results.append({
            'bbox_label': entry.get('bbox_label', 'unknown'),
            'bbox_center': np.array(entry['bbox_center']),
            'bbox_size': np.array(entry['bbox_size']),
            'bbox_rotation': np.array(entry['bbox_rotation']),
            'bbox_corners': np.array(entry['bbox_corners']),
            'pose_id': entry.get('pose_id', None)
        })
    return results

def load_transforms(tt):
    with open(tt, "r") as fh:
        json_obj = json.load(fh)
    transforms_child_to_parent = dict() 
    transforms_to_base = dict()
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
            transforms_child_to_parent[child] = (parent, T)
    # Now for each top-level key (frame group) compute transform from that frame -> base_link
    for key in json_obj.keys():
        T = compute_child_to_base(transforms_child_to_parent, key)
        if T is not None:
            transforms_to_base[key] = T

    return transforms_child_to_parent, transforms_to_base

def load_states(states_path):
    if not os.path.exists(states_path):
        return None
    with open(states_path, "r") as fh:
        state_data = json.load(fh)
    states = defaultdict(dict)
    if len(state_data) == 0:
        return states
    actions_dict = state_data.get("actions", {})
    for action_id, entry in actions_dict.items():
        states[action_id]['action'] = entry
    states_dict = state_data.get("states", {})
    for state_id, entry in states_dict.items():
        states[state_id]['state'] = entry
    return states
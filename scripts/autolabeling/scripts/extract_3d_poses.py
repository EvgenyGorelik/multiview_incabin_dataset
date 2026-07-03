from scipy.spatial.transform import Rotation as R
import numpy as np
from argparse import ArgumentParser
import json
import os
from tqdm import tqdm

from ndf_dataset import NDFDataset
from ndf_dataset.io import load_poses

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)



def compute_pose_orientation(keypoints):
    p9 = keypoints[9]
    p10 = keypoints[10]

    # Project both points to ground plane (z = 0)
    p9_ground = (float(p9[0]), float(p9[1]), 0.0)
    p10_ground = (float(p10[0]), float(p10[1]), 0.0)

    # Direction on ground plane from keypoint 9 -> 10
    dx = p10_ground[0] - p9_ground[0]
    dy = p10_ground[1] - p9_ground[1]

    # One valid in-plane normal (the opposite sign is also valid)
    normal_vector = (-dy, dx, 0.0)

    # Unit normal
    norm = (normal_vector[0] ** 2 + normal_vector[1] ** 2) ** 0.5
    normal_unit_vector = (
        normal_vector[0] / norm,
        normal_vector[1] / norm,
        0.0,
    ) if norm > 0 else (0.0, 0.0, 0.0)
    
    pose_center = (p9 + p10) / 2
    pose_heading = pose_center + np.array(normal_unit_vector) * 0.1
    return pose_center, pose_heading

def calculate_bounding_box(keypoints):
    center, orientation = compute_pose_orientation(keypoints)
    source_vec = np.array([0.0, 1.0, 0.0])
    target_vec = orientation - center

    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        raise ValueError("orientation - center is zero; rotation is undefined.")

    target_vec = target_vec / target_norm
    dot = np.clip(np.dot(source_vec, target_vec), -1.0, 1.0)

    if np.isclose(dot, 1.0):
        angle = 0.0
        rotation = R.identity()
    elif np.isclose(dot, -1.0):
        angle = np.pi
        rotation = R.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0]))
    else:
        axis = np.cross(source_vec, target_vec)
        axis /= np.linalg.norm(axis)
        if not np.isclose(axis[0], 0.0) or not np.isclose(axis[1], 0.0):
            print("Rotation axis has non-zero x or y component.")
        angle = np.arccos(dot)
        rotation = R.from_rotvec(axis * angle)

    rotation_matrix = rotation.as_matrix()
    rotation_quat = rotation.as_quat()  # [x, y, z, w]

    T_origin_to_pose = np.eye(4)
    T_origin_to_pose[:3, :3] = rotation_matrix
    T_origin_to_pose[:3, 3] = center
    T_pose_to_origin = np.linalg.inv(T_origin_to_pose)

    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])  # (K, 4)
    keypoints_transformed = (T_pose_to_origin @ keypoints_homogeneous.T).T[:, :3]  # (K, 3)


    box_min = np.min(keypoints_transformed, axis=0)
    box_max = np.max(keypoints_transformed, axis=0)
    box_max[2] += 0.2 # add some height for eye-head distance
    box_corners = np.array([[box_min[0], box_min[1], box_min[2]],
                            [box_max[0], box_min[1], box_min[2]],
                            [box_max[0], box_max[1], box_min[2]],
                            [box_min[0], box_max[1], box_min[2]],
                            [box_min[0], box_min[1], box_max[2]],
                            [box_max[0], box_min[1], box_max[2]],
                            [box_max[0], box_max[1], box_max[2]],
                            [box_min[0], box_max[1], box_max[2]]])
    bbox_corners = (T_origin_to_pose @ np.hstack([box_corners, np.ones((box_corners.shape[0], 1))]).T).T[:, :3]
    bbox_center = bbox_corners.mean(axis=0)
    bbox_size = box_max - box_min
    return bbox_center, bbox_size, rotation_quat, bbox_corners

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=str, help="Path to NDF dataset")
    args = parser.parse_args()

    ds = NDFDataset(args.root)
    for ts, pose_path in tqdm(ds.poses):
        poses = load_poses(pose_path)
        results = []
        for pose in poses:
            bbox_center, bbox_size, rotation_quat, bbox_corners = calculate_bounding_box(pose['keypoints'])
            results.append({
                'bbox_label': 'human',
                'bbox_center': bbox_center.tolist(),
                'bbox_size': bbox_size.tolist(),
                'bbox_rotation': rotation_quat.tolist(),
                'bbox_corners': bbox_corners.tolist(),
                'pose_id': pose['pose_id']
            })
        out_path = pose_path.replace("poses", "bboxes_3d")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, cls=NumpyEncoder)

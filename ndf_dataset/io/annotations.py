"""Readers for current 3D box annotations and legacy annotation formats."""

import json
import os
from collections import defaultdict

import numpy as np

from ndf_dataset.models.bounding_box import BoundingBox3D
from .camera import load_image

def load_bboxes_3d(bbox_path):
    if not os.path.exists(bbox_path):
        return list()
    with open(bbox_path, "r") as fh:
        bbox_data = json.load(fh)
    results = []
    for entry in bbox_data:
        results.append(BoundingBox3D(
            bbox_label=entry.get('bbox_label', 'unknown'),
            bbox_center=np.array(entry['bbox_center']),
            bbox_size=np.array(entry['bbox_size']),
            bbox_rotation=np.array(entry['bbox_rotation']),
            bbox_corners=np.array(entry['bbox_corners']),
            obj_id=entry.get('obj_id'),
            attr=entry.get('attr'),
            action=entry.get('action'),
            velocity=np.array(entry['velocity']) if entry.get('velocity') is not None else None,
        ))
    return results



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

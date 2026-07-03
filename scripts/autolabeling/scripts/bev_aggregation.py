import json
import cv2
import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import trange
from yolox.tracker.byte_tracker import BYTETracker, STrack
from scipy.optimize import minimize

from ndf_dataset import NDFDataset
from ndf_dataset.transforms import project_points, project_body_to_base
from ndf_dataset.utility import MHR_CONNECTIONS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)
    

def bevbbox(vertices):
    bev_xy = vertices[:, [0, 1]]
    bbox = np.array([
        [bev_xy[:, 0].min(), bev_xy[:, 1].min()],
        [bev_xy[:, 0].max(), bev_xy[:, 1].min()],
        [bev_xy[:, 0].max(), bev_xy[:, 1].max()],
        [bev_xy[:, 0].min(), bev_xy[:, 1].max()],
    ])
    return bbox

class BYTETrackerArgs:
    def __init__(self):
        self.track_thresh = 0.3
        self.mot20 = False
        self.track_buffer = 30
        self.match_thresh = 0.3

def encode_bev_bbox(bevbbox, margins=[-20, -20, 20, 20]):
    canvas_margins = np.array(margins) # tl, br margins in meters for x and y (left, top, right, bottom)
    canvas_size = canvas_margins[2:] - canvas_margins[:2] 
    bbox_encoded = (bevbbox - canvas_margins[:2]) / canvas_size
    bbox_xy = np.array([bbox_encoded[:, 0].min(), bbox_encoded[:, 1].min(), bbox_encoded[:, 0].max(), bbox_encoded[:, 1].max()])
    return bbox_xy

def decode_bev_bbox(encoded_bbox, margins=[-20, -20, 20, 20]):
    canvas_margins = np.array(margins)
    canvas_size = canvas_margins[2:] - canvas_margins[:2]
    bevbbox = encoded_bbox * canvas_size + canvas_margins[:2]
    bbox = np.array([
        bevbbox[:2],
        [bevbbox[2], bevbbox[1]],
        bevbbox[2:],
        [bevbbox[0], bevbbox[3]],
    ])
    return bevbbox

def camera_constrained_clustering(detections, eps, prev_clusters=None):
    clusters = []  # each item: {"obj_ids": [], "cams": set(), "center": np.ndarray}
    centers = np.array([center for _, _, center in detections])
    obj_ids = [obj_id for obj_id, _, _ in detections]

    for obj_id, cam_src, center in detections:
        best_cluster = None
        best_dist = None

        for i, cluster in enumerate(clusters):
            if cam_src in cluster["cams"]:
                continue
            dist = np.linalg.norm(center - cluster["center"])
            if dist <= eps and (best_dist is None or dist < best_dist):
                best_cluster = i
                best_dist = dist

        if best_cluster is None:
            clusters.append({"obj_ids": [obj_id], "cams": {cam_src}, "center": center.copy()})
        else:
            cluster = clusters[best_cluster]
            cluster["obj_ids"].append(obj_id)
            cluster["cams"].add(cam_src)
            cluster["center"] = centers[[obj_ids.index(oid) for oid in cluster["obj_ids"]]].mean(axis=0)
    
    selected_obj_by_cluster = {}
    for lbl, cluster in enumerate(clusters):
        cluster_center = cluster["center"]
        cluster_obj_ids = cluster["obj_ids"]
        idxs = np.array([obj_ids.index(obj_id) for obj_id in cluster_obj_ids])
        dists = np.linalg.norm(centers[idxs] - cluster_center, axis=1)
        # If previous clusters provided, prefer object closest to previous cluster center
        if prev_clusters is not None and len(prev_clusters) > 0:
            # find closest previous cluster center to this cluster
            prev_centers = np.array([c["center"] for c in prev_clusters])
            prev_dists = np.linalg.norm(prev_centers - cluster_center, axis=1)
            prev_idx = int(np.argmin(prev_dists))
            prev_center = prev_centers[prev_idx]
            # choose object in this cluster whose center is closest to previous center
            dists_to_prev = np.linalg.norm(centers[idxs] - prev_center, axis=1)
            chosen = int(np.argmin(dists_to_prev))
        else:
            chosen = int(np.argmin(dists))
        selected_obj_by_cluster[lbl] = cluster_obj_ids[chosen]
    return selected_obj_by_cluster, clusters


def projection_error(keypoints_3d, projected_keypoints_2d, cam_intrinsics, cam_to_base):
    projected = project_points(keypoints_3d, cam_intrinsics, np.linalg.inv(cam_to_base))
    error = np.linalg.norm(projected - projected_keypoints_2d)
    return error

def optimize_pose(keypoints_3d, projected_keypoints_2d, cam_intrinsics, cam_to_base):
    def objective(x):
        translation = x.reshape(3)
        transformed_keypoints = keypoints_3d + translation
        error = 0.0
        for cam, proj_kp_2d in projected_keypoints_2d.items():
            error += projection_error(transformed_keypoints, proj_kp_2d, cam_intrinsics[cam], cam_to_base[cam])
        return error
    x0 = np.zeros(3)  # initial translation guess
    res = minimize(objective, x0, method='Nelder-Mead')
    optimized_translation = res.x
    optimized_keypoints = keypoints_3d + optimized_translation.reshape(3)
    return optimized_keypoints


def refine_poses(bodies_in_base, keypoints_in_base, cams_source, cam_intrinsics, cam_extrinsics, eps=3.0, prev_clusters=None, dist_thresh=8.0):
    detections = [
        (obj_id, cams_source.get(obj_id), bodies_in_base[obj_id][:, :2].mean(axis=0))
        for obj_id in bodies_in_base.keys()
    ]
    selected_obj_by_cluster, clusters = camera_constrained_clustering(detections, eps, prev_clusters)

    # Remove clusters with only one entry that are too far from camera
    clusters_to_remove = []
    for lbl, cluster in enumerate(clusters):
        if len(cluster["obj_ids"]) == 1:
            dist_to_camera = np.linalg.norm(cluster["center"] - cam_extrinsics[cams_source[cluster["obj_ids"][0]]][:2, 3])
            if dist_to_camera > dist_thresh:
                clusters_to_remove.append(lbl)
    
    for lbl in reversed(sorted(clusters_to_remove)):
        del selected_obj_by_cluster[lbl]
        clusters.pop(lbl)
    
    # Reindex selected_obj_by_cluster after removal
    selected_obj_by_cluster = {new_lbl: obj_id for new_lbl, obj_id in enumerate(selected_obj_by_cluster.values())}

    refined_poses = {}
    for lbl, selected_id in selected_obj_by_cluster.items():
        keypoints_before = keypoints_in_base[selected_id].copy()

        cluster_obj_ids = clusters[lbl]["obj_ids"] if lbl < len(clusters) else [selected_id]
        projected_keypoints_2d = {}
        for oid in cluster_obj_ids:
            cam = cams_source.get(oid)
            if cam is None:
                continue
            intr = cam_intrinsics[cam]
            inv_cam_tf = np.linalg.inv(cam_extrinsics[cam])
            projected_keypoints_2d[cam] = project_points(keypoints_in_base[oid], intr, inv_cam_tf)
        refined_poses[lbl] = optimize_pose(keypoints_before, projected_keypoints_2d, cam_intrinsics, cam_extrinsics)

    return refined_poses, clusters

@staticmethod
def init_count(start_id):
    STrack._count = start_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('root', type=str, help='Root directory of the NDF dataset')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the clustering results')
    parser.add_argument('--eps', type=float, default=1.0, help='DBSCAN eps parameter for clustering BEV centers')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for processing frames')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index for processing frames')
    parser.add_argument('--start_cluster_id', type=int, default=0, help='Starting cluster ID for tracking')
    parser.add_argument('--dist_thresh', type=float, default=8.0, help='Distance threshold for removing clusters')
    args = parser.parse_args()
    ds = NDFDataset(args.root)
    os.makedirs(os.path.join(ds.root, 'poses'), exist_ok=True)

    tracker = BYTETracker(args=BYTETrackerArgs())
    init_count(args.start_cluster_id)  # Initialize BYTETracker's internal ID counter

    cams_source = {}
    prev_clusters = None

    cam_intrinsics = {cam: ds.cameras['color'][cam]['intrinsics'] for cam in ds.cameras['color'].keys()}
    cam_extrinsics = {cam: ds.transforms_to_base[ds.frame_id_map[cam]] for cam in ds.cameras['color'].keys()}

    for idx in trange(len(ds)):
        if idx < args.start_idx:
            continue
        if args.end_idx is not None and idx >= args.end_idx:
            break
        ts, sample = ds[idx]

        poses = {}
        bodies_in_base = {}
        keypoints_in_base = {}
        for cam in ds.cameras['color'].keys():
            for body, bbox in zip(sample['body_3d'][cam], sample['bboxes_2d'][cam][0]):
                body_base = project_body_to_base(body, ds.transforms_to_base[ds.frame_id_map[cam]], 'pred_vertices')
                bodies_in_base[bbox['obj_id']] = body_base
                keypoints_in_base[bbox['obj_id']] = project_body_to_base(body, ds.transforms_to_base[ds.frame_id_map[cam]], 'pred_keypoints_3d')
                cams_source[bbox['obj_id']] = cam

        if bodies_in_base:
            refined_poses, prev_clusters = refine_poses(bodies_in_base, keypoints_in_base, cams_source, cam_intrinsics, cam_extrinsics, eps=args.eps, dist_thresh=args.dist_thresh, prev_clusters=prev_clusters)
            
            
        if bodies_in_base and len(refined_poses) > 0:
            bboxes_encoded = np.array([encode_bev_bbox(bevbbox(pose)) for pose in refined_poses.values()])
            bboxes_2d_converted = torch.tensor([bbox for bbox in bboxes_encoded])
            bboxes_2d_converted = torch.hstack([bboxes_2d_converted, torch.ones(len(bboxes_2d_converted), 1)])  # Add confidence
            tracker_outputs = tracker.update(bboxes_2d_converted, img_info=(1, 1), img_size=(1, 1))
            for output in tracker_outputs:
                tid = output.track_id
                cur_id = output.current_idx
                poses[tid] = refined_poses[cur_id].tolist()  # Store the optimized keypoints for this track
        else:
            tracker.update(torch.empty((0, 5)), img_info=(1, 1), img_size=(1, 1))
        with open(os.path.join(ds.root, 'poses', f'{ts}.json'), 'w+') as f:
            json.dump(poses, f, cls=NumpyEncoder)
        if args.visualize:
            cam_keys = list(ds.cameras['color'].keys())
            stacked_img = {cam: sample['images'][cam].copy() for cam in cam_keys}
            for cam in ds.cameras['color'].keys():
                for pose_id, pose in poses.items():
                    keypoints_3d = np.array(pose)
                    projected_2d = project_points(keypoints_3d, cam_intrinsics[cam], np.linalg.inv(cam_extrinsics[cam]))
                    for x, y in projected_2d:
                        if 0 <= x < stacked_img[cam].shape[1] and 0 <= y < stacked_img[cam].shape[0]:
                            cv2.circle(stacked_img[cam], (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(stacked_img[cam], f'ID: {pose_id}', (int(projected_2d[0, 0]), int(projected_2d[0, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            stacked_img = [stacked_img[cam] for cam in cam_keys]
            top_row = np.hstack(stacked_img[:2])
            bottom_row = np.hstack(stacked_img[2:4])
            viz_img = np.vstack([top_row, bottom_row])
            viz_img = cv2.putText(viz_img, f'Time: {ts} ({idx})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Refined Poses', viz_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

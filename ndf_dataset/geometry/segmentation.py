"""Point cloud selection using image segmentation."""

import numpy as np

from .projection import project_pointcloud_to_image

def segmented_pointcloud(seg_mask, pcd, caminfo, T_cam_to_baselink, depth_threshold=0.5):
    """Return Nx3 array of points from the lidar that project onto the segmented area of the image.
        - cam_name: camera folder name (e.g., "camera_1")
        - seg_path: path to segmentation mask image (same size as camera image, nonzero pixels indicate segmented area)
        - lidar_path: path to lidar .pcd file
        - depth_threshold: max allowed depth difference from median to filter out outliers (meters)
    """
    uv, valid = project_pointcloud_to_image(pcd, caminfo, T_cam_to_baselink)
    uv_valid = uv[valid]
    depths = uv_valid[:, 2]

    uv_seg = list()
    for i in range(uv_valid.shape[0]):
        u_int = int(np.floor(uv_valid[i, 0]))
        v_int = int(np.floor(uv_valid[i, 1]))
        if 0 <= uv_valid[i, 0] < seg_mask.shape[1] and 0 <= uv_valid[i, 1] < seg_mask.shape[0] and seg_mask[v_int, u_int] > 0:
            uv_seg.append([uv_valid[i, 0], uv_valid[i, 1], depths[i]])

    if len(uv_seg) < 2:
        return np.empty((0, 3))  # not enough points to compute median, return empty

    uv_seg = np.array(uv_seg)
    uv_median = np.median(uv_seg[:, 2])
    uv_mask = uv_seg[np.abs(uv_seg[:, 2] - uv_median) < depth_threshold, :]

    cam_K = np.array([[caminfo["fx"], 0, caminfo["cx"]],
                      [0, caminfo["fy"], caminfo["cy"]],
                      [0, 0, 1]])

    xyz_cam = (np.linalg.inv(cam_K) @ np.hstack([uv_mask[:, :2], np.ones((uv_mask.shape[0], 1))]).T) * uv_mask[:, 2:3].T
    xyz_baselink = (T_cam_to_baselink @ np.vstack((xyz_cam, np.ones((1, xyz_cam.shape[1])))))[:3].T
    return xyz_baselink

"""Camera intrinsics and projection onto image planes."""

import numpy as np

def get_intrinsics(cam_info):
    K = np.eye(3)
    K[0, 0] = cam_info.get('fx', 1.0)
    K[1, 1] = cam_info.get('fy', 1.0)
    K[0, 2] = cam_info.get('cx', 0.0)
    K[1, 2] = cam_info.get('cy', 0.0)
    return K



def project_pointcloud_to_image(pts, caminfo, T_lidar_to_cam):
    """Project pointcloud to image plane using intrinsics and transform.
        Returns Nx3 array of (u, v, depth) for each point.
    """
    # convert to homogeneous and transform
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))]).T  # 4xN
    cam_pts_h = (T_lidar_to_cam @ hom).T  # Nx4
    cam_xyz = cam_pts_h[:, :3]

    # projection (pinhole)
    fx = caminfo["fx"]; fy = caminfo["fy"]; cx = caminfo["cx"]; cy = caminfo["cy"]
    x = cam_xyz[:, 0]; y = cam_xyz[:, 1]; z = cam_xyz[:, 2]
    u = (fx * (x / z) + cx)
    v = (fy * (y / z) + cy)

    uv = np.vstack([u, v, z]).T

    valid = np.logical_and(z > 0, np.logical_and(u > 0, np.logical_and(v > 0, np.logical_and(u < caminfo["width"], v < caminfo["height"]))))

    return uv, valid



def project_points(pts, cam_int, T_base_to_cam):
    """Project 3D points in base frame to 2D image plane using camera intrinsics and transform."""
    pts_cam = (T_base_to_cam[:3, :3] @ pts.T + T_base_to_cam[:3, 3:4]).T
    pts_proj_h = cam_int @ pts_cam.T
    pts_proj_2d = (pts_proj_h[:2] / pts_proj_h[2]).T
    return pts_proj_2d



def project_poses_to_camera(poses, camera, transform, key='vertices', return_valid=False):
    """Project legacy pose arrays using a base-to-camera transformation."""
    results = []
    for pose in poses:
        uv, valid = project_pointcloud_to_image(np.array(pose[key]), camera.cam_info, transform)
        results.append((uv, valid) if return_valid else uv[valid])
    return results

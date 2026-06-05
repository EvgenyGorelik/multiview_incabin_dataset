import numpy as np

def rpy_xyz_to_homogeneous(rpy, xyz):
    """Create 4x4 homogeneous matrix from RPY (radians) and XYZ (meters).
       Rotation order: roll (X), pitch (Y), yaw (Z).
    """
    r, p, y = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def xyzrpy_to_transform(xyzrpy: np.ndarray):
    x, y, z = xyzrpy[:3]
    roll, pitch, yaw = xyzrpy[3:]
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def xyzrpy_from_transform(T: np.ndarray):
    x, y, z = T[:3, 3]
    roll = np.arctan2(T[2, 1], T[2, 2])
    pitch = np.arctan2(-T[2, 0], np.sqrt(T[0, 0]**2 + T[1, 0]**2))
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, z, roll, pitch, yaw])

def corners_to_bbox(corners):
    """Convert 8 corners of a 3D bounding box to (x, y, z, l, w, h, yaw) format.
       Assumes corners are in the order: front-left-top, front-right-top, rear-right-top, rear-left-top,
                                        front-left-bottom, front-right-bottom, rear-right-bottom, rear-left-bottom.
    """
    x = np.mean(corners[:, 0])
    y = np.mean(corners[:, 1])
    z = np.mean(corners[:, 2])
    l = np.linalg.norm(corners[0] - corners[1])  # length along x-axis
    w = np.linalg.norm(corners[0] - corners[3])  # width along y-axis
    h = np.linalg.norm(corners[0] - corners[4])  # height along z-axis
    yaw = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0])  # yaw from front edge
    return np.array([x, y, z, l, w, h, yaw])


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

def transform_pointcloud(pts, T):
    """Apply homogeneous transform T to pointcloud pts."""
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4
    transformed_hom = (T @ hom.T).T  # Nx4
    return transformed_hom[:, :3]


def project_points(pts, cam_int, T_base_to_cam):
    """Project 3D points in base frame to 2D image plane using camera intrinsics and transform."""
    pts_cam = (T_base_to_cam[:3, :3] @ pts.T + T_base_to_cam[:3, 3:4]).T
    pts_proj_h = cam_int @ pts_cam.T
    pts_proj_2d = (pts_proj_h[:2] / pts_proj_h[2]).T
    return pts_proj_2d


def project_body_to_base(body_results: dict, T_cam_to_base: np.ndarray, key="pred_keypoints_3d"):
    assert key in body_results, f"Key '{key}' not found in body_results, available keys: {list(body_results.keys())}"
    verts = body_results[key]  # (K, 3)
    cam_t = body_results["pred_cam_t"]     # (3,)

    verts_cam = verts + cam_t[None, :]  # (K, 3)

    verts_3d_base = (T_cam_to_base[:3, :3] @ verts_cam.T + T_cam_to_base[:3, 3:4]).T  # (K, 3)
    return verts_3d_base
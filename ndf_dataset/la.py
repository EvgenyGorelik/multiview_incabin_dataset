import numpy as np
from ndf_dataset.transforms import project_pointcloud_to_image


def closest_points_between_rays(p1, v1, p2, v2):
    """Compute closest points c1, c2 on two rays (p1 + t*v1, p2 + s*v2). Returns None, None if rays are parallel."""
    v1_dot_v2 = np.dot(v1, v2)
    if abs(v1_dot_v2) < 1e-6:
        print("Rays are parallel, cannot find a unique closest point.")
        return None, None
    else:
        dP = p2 - p1
        A = np.array([[np.dot(v1, v1), -np.dot(v1, v2)],
                      [np.dot(v1, v2), -np.dot(v2, v2)]])
        b = np.array([np.dot(v1, dP), np.dot(v2, dP)])
        try:
            t1, t2 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Singular system, skipping.")
            return None, None
        else:
            c1 = p1 + t1 * v1
            c2 = p2 + t2 * v2
            return c1, c2
        

def calculate_rays(r, c, K, T_to_base, z=1.0):
    """Calculate ray origin and direction in base frame for pixel (r, c) given camera intrinsics K and transform to base."""
    d = np.ones_like(c)
    x = (c - K[0, 2]) / K[0, 0]
    y = (r - K[1, 2]) / K[1, 1]
    p = np.stack([x * z, y * z, z * d, d])
    ray_direction = p[:3] / np.linalg.norm(p[:3], axis=0)
    ray_direction_base = (T_to_base[:3, :3] @ ray_direction).T
    ray_origin_base = np.broadcast_to(T_to_base[:3, 3], ray_direction_base.shape)
    return ray_origin_base, ray_direction_base



def calculate_intersections(T_source_to_base, T_target_to_base, rc_s, rc_t, K_source, K_target, intersection_dist=0.01):
    """Calculate intersection points between rays from source and target cameras. Returns intersection points, non-intersection points, and matched/unmatched pixel pairs."""
    intersection_points = []
    non_intersection_points = []
    matched_points = []
    unmatched_points = []
    ray_origin_source_base, ray_direction_source_base = calculate_rays(rc_s[:, 0], rc_s[:, 1], K_source, T_source_to_base)
    ray_origin_target_base, ray_direction_target_base = calculate_rays(rc_t[:, 0], rc_t[:, 1], K_target, T_target_to_base)
    for i in range(ray_origin_source_base.shape[0]):
        for j in range(ray_origin_target_base.shape[0]):
            # solve for closest points between the two rays in the base frame
            c1, c2 = closest_points_between_rays(ray_origin_source_base[i], ray_direction_source_base[i], ray_origin_target_base[j], ray_direction_target_base[j])
            if c1 is not None and c2 is not None:
                mid = (c1 + c2) / 2.0
                if np.linalg.norm(c1 - c2) < intersection_dist:
                    intersection_points.append(mid)
                    matched_points.append((rc_s[i], rc_t[j]))
                else:
                    non_intersection_points.append(mid)
                    unmatched_points.append((rc_s[i], rc_t[j]))
    return np.array(intersection_points), np.array(non_intersection_points), matched_points, unmatched_points


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
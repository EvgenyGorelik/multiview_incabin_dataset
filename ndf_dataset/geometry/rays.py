"""Camera rays and triangulation."""

import numpy as np

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

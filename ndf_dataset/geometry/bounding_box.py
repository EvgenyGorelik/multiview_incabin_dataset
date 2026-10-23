"""Bounding box geometry conversions."""

import numpy as np

def vertices_to_3d_bbox(verts):
    min_pt = np.min(verts, axis=0)
    max_pt = np.max(verts, axis=0)
    bbox_corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
    ])
    return bbox_corners



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

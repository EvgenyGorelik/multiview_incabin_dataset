"""Rigid transformations and coordinate frame chains."""

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



def transform_pointcloud(pts, T):
    """Apply homogeneous transform T to pointcloud pts."""
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4
    transformed_hom = (T @ hom.T).T  # Nx4
    return transformed_hom[:, :3]



def compute_child_to_base(transforms_child_to_parent, child):
    """Walk child->parent links until base_link or until can't continue.
        Compose transforms so final T maps original child coords -> base_link coords.
    """
    cur = child
    mats = []
    visited = set()
    while True:
        if cur in visited:
            return None
        visited.add(cur)
        entry = transforms_child_to_parent.get(cur)
        if entry is None:
            return None
        parent, T_child_to_parent = entry
        mats.append(T_child_to_parent)  # child -> parent
        if parent == "base_link":
            # compose: total = T_parent_n * ... * T_child_to_parent
            total = np.eye(4)
            for M in mats:
                total = M @ total
            return total
        # next child is parent frame (walk upward)
        cur = parent
        if cur is None:
            return None

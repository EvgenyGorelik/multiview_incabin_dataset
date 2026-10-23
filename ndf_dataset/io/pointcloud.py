"""Point cloud readers for PCD, binary scans and PLY maps."""

import os

import numpy as np

def load_pointcloud(pcd_path, intensity=False):
    """Return Nx3 numpy array of points."""
    if not os.path.exists(pcd_path):
        return None

    if os.fspath(pcd_path).lower().endswith(".ply"):
        from plyfile import PlyData

        vertices = PlyData.read(os.fspath(pcd_path))['vertex']
        return np.column_stack([vertices[axis] for axis in ('x', 'y', 'z')])

    if os.fspath(pcd_path).lower().endswith(".bin"):
        # Assume binary format with float32 x,y,z
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        if intensity:
            return points
        else:
            return points[:, :3]  # Ignore intensity if present

    with open(pcd_path, "r") as f:
        lines = f.readlines()
    header_ended = False
    pts = []
    for line in lines:
        if not header_ended:
            if line.strip() == "DATA ascii":
                header_ended = True
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                pts.append([x, y, z])
            except ValueError:
                continue
    pts = np.array(pts, dtype=np.float32)
    return pts

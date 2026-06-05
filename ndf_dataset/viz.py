import open3d as o3d
import numpy as np


def visualize_pointcloud_with_detections(points, bboxes):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use x,y,z for points
    if points.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(points[:, 3:4] / 255.0, (1, 3)))  # Use intensity for color

    # Create Open3D geometries for detections
    geometries = [pcd]
    for bbox in bboxes:
        center = bbox[:3] # Assuming bbox is in the format [x, y, z, l, w, h, yaw]
        size = bbox[3:6]
        yaw = bbox[6]
        box = o3d.geometry.OrientedBoundingBox(center, np.eye(3), size)
        box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]), center)
        box.color = (1, 0, 0)  # Red color for bounding boxes
        geometries.append(box)

    # Visualize
    o3d.visualization.draw_geometries(geometries)

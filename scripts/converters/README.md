## ROSBag Converter

Converts a ros bag to `NDFDataset` format.


<details>

<summary>Example of rosbag info</summary>

```
Files:             rosbag2_2026_05_21-09_40_56_0.mcap
Bag size:          17.9 GiB
Storage id:        mcap
ROS Distro:        jazzy
Duration:          127.724184844s
Start:             May 21 2026 09:40:56.835070054 (1779356456.835070054)
End:               May 21 2026 09:43:04.559254898 (1779356584.559254898)
Messages:          29610
Topic information: Topic: /edge_0/camera/color/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1905 | Serialization Format: cdr
                   Topic: /edge_0/camera/color/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1572 | Serialization Format: cdr
                   Topic: /edge_0/camera/depth/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 2083 | Serialization Format: cdr
                   Topic: /edge_0/camera/depth/image_rect_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1788 | Serialization Format: cdr
                   Topic: /edge_1/camera/color/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1911 | Serialization Format: cdr
                   Topic: /edge_1/camera/color/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1631 | Serialization Format: cdr
                   Topic: /edge_1/camera/depth/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1944 | Serialization Format: cdr
                   Topic: /edge_1/camera/depth/image_rect_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1636 | Serialization Format: cdr
                   Topic: /edge_3/camera/color/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1904 | Serialization Format: cdr
                   Topic: /edge_3/camera/color/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1553 | Serialization Format: cdr
                   Topic: /edge_3/camera/depth/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1911 | Serialization Format: cdr
                   Topic: /edge_3/camera/depth/image_rect_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1569 | Serialization Format: cdr
                   Topic: /edge_5/camera/color/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1884 | Serialization Format: cdr
                   Topic: /edge_5/camera/color/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1553 | Serialization Format: cdr
                   Topic: /edge_5/camera/depth/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 1899 | Serialization Format: cdr
                   Topic: /edge_5/camera/depth/image_rect_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 1589 | Serialization Format: cdr
                   Topic: /ouster/points | Type: sensor_msgs/msg/PointCloud2 | Count: 1278 | Serialization Format: cdr
Service:           0
Service information: 
``` 
</details>

Can only be used with a ros2 installation (Tested with `jazzy`)

Convert using:
```
python convert_rosbag.py
```


## NuScenes Export

Export `NDFDataset` to NuScenes Format (Version `v1.0-mini`)

```
python export_nuscenes.py <path/to/dataset>
```
import os
import json
import uuid
import shutil
from argparse import ArgumentParser
from fastjsonschema import VERSION
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from ndf_dataset import NDFDataset
from ndf_dataset.io import load_bboxes_3d, load_pointcloud

NUSCENES_VAL_SPLIT = ["scene-0103", "scene-0916"]
NUSCENES_TRAIN_SPLIT = [
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-1077",
    "scene-1094",
    "scene-1100",
]


VERSION = "v1.0-mini"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


if __name__ == "__main__":
    parser = ArgumentParser("Export NDF dataset to nuScenes format")
    parser.add_argument("root", type=str, help="Path to input NDF dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="nuscenes_out",
        help="Path to output directory for nuScenes format",
    )
    args = parser.parse_args()

    ds = NDFDataset(args.root)

    camera_names = {}

    for modality in ds.modalities["cameras"]:
        if "depth" in modality:
            continue
        camera_names[modality] = f"CAM_{modality.upper()}"
        os.makedirs(
            os.path.join(args.output, f"samples/{camera_names[modality]}"),
            exist_ok=True,
        )

    assert len(ds.modalities["lidar"]) == 1, "Only one lidar modality is supported"

    lidar_channel = "LIDAR_TOP"
    os.makedirs(os.path.join(args.output, f"samples/{lidar_channel}"), exist_ok=True)

    os.makedirs(os.path.join(args.output, VERSION), exist_ok=True)

    def token():
        return str(uuid.uuid4().hex)

    def parse_transform(tf_matrix):
        tf_matrix = np.array(tf_matrix)
        xyz = tf_matrix[:3, 3].tolist()
        quat = R.from_matrix(tf_matrix[:3, :3]).as_quat(scalar_first=True).tolist()
        return xyz, quat

    def quat_to_nuscenes(quat):
        quat = np.asarray(quat, dtype=np.float64)
        r_in = R.from_quat(quat)  # input: [x, y, z, w]
        r_z90 = R.from_euler("z", 90, degrees=True)
        quat = (r_z90 * r_in).as_quat()  # still [x, y, z, w]
        return [quat[3], quat[0], quat[1], quat[2]]

    # lidar calibration
    lidar_xyz, lidar_quat = parse_transform(
        ds.transforms_to_base[ds.frame_id_map["lidar"]]
    )

    # ------------------------
    # nuScenes tables
    # ------------------------
    attribute = []
    samples = []
    category = []
    sample_data = []
    ego_pose = []
    calibrated_sensor = []
    annotations = []
    instances = []
    visibility = []  # not used, but required by format
    map_json = []  # not used, but required by format

    # static tokens
    log_token = token()
    map_json.append(
        {
            "category": "empty_map",
            "token": token(),
            "filename": "maps/empty_map.png",
            "log_tokens": [log_token],
        }
    )

    os.makedirs(os.path.join(args.output, "maps"), exist_ok=True)
    cv2.imwrite(
        os.path.join(args.output, "maps/empty_map.png"),
        np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    # ------------------------
    # scene
    # ------------------------
    scene = [
        {
            "token": token(),
            "first_sample_token": None,
            "last_sample_token": None,
            "log_token": log_token,
            "nbr_samples": 0,
            "name": n,
            "description": "",
        }
        for n in np.random.permutation(
            NUSCENES_TRAIN_SPLIT + NUSCENES_VAL_SPLIT
        ).tolist()
    ]

    num_samples_per_scene = np.ceil(len(ds) / len(scene)).astype(int)

    lidar_sensor_token = token()

    cam_sensor_tokens = {
        camera_names[modality]: token() for modality in camera_names.keys()
    }

    lidar_calib_token = token()

    cam_calib_tokens = {
        camera_names[modality]: token() for modality in camera_names.keys()
    }

    # sensors
    sensor = [
        {"token": lidar_sensor_token, "channel": lidar_channel, "modality": "lidar"},
        *[
            {
                "token": cam_sensor_tokens[camera_names[modality]],
                "channel": camera_names[modality],
                "modality": "camera",
            }
            for modality in camera_names.keys()
        ],
    ]

    # calibrated sensors
    calibrated_sensor.append(
        {
            "token": lidar_calib_token,
            "sensor_token": lidar_sensor_token,
            "translation": lidar_xyz,
            "rotation": lidar_quat,
            "camera_intrinsic": [],
        }
    )

    for cam_name in ds.cameras["color"].keys():
        channel_name = camera_names[cam_name]
        cam_xyz, cam_quat = parse_transform(
            ds.transforms_to_base[ds.frame_id_map[cam_name]]
        )
        cam_int = ds.cameras["color"][cam_name]["intrinsics"]
        calibrated_sensor.append(
            {
                "token": cam_calib_tokens[channel_name],
                "sensor_token": cam_sensor_tokens[channel_name],
                "translation": cam_xyz,
                "rotation": cam_quat,
                "camera_intrinsic": cam_int.tolist(),
            }
        )

    prev_sample_token = ""

    obj_id_instance_map = {}
    i = 0

    for idx, ts in tqdm(
        enumerate(ds.timestamps), total=len(ds.timestamps), desc="Processing samples"
    ):
        sample_files = ds.get_sample_files(ts)

        bboxes_3d_path = sample_files["bboxes_3d"][0]
        bboxes_3d = load_bboxes_3d(bboxes_3d_path) if bboxes_3d_path else []
        if len(bboxes_3d) == 0:
            continue

        sample_token = token()
        ego_token = token()

        # copy lidar
        src_lidar = os.path.join(sample_files["lidar"])
        pcd = load_pointcloud(src_lidar)
        dst_lidar = os.path.join(
            args.output,
            f"samples/{lidar_channel}",
            os.path.basename(src_lidar).replace(".pcd", ".bin"),
        )
        if pcd.shape[1] < 5:
            pcd = np.hstack(
                (pcd, np.zeros((pcd.shape[0], 5 - pcd.shape[1]), dtype=np.float32))
            )
        pcd.astype(np.float32).tofile(dst_lidar)

        # sample_data lidar
        sample_data.append(
            {
                "token": token(),
                "sample_token": sample_token,
                "ego_pose_token": ego_token,
                "calibrated_sensor_token": lidar_calib_token,
                "timestamp": ts,
                "filename": f"samples/{lidar_channel}/{os.path.basename(src_lidar).replace('.pcd', '.bin')}",
                "fileformat": "bin",
                "is_key_frame": True,
                "width": 0,
                "height": 0,
                "prev": "",
                "next": "",
            }
        )

        # copy images
        for modality in ds.cameras["color"].keys():
            cam_name = camera_names[modality]
            src_img = sample_files["images"][modality]
            dst_img = os.path.join(
                args.output, f"samples/{cam_name}", os.path.basename(src_img)
            )
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

            # sample_data camera
            if os.path.exists(src_img):
                sample_data.append(
                    {
                        "token": token(),
                        "sample_token": sample_token,
                        "ego_pose_token": ego_token,
                        "calibrated_sensor_token": cam_calib_tokens[cam_name],
                        "timestamp": sample_files["timestamps"][modality],
                        "filename": f"samples/{cam_name}/{os.path.basename(src_img)}",
                        "fileformat": "jpg",
                        "is_key_frame": True,
                        "width": ds.cameras["color"][modality]["cam_info"]["width"],
                        "height": ds.cameras["color"][modality]["cam_info"]["height"],
                        "prev": "",
                        "next": "",
                    }
                )

        current_scene = scene[i // num_samples_per_scene]
        current_scene_idx = i % num_samples_per_scene

        # sample
        samples.append(
            {
                "token": sample_token,
                "timestamp": ts,
                "prev": "",
                "next": "",
                "scene_token": current_scene["token"],
            }
        )
        if current_scene_idx > 0:
            samples[-1]["prev"] = samples[-2]["token"]
        if prev_sample_token and current_scene_idx > 0:
            samples[-2]["next"] = sample_token

        if current_scene["first_sample_token"] is None:
            current_scene["first_sample_token"] = sample_token
        current_scene["nbr_samples"] += 1
        current_scene["last_sample_token"] = sample_token

        # ego pose (fake static pose)
        ego_pose.append(
            {
                "token": ego_token,
                "timestamp": ts,
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
            }
        )

        # ------------------------
        # annotations
        # ------------------------
        for box in bboxes_3d:
            ann_token = token()
            if box["pose_id"] not in obj_id_instance_map:
                obj_id_instance_map[box["pose_id"]] = token()

                label = "human.pedestrian.adult"  # box["bbox_label"]

                if label not in [c["name"] for c in category]:
                    category.append(
                        {"token": token(), "name": label, "description": ""}
                    )

                inst_token = obj_id_instance_map[box["pose_id"]]
                instances.append(
                    {
                        "token": inst_token,
                        "first_annotation_token": ann_token,
                        "last_annotation_token": ann_token,
                        "category_token": next(
                            c["token"] for c in category if c["name"] == label
                        ),
                    }
                )

            inst_token = obj_id_instance_map[box["pose_id"]]

            annotations.append(
                {
                    "token": ann_token,
                    "sample_token": sample_token,
                    "instance_token": inst_token,
                    "attribute_tokens": [],
                    "translation": box["bbox_center"],
                    "size": box["bbox_size"],
                    "rotation": quat_to_nuscenes(box["bbox_rotation"]),
                    "num_lidar_pts": 10,
                    "num_radar_pts": 0,
                    "visibility_token": "1",
                    "prev": "",
                    "next": "",
                }
            )

            # update instance last_annotation_token
            for inst in instances:
                if inst["token"] == inst_token:
                    inst["last_annotation_token"] = ann_token
                    break

        prev_sample_token = sample_token
        i += 1

    log = [{"token": log_token, "location": "custom"}]

    # ------------------------
    # save
    # ------------------------
    def save(name, data):
        with open(os.path.join(args.output, VERSION, name), "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    save("sample.json", samples)
    save("sample_data.json", sample_data)
    save("ego_pose.json", ego_pose)
    save("category.json", category)
    save("attribute.json", attribute)
    save("calibrated_sensor.json", calibrated_sensor)
    save("sample_annotation.json", annotations)
    save("instance.json", instances)
    save("scene.json", scene)
    save("log.json", log)
    save("sensor.json", sensor)
    save("visibility.json", visibility)
    save("map.json", map_json)

    print("Conversion done.")

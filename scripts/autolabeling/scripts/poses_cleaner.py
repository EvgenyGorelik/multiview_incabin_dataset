import json, cv2, os, numpy as np
from tqdm import tqdm

from argparse import ArgumentParser
from ndf_dataset import NDFDataset
from ndf_dataset.io import load_poses, load_bboxes_3d
from ndf_dataset.utility import MHR_NAMES, MHR_CONNECTIONS
from ndf_dataset.transforms import project_points

BBOX_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
]

def process_segmentation_folder(folder, cleaned_poses_path):
    with open(cleaned_poses_path, 'r') as f:
        cleaned_poses = json.load(f)
    for i, fname in enumerate(os.listdir(folder)):
        if fname.endswith('.json'):
            json_path = os.path.join(folder, fname)
            with open(json_path, 'r') as f:
                data = json.load(f)
            valid_poses = cleaned_poses.get(str(i), [])
            valid_ids = [obj_id for j, obj_id in enumerate(data.keys()) if j in valid_poses]
            data = {j: pose for j, pose in data.items() if j in valid_ids}
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)


def get_box(keypoint_proj, img_size=(720, 1280)):
    if len(keypoint_proj) == 0:
        return [0, 0, 0, 0]
    x1 = int(np.min(keypoint_proj[:, 0]))
    y1 = int(np.min(keypoint_proj[:, 1]))
    x2 = int(np.max(keypoint_proj[:, 0]))
    y2 = int(np.max(keypoint_proj[:, 1]))
    # Ensure the box is within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_size[1], x2)
    y2 = min(img_size[0], y2)
    return [x1, y1, x2, y2]

def load_sample(ds: NDFDataset, idx, cam_name):
    sample_files = ds.get_sample_files(ds.timestamps[idx])  # make sure the sample files are loaded for projection
    img = cv2.imread(sample_files["images"][cam_name])
    poses = load_poses(sample_files["poses"][0])
    keypoint_projections = ds.project_poses_to_image(poses, cam_name, key='keypoints', return_valid=True)
    if len(sample_files["bboxes_3d"]) > 0:
        bboxes = load_bboxes_3d(sample_files["bboxes_3d"][0])
    else:
        bboxes = []
    bboxes_projected = []
    for bbox in bboxes:
        bbox_projected = project_points(bbox['bbox_corners'], ds.cameras['color'][cam_name]["intrinsics"], np.linalg.inv(ds.transforms_to_base[ds.frame_id_map[cam_name]]))
        bbox_projected = np.clip(bbox_projected, 0, img.shape[1]-1)
        bboxes_projected.append(bbox_projected)

    pose_boxes = [get_box(vp[0], img.shape[:2]) for vp in keypoint_projections]
    obj_ids = [pose['pose_id'] for pose in poses]
    return img, keypoint_projections, pose_boxes, bboxes_projected, obj_ids


if __name__ == "__main__":
    parser = ArgumentParser(description="NDF dataset pose cleaner tool")
    parser.add_argument("root", type=str, help="root directory of the ndf dataset")
    parser.add_argument("--file", type=str, default="cleaned_poses.json", help="output json file to save cleaned poses (default: cleaned_poses.json)")
    parser.add_argument("--clean", action="store_true", help="if set, will clean the pose json files based on the cleaned_poses.json file")
    parser.add_argument("--ids", nargs="+", type=str, default=[], help="list of pose indices to keep (1-based), if not set, will keep all poses")
    parser.epilog = "Tool for manually cleaning bounding boxes.\n\rHover over a box and press 'd' to delete it, 'u' to undo last deletion. Press 's' to switch camera. Press 'n' to save and go to next image, 'b' to save and go back to previous image. 'q' to quit without saving."
    args = parser.parse_args()

    if args.clean:
        process_segmentation_folder(os.path.join(args.root, "poses"), args.file)
        print(f"Poses folders cleaned based on {args.file}")
        exit()

    ds = NDFDataset(args.root)
    AVAILBLE_CAMERAS = list(ds.cameras['color'].keys())
    cam_idx = 0
    cam_name = AVAILBLE_CAMERAS[cam_idx]  # just use the first camera for cleaning

    if args.ids:
        results = {}
        for i in tqdm(range(len(ds)), desc="Processing samples"):
            img, keypoint_projections, pose_boxes, bboxes, obj_ids = load_sample(ds, i, cam_name)
            str_i = str(i)
            valid_poses = np.zeros(len(keypoint_projections), dtype=bool)
            for j, obj_id in enumerate(obj_ids):
                if obj_id in args.ids:
                    valid_poses[j] = True
            results[str_i] = [j for j in range(len(keypoint_projections)) if valid_poses[j]]
        with open(args.file, "w") as f:
            json.dump(results, f)
        print(f"Poses cleaned based on provided IDs and saved to {args.file}")
        exit()

    mouse = [0, 0]
    def move(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y

    cv2.namedWindow("viewer")
    cv2.setMouseCallback("viewer", move)

    undo_box = None

    results = {}  # dictionary to store valid pose indices for each sample
    if os.path.exists(args.file):
        with open(args.file, "r") as f:
            results = json.load(f)

    i = 0
    while i < len(ds):
        if i < 0:
            i += len(ds)
        img, keypoint_projections, pose_boxes, bboxes, obj_ids = load_sample(ds, i, cam_name)
        str_i = str(i)
        valid_poses = np.zeros(len(keypoint_projections), dtype=bool)
        valid_poses[results.get(str_i, np.arange(len(keypoint_projections)))] = True
        
        pose_boxes = [get_box(vp[0], img.shape[:2]) for vp in keypoint_projections]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, img.shape[1] / 1000.0)
        thickness = 2
        pad = 6
        x0, y0 = 10, 10
        text_size = cv2.getTextSize(str(i), font, scale, thickness)[0]
        rect_tl = (x0, y0)
        rect_br = (x0 + text_size[0] + pad * 2, y0 + text_size[1] + pad * 2)
        text_org = (x0 + pad, y0 + text_size[1] + pad)
        cv2.rectangle(img, rect_tl, rect_br, (0, 0, 0), -1)
        cv2.putText(img, str(i), text_org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        while True:
            vis = img.copy()
            for j, (uv_k, keypoint_validity) in enumerate(keypoint_projections):
                if not valid_poses[j]:
                    continue
                x1, y1, x2, y2 = pose_boxes[j]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID: {obj_ids[j]}", (x1, y1 - 10), font, scale * 0.8, (0, 255, 0), thickness, cv2.LINE_AA)

                for idx_a, idx_b in MHR_CONNECTIONS:
                    if keypoint_validity[idx_a] and keypoint_validity[idx_b]:
                        pt_a = tuple(uv_k[idx_a, :2].astype(int))
                        pt_b = tuple(uv_k[idx_b, :2].astype(int))
                        cv2.line(vis, pt_a, pt_b, (0, 255, 0), 2)

                if len(bboxes) > 0:
                    for idx_a, idx_b in BBOX_CONNECTIONS:
                        pt_a = tuple(bboxes[j][idx_a].astype(int))
                        pt_b = tuple(bboxes[j][idx_b].astype(int))
                        cv2.line(vis, pt_a, pt_b, (255, 0, 0), 2)


            cv2.imshow("viewer", vis)
            k = cv2.waitKey(10)

            if k == ord("s"):
                cam_idx = (cam_idx + 1) % len(AVAILBLE_CAMERAS)
                cam_name = AVAILBLE_CAMERAS[cam_idx]
                img, keypoint_projections, pose_boxes, bboxes, obj_ids = load_sample(ds, i, cam_name)

                valid_poses = np.zeros(len(keypoint_projections), dtype=bool)
                valid_poses[results.get(str_i, np.arange(len(keypoint_projections)))] = True

                cv2.rectangle(img, rect_tl, rect_br, (0, 0, 0), -1)
                cv2.putText(img, str_i, text_org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            if k == ord("d"):
                mx,my = mouse
                mx, my = mouse
                candidates = []
                for jj, (x1, y1, x2, y2) in enumerate(pose_boxes):
                    if not valid_poses[jj]:
                        continue
                    if x1 <= mx <= x2 and y1 <= my <= y2:
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        candidates.append((w * h, jj))
                if candidates:
                    _, sel = min(candidates, key=lambda x: x[0])
                    valid_poses[sel] = False
                    undo_box = sel

                results[str_i] = [j for j in range(len(keypoint_projections)) if valid_poses[j]]

            if k == ord("n"):
                results[str_i] = [j for j in range(len(keypoint_projections)) if valid_poses[j]]
                with open(args.file, "w") as f:
                    json.dump(results, f)
                break

            if k == ord("b"):
                results[str_i] = [j for j in range(len(keypoint_projections)) if valid_poses[j]]
                with open(args.file, "w") as f:
                    json.dump(results, f)
                i = i-2
                break

            if k == ord(" "):
                i += 10
                break

            if k == ord("u") and undo_box is not None:
                valid_poses[undo_box] = True
                undo_box = None
            
            if k == ord("q"):
                exit()
            
        i += 1
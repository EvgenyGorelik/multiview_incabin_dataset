import cv2
import numpy as np
import torch
import os
import json
from argparse import ArgumentParser

from tqdm import tqdm

from huggingface_hub import login

from sam_3d_body.func import setup_sam_3d_body

from ndf_dataset import NDFDataset


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


if __name__ == "__main__":
    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    # Set up SAM 3D Body estimator
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    # Load the dataset
    parser = ArgumentParser()
    parser.add_argument("root", help="Root directory of the dataset")
    args = parser.parse_args()
    ds = NDFDataset(args.root)

    for cam_name in ds.cameras['color'].keys():
        output_dir = os.path.join(ds.root, "body_3d", cam_name)
        os.makedirs(output_dir, exist_ok=True)

        cam_int = torch.eye(3)
        cam_int[0, 0] = ds.cameras['color'][cam_name]['cam_info']['fx']
        cam_int[1, 1] = ds.cameras['color'][cam_name]['cam_info']['fy']
        cam_int[0, 2] = ds.cameras['color'][cam_name]['cam_info']['cx']
        cam_int[1, 2] = ds.cameras['color'][cam_name]['cam_info']['cy']
        cam_int = cam_int.unsqueeze(0)

        for ts, img_path in tqdm(ds.cameras['color'][cam_name]['images'], desc=f"Processing {cam_name}"):
            img = cv2.imread(img_path)

            with open(ds.cameras['color'][cam_name]['bboxes_2d'][ts][0], 'r') as f:
                bboxes = json.load(f)
            for i, bbox in enumerate(bboxes):
                bbox_results = {'box': bbox['box']}
                estimator_results = estimator.process_one_image(img, bboxes=np.array([bbox_results['box']]), cam_int=cam_int)
                output_path = os.path.join(output_dir, f'{ts}_person_{i}.json')
                with open(output_path, 'w') as f:
                    json.dump(estimator_results[0], f, cls=NumpyEncoder)
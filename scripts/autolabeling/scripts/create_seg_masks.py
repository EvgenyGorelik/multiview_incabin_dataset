#!/usr/bin/env python3
"""
Data loader + visualizer for the "output" dataset layout.

Dependencies: numpy, pyyaml, opencv-python, open3d
Install (if needed):
    pip install numpy pyyaml opencv-python open3d
"""

import os
import json
import numpy as np

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("Transformers library not found. Please install it with 'pip install transformers==5.3.0' and try again.")
    exit(1)

from huggingface_hub import login
import torch
from PIL import Image
from argparse import ArgumentParser

from tqdm import tqdm

from ndf_dataset import NDFDataset


def store_results(results, output_path):
    """Store results in a JSON file."""
    out = []
    for mask, box, score in zip(results["masks"], results["boxes"], results["scores"]):
        mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
        mask_path = output_path.replace(".json", f"_mask_{len(out)}.png")
        mask_img.save(mask_path)
        out.append({
            "box": box.cpu().numpy().tolist(),
            "score": float(score.cpu().numpy()),
            "seg_fname": os.path.basename(mask_path)
        })
    with open(output_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for processing images")
    parser.add_argument("--text_prompt", type=str, default="person", help="Text prompt for segmentation")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for segmentation")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold for segmentation")
    args = parser.parse_args()

    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    root = args.root
    ds = NDFDataset(root)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    batch_size = args.batch_size
    for cam_folder, cam_data in ds.cameras['color'].items():
        output_dir = os.path.join(root, 'segmentations', cam_folder)
        os.makedirs(output_dir, exist_ok=True)

        images = ds.cameras['color'][cam_folder]["images"]
        # process in batches
        for i in tqdm(range(0, len(images), batch_size), desc=f"Processing {cam_folder} images"):
            batch = images[i:i+batch_size]
            ts_list, img_paths = zip(*batch)

            # Segment using text prompt for the batch
            inputs = processor(images=list(img_paths), text=[args.text_prompt for _ in img_paths], return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process results (one entry per image in batch)
            results_list = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.conf_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )

            # store each image's results
            for res, img_path in zip(results_list, img_paths):
                out_path = os.path.join(output_dir, os.path.basename(img_path).replace(".jpg", ".json"))
                store_results(res, out_path)
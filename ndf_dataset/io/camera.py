"""Camera calibration YAML and image readers."""

import os

import numpy as np
import yaml
from PIL import Image

def load_camera_info(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        obj = yaml.safe_load(fh)
    # camera_matrix 'data' length 9 (3x3)
    cam = {}
    data = obj.get("camera_matrix", {}).get("data", [])
    if len(data) >= 9:
        fx = float(data[0]); fy = float(data[4]); cx = float(data[2]); cy = float(data[5])
        cam["fx"] = fx; cam["fy"] = fy; cam["cx"] = cx; cam["cy"] = cy
        cam["width"] = int(obj.get("image_width", 0))
        cam["height"] = int(obj.get("image_height", 0))
    else:
        # fallback: projection matrix
        proj = obj.get("projection_matrix", {}).get("data", [])
        if len(proj) >= 12:
            fx = float(proj[0]); fy = float(proj[5]); cx = float(proj[2]); cy = float(proj[6])
            cam["fx"] = fx; cam["fy"] = fy; cam["cx"] = cx; cam["cy"] = cy
    return cam



def load_image(image_path, image_type="color"):
    """Load image as HxWx3 uint8 numpy array (RGB order)."""
    assert image_type in ["color", "depth", "segmentation"], "image_type must be 'color' or 'depth' or 'segmentation'"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        if image_type == "depth":
            img = img.convert("I")  # 32-bit float for depth
        elif image_type == "segmentation":
            img = img.convert("L")  # 8-bit grayscale for segmentation
        else:
            img = img.convert("RGB")  # Ensure color images are RGB
        img = np.array(img)
        return img
    return None

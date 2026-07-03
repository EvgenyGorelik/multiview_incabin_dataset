## Autolabeling Pipeline

### Installation

Install dependencies
```
pip install -r requirements.txt
```

For SAM3 Body 3D follow the installation on https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md

For using [SAM3](https://huggingface.co/facebook/sam3) and [SAM3 Body 3D](https://huggingface.co/facebook/sam-3d-body-dinov3) access on HuggingFace is required.

Once access is granted add the `HF_TOKEN` to the environment variables.

```
export HF_TOKEN=<Your huggingface access token>
```



### How To Use

Execute the following scripts:

```
python scripts/create_seg_masks.py <path/to/data>
python scripts/bbox_cleaner.py <path/to/data> --cam <camera> # for all cameras
python scripts/bbox_cleaner.py <path/to/data> --cam <camera> --clean # for all cameras
python scripts/track_seg.py <path/to/data> --iou_threshold 0.5
python scripts/create_body_poses.py <path/to/data>
python scripts/bev_aggregation.py <path/to/data> --eps 2.0
python scripts/poses_cleaner.py <path/to/data>
python scripts/extract_3d_bboxes.py <path/to/data>
```
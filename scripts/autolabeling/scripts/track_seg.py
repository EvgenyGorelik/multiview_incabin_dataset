import os
from argparse import ArgumentParser
import json
import torch

from ndf_dataset import NDFDataset
from ndf_dataset.io import load_bounding_boxes

from yolox.tracker.byte_tracker import BYTETracker, STrack


class BYTETrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5
        self.mot20 = False
        self.track_buffer = 30
        self.match_thresh = 0.8

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('root', type=str, help="Root directory of the NDF dataset")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for tracking")
    args = parser.parse_args()

    ds = NDFDataset(args.root)
    for cam_name in ds.cameras['color'].keys():
        print(f"Running tracking for camera: {cam_name}")
        tracker = BYTETracker(args=BYTETrackerArgs())

        timestamps = [s[0] for s in ds.cameras['color'][cam_name]['images']]
        img_info = (ds.cameras['color'][cam_name]['cam_info']['width'], ds.cameras['color'][cam_name]['cam_info']['height'])

        for timestamp in timestamps:
            bboxes_2d_path = ds.cameras['color'][cam_name]['bboxes_2d'][timestamp]
            bboxes_2d = load_bounding_boxes(bboxes_2d_path[0])
            if len(bboxes_2d) == 0:
                tracker.update(torch.empty((0, 5)), img_info, img_info)
                continue
            bboxes_2d_converted = torch.tensor([bbox['box'] for bbox in bboxes_2d])
            bboxes_2d_converted = torch.hstack([bboxes_2d_converted, torch.ones(len(bboxes_2d_converted), 1)])  # Add confidence scores (dummy values)
            tracker.update(bboxes_2d_converted, img_info, img_info)
            for i in range(len(tracker.tracked_stracks)):
                bboxes_2d[tracker.tracked_stracks[i].current_idx]['obj_id'] = tracker.tracked_stracks[i].track_id 
            with open(bboxes_2d_path[0], 'w') as f:
                json.dump(bboxes_2d, f)
    print('Total unique tracked IDs: ', STrack._count)
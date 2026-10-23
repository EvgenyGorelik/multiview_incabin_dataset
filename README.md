# Multi-View Incabin Dataset


Multi-View Multi-Modal dataset for incabin monitoring. 

## Abstract

We introduce a multi-view in-cabin monitoring dataset for public transportation with synchronized RGB and depth images from four inward-facing cameras and a rotating LiDAR covering the vehicle interior of a digitalized and partly automated German city bus. The dataset contains 9.136 synchronized samples and is accompanied by a calibration and pseudo-labeling pipeline that generates 3D human pose estimates and oriented 3D bounding boxes for occupants. We further provide a nuScenes-format conversion and benchmark representative multi-view 3D detection models (e.g., Lift-Splat-Shoot and BEVFusion), supporting comparative evaluation and small-scale training of multi-view in-cabin perception models.

## Methodology

![](assets/methodology.drawio.png)

## Setup

```
git clone https://github.com/EvgenyGorelik/multiview_incabin_dataset.git
cd multiview_incabin_dataset
pip install .
```

The dataset can be downloaded 
... on [Zenodo](
https://zenodo.org/records/20559664?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRhMDg0M2RiLTgyZTItNGJmNC1iNTk4LTM3OTk2N2U3N2Q0ZSIsImRhdGEiOnt9LCJyYW5kb20iOiI3MTZjMTU1ZDgwZDNmZmFiNmUzMDY5MmFkM2YzZGJjYSJ9.C1qemRRecTB5cWshlm-09ElAlYB2Uix-otwhTaBN97iEFQYq7ogPE8EsFBPU0VgkuXDfsNcEeUrr0Fa1R8Jofg)

OR 

... on [huggingface](https://huggingface.co/datasets/evgenygorelik96/multiview_incabin_dataset)


## How To Use

See [tutorial](tutorial.ipynb)

`NDFDataset(root)` indexes the sorted integer timestamps of JSON files in
`bboxes_3d/`. Each sample contains those boxes, the nearest image from each
camera, and the nearest LiDAR scan when available. Sensor frames retain their
own timestamps. Ties choose the earlier frame. Without 3D box files the dataset
has no samples; annotation-only datasets are supported.

```python
from ndf_dataset import NDFDataset

dataset = NDFDataset("/path/to/dataset")
sample = dataset[0]  # Sample dataclass
print(sample.timestamp)
for box in sample.bboxes_3d:
    print(box.obj_id, box.bbox_center, box.attr, box.action, box.velocity)
for name, frame in sample.cameras.get("color", {}).items():
    print(name, frame.timestamp, frame.image.shape if frame.image is not None else None)
if sample.lidar is not None:
    print(sample.lidar.timestamp, sample.lidar.points.shape)
```

`dataset.pcd_map` loads `map/dense_cleaned.ply` once during initialization
as an `(N, 3)` NumPy array of XYZ coordinates, consistent with LiDAR points.
It is `None` when the map file is absent. ASCII and binary PLY files are supported;
extra PLY properties such as colors and normals are ignored.

`dataset.cameras[type][name]` contains `Camera` calibration records.
`sample.cameras[type][name]` contains `CameraFrame` records with `camera`,
`timestamp`, `path`, and loaded `image`. Color and depth frames keep separate
sensor timestamps even when they share a camera name. Empty camera streams
have `None` for their timestamp, path, and image. Missing LiDAR is `None`.

`get_sample(timestamp)` selects the nearest annotation timestamp.
`get_sample_files(timestamp)` returns a `SampleFiles` record with
`bboxes_3d_path` and the same sensor frame structure, without loading data.
Images and points in those file records are `None`.

`BoundingBox3D` preserves box geometry (NumPy arrays), `obj_id`, `attr`, `action`,
and optional `velocity`. Quaternions use `(x, y, z, w)` order. `pose_id` is ignored.
The loader ignores 2D boxes, segmentations, body poses, poses, and states.
This replaces the former dictionary API; legacy pose/segmentation autolabeling
scripts that depend on those annotations require a separate annotation loader.

Run the sampling regression tests with:

```bash
python -m unittest discover -s tests -v
```



## Scenes

`dataset.scenes` maps recording names from `scenes.json` to `Scene` objects:

```python
for name, scene in dataset.scenes.items():
    print(name, len(scene), scene.timestamps)

scene = next(iter(dataset.scenes.values()))
if len(scene):
    sample = scene[0]  # Load this scene's first sample on demand.
    print(sample.scene_id, sample.timestamp)
    files = scene.get_sample_files(scene.timestamps[0])
```

Each scene exposes `cameras` (all assigned image files and calibration),
`lidar_scans`, `bboxes_3d` (timestamp-to-file mappings), `timestamps`, and the
original `metadata`. `scene.calibration`, `scene.frame_id_map`, `scene.modalities`,
and `scene.pcd_map` share the dataset's metadata and map without copying it.
Indexed access, iteration, `get_sample()`, and `get_sample_files()` load data
on demand and select sensors only within that scene. `sample.scene_id` and
`files.scene_id` identify the recording, including for dataset-level access.

Exported paths are matched against locally indexed camera type, camera name,
and timestamp, so moving the dataset or changing image/scan extensions does
not break the associations. Unavailable files remain in `scene.metadata` and
are excluded from its local file index. Scenes with images but no 3D boxes
still expose their images and have zero indexed samples.

Annotation membership uses exact LiDAR timestamps listed in `scenes.json`,
even if the local scan is missing. Other annotation timestamps are assigned
when they fall within exactly one scene's sensor timestamp range. Ambiguous
assignments raise `ValueError`. Unassigned samples remain accessible through
the dataset with `scene_id=None`. Without `scenes.json`, `dataset.scenes` is empty
and sampling continues as before.

## Package structure

`dataset.py` provides the `NDFDataset` entry point and delegates to focused modules:

```text
ndf_dataset/
├── dataset.py          # Dataset API and composition
├── indexing.py         # Discovery of cameras and timestamped files
├── sampling.py         # Sensor association and sample loading
├── scene.py            # Scene API and access to recording data
├── scene_indexing.py   # Build scene-local sensor and annotation indexes
├── timestamps.py       # Timestamp parsing and nearest-frame lookup
├── calibration.py      # Sensor frame transforms
├── models/             # Camera, LiDAR, box and sample records
├── io/                 # Camera, point cloud, annotation and metadata readers
├── geometry/           # Transforms, projections, boxes, skeletons and rays
└── viz.py              # Point cloud visualization
```

Existing imports from `ndf_dataset.io`, `ndf_dataset.models`,
`ndf_dataset.transforms`, `ndf_dataset.utility`, and `ndf_dataset.la` continue
to work. New code can import directly from the relevant modules, for example
`ndf_dataset.models.camera` or `ndf_dataset.geometry.projection`.


## Funding

This work is funded as a project in cooperation between TU Berlin and MAN Truck & Bus SE.

## Citation

```
@misc{gorelik2026multiviewincabinmonitoringpublic,
      title={Multi-View In-Cabin Monitoring System for Public Transport Vehicles}, 
      author={Evgeny Gorelik and Kenny Dean Karrow and Fikret Sivrikaya and Sahin Albayrak and Christian Baumann},
      year={2026},
      eprint={2606.11739},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.11739}, 
}
```

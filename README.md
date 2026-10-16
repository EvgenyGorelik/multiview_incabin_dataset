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

`NDFDataset(root)` uses LiDAR scan timestamps when scans are present. If the
`lidar/` directory is missing or contains no scans, it builds samples from the
sorted, unique image timestamps across all camera types and cameras. Each sample
uses the nearest image from each available camera, so asynchronous camera
streams can produce samples with repeated images from other views.

Camera-only samples work with `get_sample_files(timestamp)`,
`get_sample(timestamp)`, and indexed access (`dataset[i]`). Missing LiDAR data
and its timestamp are returned as `None`; missing camera types are returned as
empty dictionaries. Without LiDAR, poses, 3D boxes, and states are looked up at
the requested sample timestamp. Missing annotations remain `None` (or empty
lists for per-camera annotations).

Run the sampling regression tests with:

```bash
python -m unittest discover -s tests -v
```


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

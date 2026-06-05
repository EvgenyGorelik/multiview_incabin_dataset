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

## How To Use

See [tutorial](tutorial.ipynb)

## Funding

This work is funded as a project in cooperation between TU Berlin and MAN Truck & Bus SE.
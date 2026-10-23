"""Read frame transform chains from calibration JSON."""

import json

from ndf_dataset.geometry.transforms import compute_child_to_base, rpy_xyz_to_homogeneous


def load_transforms(path):
    """Read child-to-parent links and resolve each named frame to base_link."""
    with open(path) as stream:
        groups = json.load(stream)
    links = {}
    for joints in groups.values():
        for joint in joints:
            name = joint.get('joint_name', '')
            parent = joint.get('parent')
            parts = name.split('_to_')
            child = (parts[1] if len(parts) == 2
                     else name.replace(parent + '_to_', '') if parent else name)
            xyz = ([float(v) for v in str(joint['xyz']).split()]
                   if joint.get('xyz') is not None else [0., 0., 0.])
            rpy = ([float(v) for v in str(joint['rpy']).split()]
                   if joint.get('rpy') is not None else [0., 0., 0.])
            links[child] = (parent, rpy_xyz_to_homogeneous(rpy, xyz))
    to_base = {}
    for frame in groups:
        transform = compute_child_to_base(links, frame)
        if transform is not None:
            to_base[frame] = transform
    return links, to_base

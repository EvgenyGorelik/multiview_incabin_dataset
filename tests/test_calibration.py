"""Calibration behavior across chained, missing and cyclic frames."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ndf_dataset.calibration import Calibration
from ndf_dataset.geometry.transforms import rpy_xyz_to_homogeneous


class CalibrationTests(unittest.TestCase):
    def test_composes_chain_and_converts_between_frames(self):
        with TemporaryDirectory() as folder:
            groups = {
                'camera': [
                    dict(joint_name='base_link_to_mount', parent='base_link',
                         xyz='1 2 3', rpy='0 0 1.5707963267948966'),
                    dict(joint_name='mount_to_camera', parent='mount',
                         xyz='2 0 0', rpy='0 0 0'),
                ],
                'lidar': [dict(joint_name='base_link_to_lidar', parent='base_link',
                               xyz='0 0 1', rpy='0 0 0')],
            }
            (Path(folder) / 'target_transforms.json').write_text(json.dumps(groups))
            calibration = Calibration.from_root(folder)
        expected = rpy_xyz_to_homogeneous([0, 0, np.pi / 2], [1, 4, 3])
        np.testing.assert_allclose(calibration.get_transform('camera', 'base_link'), expected)
        np.testing.assert_allclose(calibration.get_transform('base_link', 'camera'), np.linalg.inv(expected))
        camera_to_lidar = calibration.get_transform('camera', 'lidar')
        np.testing.assert_allclose(camera_to_lidar[:3, 3], [1, 4, 2])
        np.testing.assert_allclose(
            calibration.get_transform('lidar', 'camera') @ camera_to_lidar,
            np.eye(4), atol=1e-12)
        self.assertIsNone(calibration.get_transform('unknown', 'camera'))

    def test_missing_calibration_and_base_identity(self):
        with TemporaryDirectory() as folder:
            calibration = Calibration.from_root(folder)
        self.assertEqual(calibration.transforms_to_base, {})
        np.testing.assert_array_equal(calibration.get_transform('base_link', 'base_link'), np.eye(4))
        self.assertIsNone(calibration.get_transform('camera', 'base_link'))

    def test_cyclic_and_disconnected_chains_are_unavailable(self):
        with TemporaryDirectory() as folder:
            groups = {
                'a': [dict(joint_name='b_to_a', parent='b')],
                'b': [dict(joint_name='a_to_b', parent='a')],
                'c': [dict(joint_name='missing_to_c', parent='missing')],
            }
            (Path(folder) / 'target_transforms.json').write_text(json.dumps(groups))
            calibration = Calibration.from_root(folder)
        self.assertEqual(calibration.transforms_to_base, {})
        self.assertIsNone(calibration.get_transform('a', 'b'))


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from ndf_dataset import NDFDataset


class DatasetSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def camera(self, kind, name, timestamps):
        folder = self.root / 'cameras' / kind / name
        (folder / 'images').mkdir(parents=True)
        (folder / 'camera_info.yaml').write_text(
            'image_width: 2\nimage_height: 2\n'
            'camera_matrix:\n  data: [1, 0, 1, 0, 1, 1, 0, 0, 1]\n'
        )
        for ts in timestamps:
            Image.new('RGB' if kind == 'color' else 'I;16', (2, 2), 1).save(
                folder / 'images' / f'{ts}.png'
            )
        return folder

    def test_rgb_only_without_lidar_or_annotations(self):
        self.camera('color', 'front', [100, 300])
        self.camera('color', 'back', [100, 200])
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100, 200, 300])
        self.assertEqual(len(ds), 3)
        self.assertIsNone(ds.find_nearest_lidar(200))
        files = ds.get_sample_files(200)
        self.assertIsNone(files['lidar'])
        self.assertIsNone(files['timestamps']['lidar'])
        self.assertEqual(files['timestamps']['back'], 200)
        for sample in ds:
            self.assertIsNone(sample['lidar'])
            self.assertEqual(sample['depth'], {})
            self.assertIsNone(sample['poses'])
            self.assertIsNone(sample['bboxes_3d'])
            self.assertIsNone(sample['states'])
            self.assertEqual(sample['bboxes_2d']['front'], [])
            self.assertEqual(sample['images']['front'].shape, (2, 2, 3))

    def test_empty_lidar_directory_uses_camera_timestamps(self):
        (self.root / 'lidar').mkdir()
        self.camera('color', 'front', [100])
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100])
        self.assertIsNone(ds[0]['lidar'])

    def test_color_and_depth_find_their_own_nearest_images(self):
        self.camera('color', 'front', [100, 300])
        self.camera('depth', 'depth_front', [110, 290])
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100, 110, 290, 300])
        sample = ds.get_sample(280)
        self.assertEqual(sample['timestamps']['front'], 300)
        self.assertEqual(sample['timestamps']['depth_front'], 290)
        self.assertEqual(sample['depth']['depth_front'].shape, (2, 2))

    def test_shared_camera_name_does_not_mix_color_and_depth_timestamps(self):
        self.camera('color', 'front', [200])
        self.camera('depth', 'front', [100, 290])
        ds = NDFDataset(str(self.root))
        sample = ds.get_sample_files(200)
        self.assertTrue(sample['depth']['front'].endswith('290.png'))
        self.assertEqual(sample['timestamps']['front'], 200)

    def test_depth_only_dataset(self):
        self.camera('depth', 'depth_front', [100, 200])
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100, 200])
        self.assertEqual(ds[0]['images'], {})
        self.assertIsNone(ds[0]['lidar'])

    def test_camera_timestamp_annotations_without_lidar(self):
        self.camera('color', 'front', [100])
        for folder in ('poses', 'bboxes_3d', 'states'):
            (self.root / folder).mkdir()
        (self.root / 'poses' / '100.json').write_text(json.dumps({'7': [[1, 2, 3]]}))
        (self.root / 'bboxes_3d' / '100.json').write_text('[]')
        (self.root / 'states' / '100.json').write_text('{}')
        ds = NDFDataset(str(self.root))
        files = ds.get_sample_files(100)
        for key in ('poses', 'bboxes_3d', 'states'):
            self.assertEqual(files[key], str(self.root / key / '100.json'))
        sample = ds[0]
        self.assertEqual(sample['poses'][0]['pose_id'], '7')
        self.assertEqual(sample['bboxes_3d'], [])

    def test_lidar_timeline_and_nearest_scan_are_preserved(self):
        self.camera('color', 'front', [90, 200, 310])
        (self.root / 'lidar').mkdir()
        for ts in (100, 300):
            np.array([[1, 2, 3, 4]], dtype=np.float32).tofile(
                self.root / 'lidar' / f'{ts}.bin'
            )
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100, 300])
        for query, expected in ((0, 100), (100, 100), (190, 100),
                                (200, 100), (250, 300), (400, 300)):
            with self.subTest(query=query):
                ts, path = ds.find_nearest_lidar(query)
                self.assertEqual(ts, expected)
                self.assertEqual(Path(path).name, f'{expected}.bin')
        np.testing.assert_array_equal(ds[0]['lidar'], [[1, 2, 3]])
        self.assertEqual(ds.get_sample_files(250)['timestamps']['lidar'], 300)

    def test_empty_dataset_or_cameras(self):
        for with_camera in (False, True):
            if with_camera:
                self.camera('color', 'front', [])
            ds = NDFDataset(str(self.root))
            self.assertEqual(len(ds), 0)
            self.assertIsNone(ds.get_sample_files(100))
            self.assertIsNone(ds.get_sample(100))


if __name__ == '__main__':
    unittest.main()

"""Scene associations, relocated exports and recording boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from ndf_dataset import NDFDataset, Scene


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def camera(self, kind, name, timestamps):
        folder = self.root / 'cameras' / kind / name
        (folder / 'images').mkdir(parents=True)
        (folder / 'camera_info.yaml').write_text(
            'image_width: 2\nimage_height: 2\ncamera_matrix:\n  data: [1, 0, 1, 0, 1, 1, 0, 0, 1]\n')
        for timestamp in timestamps:
            Image.new('RGB' if kind == 'color' else 'I;16', (2, 2), 1).save(
                folder / 'images' / f'{timestamp}.png')

    def boxes(self, timestamps):
        folder = self.root / 'bboxes_3d'
        folder.mkdir()
        for timestamp in timestamps:
            (folder / f'{timestamp}.json').write_text('[]')

    def scenes(self, entries):
        (self.root / 'scenes.json').write_text(json.dumps(entries))

    def test_relocated_paths_and_scene_boundaries(self):
        self.camera('color', 'front', [100, 200, 201, 300])
        self.camera('depth', 'front', [110, 210])
        self.boxes([100, 200, 201, 300, 999])
        (self.root / 'lidar').mkdir()
        for timestamp in (100, 201):
            np.array([[timestamp, 2, 3, 4]], dtype=np.float32).tofile(
                self.root / 'lidar' / f'{timestamp}.bin')
        self.scenes({
            'first': {'cameras': {'front': [
                '/old/root/cameras/color/front/images/100.jpg',
                '/old/root/cameras/depth/front/images/110.jpg',
                '/old/root/cameras/color/front/images/200.jpg']},
                'lidar': ['/old/root/lidar/100.pcd']},
            'second': {'cameras': {'front': [
                '/old/root/cameras/color/front/images/201.jpg',
                '/old/root/cameras/depth/front/images/210.jpg',
                '/old/root/cameras/color/front/images/300.jpg']},
                'lidar': ['/old/root/lidar/201.pcd']},
        })
        ds = NDFDataset(str(self.root))
        first, second = ds.scenes['first'], ds.scenes['second']
        self.assertIsInstance(first, Scene)
        self.assertEqual(first.timestamps, [100, 200])
        self.assertEqual(second.timestamps, [201, 300])
        self.assertEqual(list(first.lidar_scans), [100])
        self.assertEqual([t for t, _ in first.cameras['depth']['front'].images], [110])
        # Even at a scene boundary, closer frames from another recording are excluded.
        files = ds.get_sample_files(200)
        self.assertEqual(files.scene_id, 'first')
        self.assertEqual(files.lidar.timestamp, 100)
        self.assertEqual(files.cameras['depth']['front'].timestamp, 110)
        self.assertTrue(files.cameras['color']['front'].path.startswith(str(self.root)))
        self.assertEqual(first.get_sample_files(999).timestamp, 200)
        sample = second[0]
        self.assertEqual(sample.scene_id, 'second')
        np.testing.assert_array_equal(sample.lidar.points, [[201, 2, 3]])
        self.assertEqual([sample.timestamp for sample in first], [100, 200])
        self.assertIsNone(ds.get_sample_files(999).scene_id)
        self.assertIs(first.calibration, ds.calibration)
        self.assertIs(first.pcd_map, ds.pcd_map)
        self.assertIs(first.frame_id_map, ds.frame_id_map)

    def test_camera_only_and_annotation_free_scenes(self):
        self.camera('color', 'front', [100, 200, 300])
        self.boxes([150])
        self.scenes({
            'camera-only': {'cameras': {'front': ['100.png', '200.png']}},
            'no-boxes': {'cameras': {'front': ['300.png']}},
            'empty': {},
        })
        ds = NDFDataset(str(self.root))
        scene = ds.scenes['camera-only']
        self.assertEqual(scene.timestamps, [150])
        self.assertIsNone(scene[0].lidar)
        self.assertEqual(scene[0].cameras['color']['front'].timestamp, 100)
        self.assertEqual(len(ds.scenes['no-boxes']), 0)
        self.assertEqual(len(ds.scenes['no-boxes'].cameras['color']['front'].images), 1)
        self.assertEqual(len(ds.scenes['empty']), 0)
        self.assertIsNone(ds.scenes['empty'].get_sample(0))
        with self.assertRaises(IndexError):
            ds.scenes['empty'][0]

    def test_lidar_references_assign_annotations_without_local_scans(self):
        self.boxes([100])
        self.scenes({'recording': {'lidar': ['/old/lidar/100.bin']}})
        scene = NDFDataset(str(self.root)).scenes['recording']
        self.assertEqual(scene.timestamps, [100])
        self.assertIsNone(scene[0].lidar)

    def test_missing_scene_metadata(self):
        self.boxes([100])
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.scenes, {})
        self.assertIsNone(ds[0].scene_id)

    def test_ambiguous_scene_membership_is_rejected(self):
        self.boxes([150])
        self.scenes({
            'a': {'cameras': {'front': ['100.png', '200.png']}},
            'b': {'cameras': {'front': ['140.png', '160.png']}},
        })
        with self.assertRaisesRegex(ValueError, 'matches multiple scenes'):
            NDFDataset(str(self.root))

    def test_duplicate_lidar_membership_is_rejected(self):
        self.scenes({'a': {'lidar': ['100.bin']}, 'b': {'lidar': ['100.bin']}})
        with self.assertRaisesRegex(ValueError, 'belongs to multiple scenes'):
            NDFDataset(str(self.root))

    def test_invalid_scene_metadata(self):
        self.scenes({'bad': {'cameras': {'front': '100.png'}}})
        with self.assertRaisesRegex(ValueError, 'lists of paths'):
            NDFDataset(str(self.root))


if __name__ == '__main__':
    unittest.main()

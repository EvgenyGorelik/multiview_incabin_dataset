import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from ndf_dataset import NDFDataset, BoundingBox3D, Camera, Sample, SampleFiles


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

    def boxes(self, timestamps, entries=None):
        folder = self.root / 'bboxes_3d'
        folder.mkdir(exist_ok=True)
        for ts in timestamps:
            (folder / f'{ts}.json').write_text(json.dumps(entries or []))

    def test_missing_map(self):
        self.assertIsNone(NDFDataset(str(self.root)).pcd_map)
        (self.root / 'map').mkdir()
        self.assertIsNone(NDFDataset(str(self.root)).pcd_map)

    def test_ply_map_formats_and_extra_properties(self):
        import struct

        folder = self.root / 'map'
        folder.mkdir()
        path = folder / 'dense_cleaned.ply'
        # Use reordered fields to ensure XYZ are selected by property name.
        for encoding in ('ascii', 'binary_little_endian', 'binary_big_endian'):
            with self.subTest(encoding=encoding):
                header = (
                    f'ply\nformat {encoding} 1.0\ncomment test map\n'
                    'element vertex 2\nproperty uchar red\n'
                    'property double z\nproperty double x\nproperty double y\n'
                    'end_header\n'
                ).encode('ascii')
                if encoding == 'ascii':
                    body = b'255 3 1 2\n0 6 4 5\n'
                else:
                    endian = '<' if encoding == 'binary_little_endian' else '>'
                    body = (struct.pack(endian + 'Bddd', 255, 3, 1, 2)
                            + struct.pack(endian + 'Bddd', 0, 6, 4, 5))
                path.write_bytes(header + body)
                ds = NDFDataset(str(self.root))
                np.testing.assert_array_equal(ds.pcd_map, [[1, 2, 3], [4, 5, 6]])
                self.assertEqual(len(ds), 0)

    def test_empty_ply_map(self):
        folder = self.root / 'map'
        folder.mkdir()
        (folder / 'dense_cleaned.ply').write_text(
            'ply\nformat ascii 1.0\nelement vertex 0\n'
            'property float x\nproperty float y\nproperty float z\nend_header\n')
        self.assertEqual(NDFDataset(str(self.root)).pcd_map.shape, (0, 3))

    def test_only_box_timestamps_define_samples(self):
        self.camera('color', 'front', [90, 210, 500])
        self.boxes([100, 200])
        (self.root / 'timestamps.json').write_text('[1, 2, 3]')
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [100, 200])
        self.assertEqual([sample.timestamp for sample in ds], [100, 200])
        self.assertIsInstance(ds[0], Sample)
        self.assertIsInstance(ds.cameras['color']['front'], Camera)
        self.assertEqual(ds[0].cameras['color']['front'].image.shape, (2, 2, 3))
        self.assertIsNone(ds[0].lidar)
        self.assertEqual(ds[0].bboxes_3d, [])

    def test_no_annotations_means_no_samples(self):
        self.camera('color', 'front', [100])
        ds = NDFDataset(str(self.root))
        self.assertEqual(len(ds), 0)
        self.assertIsNone(ds.get_sample_files(100))
        self.assertIsNone(ds.get_sample(100))
        with self.assertRaises(IndexError):
            ds[0]

    def test_same_camera_name_keeps_modality_timestamps(self):
        self.camera('color', 'front', [200])
        self.camera('depth', 'front', [100, 290])
        self.boxes([200])
        ds = NDFDataset(str(self.root))
        files = ds.get_sample_files(200)
        self.assertIsInstance(files, SampleFiles)
        self.assertEqual(files.cameras['color']['front'].timestamp, 200)
        self.assertEqual(files.cameras['depth']['front'].timestamp, 290)
        self.assertIsNone(files.cameras['color']['front'].image)
        self.assertEqual(ds[0].cameras['depth']['front'].image.shape, (2, 2))

    def test_box_metadata_uses_obj_id(self):
        entry = dict(bbox_label='human', bbox_center=[1, 2, 3],
                     bbox_size=[.5, .3, 1.7], bbox_rotation=[0, 0, 0, 1],
                     bbox_corners=[[0, 0, 0]] * 8, obj_id=42, pose_id='wrong',
                     attr='enter vehicle', action='none', velocity=[.1, .2, .3])
        self.boxes([1779354652243814114], [entry])
        box = NDFDataset(str(self.root))[0].bboxes_3d[0]
        self.assertIsInstance(box, BoundingBox3D)
        self.assertEqual(box.obj_id, 42)
        self.assertFalse(hasattr(box, 'pose_id'))
        self.assertEqual(box.attr, 'enter vehicle')
        self.assertEqual(box.action, 'none')
        np.testing.assert_array_equal(box.velocity, [.1, .2, .3])
        self.assertEqual(box.bbox_corners.shape, (8, 3))
        self.assertEqual(box.bbox_label, 'human')

    def test_optional_box_metadata_does_not_fall_back_to_pose_id(self):
        entry = dict(bbox_label='human', bbox_center=[1, 2, 3],
                     bbox_size=[1, 1, 1], bbox_rotation=[0, 0, 0, 1],
                     bbox_corners=[[0, 0, 0]] * 8, pose_id='legacy')
        self.boxes([100], [entry])
        box = NDFDataset(str(self.root))[0].bboxes_3d[0]
        for name in ('obj_id', 'attr', 'action', 'velocity'):
            self.assertIsNone(getattr(box, name))

    def test_camera_calibration_and_lidar_projection(self):
        self.camera('color', 'front', [100])
        self.boxes([100])
        (self.root / 'frame_ids.json').write_text(
            json.dumps({'front': 'front_optical', 'lidar': 'lidar_frame'}))
        transforms = {
            frame: [dict(joint_name='base_link_to_' + frame,
                         parent='base_link', xyz='0 0 0', rpy='0 0 0')]
            for frame in ('front_optical', 'lidar_frame')}
        (self.root / 'target_transforms.json').write_text(json.dumps(transforms))
        ds = NDFDataset(str(self.root))
        camera = ds.cameras['color']['front']
        self.assertEqual(camera.frame_id, 'front_optical')
        np.testing.assert_array_equal(camera.intrinsics, [[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        uv, valid = ds.project_pointcloud_to_image(
            np.array([[0., 0., 1.]]), 'front', return_valid=True)
        np.testing.assert_allclose(uv, [[1, 1, 1]])
        self.assertTrue(valid[0])

    def test_lidar_timeline_does_not_replace_annotations(self):
        self.boxes([190, 250])
        (self.root / 'lidar').mkdir()
        for ts in (100, 300):
            np.array([[ts, 2, 3, 4]], dtype=np.float32).tofile(
                self.root / 'lidar' / f'{ts}.bin')
        ds = NDFDataset(str(self.root))
        self.assertEqual(ds.timestamps, [190, 250])
        for query, expected in ((0, 100), (100, 100), (190, 100),
                                (200, 100), (250, 300), (400, 300)):
            with self.subTest(query=query):
                ts, path = ds.find_nearest_lidar(query)
                self.assertEqual(ts, expected)
                self.assertEqual(Path(path).name, f'{expected}.bin')
        sample = ds[1]
        self.assertEqual(sample.timestamp, 250)
        self.assertEqual(sample.lidar.timestamp, 300)
        np.testing.assert_array_equal(sample.lidar.points, [[300, 2, 3]])
        self.assertEqual(ds.get_sample_files(250).bboxes_3d_path,
                         str(self.root / 'bboxes_3d' / '250.json'))
        self.assertIsNone(ds.get_sample_files(250).lidar.points)

    def test_empty_lidar_and_camera_stream(self):
        (self.root / 'lidar').mkdir()
        self.camera('color', 'front', [])
        self.boxes([100])
        sample = NDFDataset(str(self.root))[0]
        self.assertIsNone(sample.lidar)
        frame = sample.cameras['color']['front']
        self.assertIsNone(frame.timestamp)
        self.assertIsNone(frame.path)
        self.assertIsNone(frame.image)

    def test_annotation_only_dataset(self):
        self.boxes([100])
        sample = NDFDataset(str(self.root))[0]
        self.assertEqual(sample.cameras, {})
        self.assertEqual(sample.bboxes_3d, [])
        self.assertIsNone(sample.lidar)

    def test_nearest_annotation_and_camera_ties_choose_earlier(self):
        self.camera('color', 'front', [90, 110])
        self.boxes([100, 200])
        ds = NDFDataset(str(self.root))
        sample = ds.get_sample(150)
        self.assertEqual(sample.timestamp, 100)
        self.assertEqual(sample.cameras['color']['front'].timestamp, 90)
        self.assertEqual(ds.get_sample(999).timestamp, 200)
        self.assertEqual(ds.get_sample(0).timestamp, 100)

    def test_ignored_annotations_are_never_read(self):
        self.boxes([100])
        for folder in ('poses', 'body_3d', 'segmentations', 'bboxes', 'states'):
            target = self.root / folder
            target.mkdir()
            (target / '100.json').write_text('invalid json')
        ds = NDFDataset(str(self.root))
        sample = ds[0]
        self.assertEqual(sample.bboxes_3d, [])
        for name in ('poses', 'body_3d', 'segmentations', 'bboxes_2d', 'states'):
            self.assertFalse(hasattr(sample, name))


if __name__ == '__main__':
    unittest.main()

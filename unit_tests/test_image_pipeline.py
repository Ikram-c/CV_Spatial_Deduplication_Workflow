"""Regression tests for CPU image-pair inputs without requiring image packages."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_pipeline import (
    count_uav_detections, load_coco_annotations, read_matched_image_pairs,
)


class ImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'frames'
        self.images.mkdir()
        self.log = self.root / 'odm.txt'
        self.coco = self.root / 'annotations.json'
        self.log.write_text(
            'Matching DJI_0001.JPG and DJI_0002.JPG Matches: 70 Success: True\n'
            'Matching DJI_0002.JPG and DJI_0003.JPG Matches: 65 Success: True\n'
            'Matching DJI_0003.JPG and DJI_0001.JPG Matches: FAILED\n'
            'Matching DJI_0001.JPG and DJI_0002.JPG Matches: 60 Success: True\n',
            encoding='utf-8',
        )
        self.coco.write_text(json.dumps({
            'images': [
                {'id': 1, 'file_name': 'DJI_0001.JPG'},
                {'id': 2, 'file_name': 'DJI_0002.JPG'},
                {'id': 3, 'file_name': 'DJI_0003.JPG'},
            ],
            'annotations': [
                {'id': 101, 'image_id': 1, 'category_id': 1, 'bbox': [0.25, 0.3, 5.2, 5.2]},
                {'id': 102, 'image_id': 2, 'category_id': 1, 'bbox': [4.9, 0.6, 5.2, 5.2]},
                {'id': 103, 'image_id': 3, 'category_id': 1, 'bbox': [9.5, 0.6, 5.2, 5.2]},
                {'id': 104, 'image_id': 3, 'category_id': 1, 'bbox': [30, 30, 4, 4]},
            ],
            'categories': [{'id': 1, 'name': 'litter'}]
        }), encoding='utf-8')

    def test_pair_parser_keeps_only_unique_successes(self):
        self.assertEqual(read_matched_image_pairs(self.log), [
            ('DJI_0001.JPG', 'DJI_0002.JPG'),
            ('DJI_0002.JPG', 'DJI_0003.JPG'),
        ])

    def test_load_coco_keeps_fractional_box_geometry(self):
        names, boxes, categories = load_coco_annotations(self.coco)
        self.assertEqual(names['DJI_0001.JPG'], 1)
        self.assertAlmostEqual(boxes[1][0][2], 5.45)
        self.assertEqual(categories[104], 1)

    def test_complete_image_mode_with_mocked_registration(self):
        # Zero model calls: registration is mocked to an identity transform.
        # All boxes have already been expressed in the same global coordinates.
        with patch('image_pipeline.find_cpu_homography', return_value='identity'), \
             patch('image_pipeline.transform_boxes_float', side_effect=lambda boxes, _: boxes):
            result = count_uav_detections(self.log, self.images, self.coco)
        self.assertEqual(result['raw_total'], 4)
        self.assertEqual(result['unique_total'], 2)
        self.assertEqual(result['duplicates_removed'], 2)
        self.assertEqual(result['overlap_edges'], [[101, 102], [102, 103]])
        self.assertEqual(result['aligned_pairs'], 2)
        self.assertEqual(result['skipped_pairs'], [])

    def test_unalignable_pairs_keep_singleton_counts(self):
        with patch('image_pipeline.find_cpu_homography', return_value=None):
            result = count_uav_detections(self.log, self.images, self.coco)
        self.assertEqual(result['unique_total'], 4)
        self.assertEqual(result['aligned_pairs'], 0)
        self.assertEqual(len(result['skipped_pairs']), 2)

    def test_negative_coco_width_rejected(self):
        data = json.loads(self.coco.read_text(encoding='utf-8'))
        data['annotations'][0]['bbox'][2] = -1
        self.coco.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'invalid bbox'):
            load_coco_annotations(self.coco)


if __name__ == '__main__':
    unittest.main()

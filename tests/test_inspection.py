"""Local inspection uses the same pixel coordinates as the object editor."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork.inspection import changed_bounds, inspect_edit


class InspectionTests(unittest.TestCase):
    def test_selection_locks_detail_but_reports_changes_across_the_whole_image(self):
        old = np.full((18, 25, 3), [4, 30, 130], np.uint8)
        new = old.copy()
        new[1:3, 2:4] = [200, 5, 70]
        new[15, 24] = 0
        self.assertEqual(changed_bounds(old, new), ([2, 1, 25, 16], 5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '试画' / '眼睛.png'
            view = inspect_edit(old, new, bbox=[-7, -2, 5, 7], padding=2, path=path)
            info = view.metadata
            self.assertEqual(info['canvas_size'], [25, 18])
            self.assertEqual(info['selection_bbox'], [0, 0, 5, 7])
            self.assertEqual(info['changed_bbox'], [2, 1, 25, 16])
            self.assertEqual(info['crop_bbox'], [0, 0, 7, 9])
            self.assertEqual(info['changed_pixels'], 5)
            self.assertEqual(info['changed_outside_selection'], 1)
            self.assertEqual(info['path'], str(path.resolve()))
            decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            np.testing.assert_array_equal(view.image, decoded)
            for rgb in ([4, 30, 130], [200, 5, 70]):
                self.assertTrue((view.image == rgb).all(axis=2).any())

    def test_irregular_mask_counts_the_hole(self):
        old = np.zeros((7, 9, 3), np.uint8)
        new = old.copy()
        new[3, 3:5, 1] = 1
        mask = np.zeros((7, 9), np.uint8)
        mask[1:6, 2:7] = 128
        mask[3, 4] = 0
        info = inspect_edit(old, new, mask=mask, padding=0).metadata
        self.assertEqual(info['crop_bbox'], [2, 1, 7, 6])
        self.assertEqual(info['changed_bbox'], [3, 3, 5, 4])
        self.assertEqual(info['changed_outside_selection'], 1)
        info = inspect_edit(old, new, mask=np.zeros_like(mask), padding=0).metadata
        self.assertIsNone(info['selection_bbox'])
        self.assertEqual(info['crop_bbox'], info['changed_bbox'])
        self.assertEqual(info['changed_outside_selection'], 2)

    def test_no_change_and_one_pixel_at_bottom_right(self):
        old = np.zeros((3, 11, 4), np.uint8)
        view = inspect_edit(old, old, padding=0)
        self.assertEqual(changed_bounds(old, old), (None, 0))
        self.assertIsNone(view.metadata['changed_bbox'])
        self.assertEqual(view.metadata['changed_pixels'], 0)
        self.assertEqual(view.metadata['crop_bbox'], [0, 0, 11, 3])
        self.assertTrue((view.image == [235, 235, 235]).all(axis=2).any())
        new = old.copy()
        new[-1, -1, 3] = 40
        info = inspect_edit(old, new, padding=0).metadata
        self.assertEqual(info['crop_bbox'], [10, 2, 11, 3])
        self.assertEqual(info['changed_pixels'], 1)

    def test_invalid_selection(self):
        old = np.zeros((4, 3, 3), np.uint8)
        for kwargs in (dict(bbox=[3, 0, 5, 7]), dict(bbox=[0, 0, 1.2, 3]),
                       dict(bbox=[0, 0, 1, 1], mask=np.ones((4, 3), bool)),
                       dict(mask=np.ones((4, 2))), dict(padding=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inspect_edit(old, old, **kwargs)
        with self.assertRaises(ValueError):
            inspect_edit(old, old[:2])


if __name__ == '__main__':
    unittest.main()

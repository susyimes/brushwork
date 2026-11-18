"""Pixel-level regressions for non-destructive editing by object ID."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import ObjectEditor


class ObjectEditorTests(unittest.TestCase):
    def test_move_hide_and_replace_restore_the_actual_underpainting(self):
        base = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        pupil = np.zeros((4, 4, 4), np.uint8)
        pupil[1:3, 1:3] = [10, 20, 30, 255]
        lid = np.full((1, 3, 3), [106, 115, 124], np.uint8)
        editor = ObjectEditor(base.copy())
        editor.add('pupil', pupil, xy=(0, 1))
        editor.add('lid', lid, xy=(2, 2))
        self.assertEqual(editor.bbox('pupil'), (1, 2, 3, 4))

        editor.move('pupil', 2, 0)
        expected = base.copy()
        expected[2:4, 3:5] = pupil[1:3, 1:3, :3]
        expected[2:3, 2:5] = lid
        np.testing.assert_array_equal(editor.render(), expected)
        self.assertEqual(editor.bbox('pupil'), (3, 2, 5, 4))

        editor.set_visible('lid', False)
        expected[2, 2] = base[2, 2]
        expected[2, 3:5] = pupil[1, 1:3, :3]
        np.testing.assert_array_equal(editor.render(), expected)
        self.assertIsNone(editor.bbox('lid'))

        replacement = np.full((1, 2, 3), [77, 12, 65], np.uint8)
        editor.replace('pupil', replacement, xy=(1, 6))
        expected = base.copy()
        expected[6, 1:3] = replacement[0]
        np.testing.assert_array_equal(editor.render(), expected)
        self.assertEqual(editor.bbox('pupil'), (1, 6, 3, 7))

        editor.replace('pupil', replacement[:, :1])
        self.assertEqual(editor.bbox('pupil'), (1, 6, 2, 7))
        editor.set_visible('pupil', False)
        np.testing.assert_array_equal(editor.render(), base)

    def test_fixed_selection_and_canvas_bounds_survive_a_move(self):
        base = np.zeros((3, 5, 3), np.uint8)
        clip = np.array([[255, 0, 128], [0, 255, 0]], np.uint8)
        tile = np.full((4, 3, 3), [255, 100, 0], np.uint8)
        editor = ObjectEditor(base)
        editor.add('iris', tile, xy=(1, -1), clip=clip, clip_xy=(1, 1))
        expected = base.copy()
        expected[1, 1] = [255, 100, 0]
        expected[1, 3] = [128, 50, 0]
        expected[2, 2] = [255, 100, 0]
        np.testing.assert_array_equal(editor.render(), expected)
        self.assertEqual(editor.bbox('iris'), (1, 1, 4, 3))

        editor.move('iris', 2, 0)
        expected[:] = 0
        expected[1, 3] = [128, 50, 0]
        np.testing.assert_array_equal(editor.render(), expected)
        self.assertEqual(editor.bbox('iris'), (3, 1, 4, 2))
        editor.move('iris', 3, 0)
        self.assertIsNone(editor.bbox('iris'))
        np.testing.assert_array_equal(editor.render(), base)

    def test_rgba_over_rgba_uses_straight_alpha(self):
        base = np.array([[[0, 0, 200, 128], [0, 150, 0, 0]]], np.uint8)
        layer = np.full((1, 2, 4), [200, 0, 0, 128], np.uint8)
        editor = ObjectEditor(base)
        editor.add('wash', layer)
        expected = np.array([[[134, 0, 66, 192], [200, 0, 0, 128]]], np.uint8)
        np.testing.assert_array_equal(editor.render(), expected)
        editor.set_visible('wash', False)
        np.testing.assert_array_equal(editor.render(), base)

    def test_unicode_png_and_editable_archive_round_trip(self):
        base = np.full((3, 6, 3), [200, 150, 100], np.uint8)
        rgba = np.array([[[30, 60, 90, 255], [180, 30, 60, 255]]], np.uint8)
        clip = np.array([[255, 255, 0]], np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, eye_path = root / '底.png', root / '右眼.png'
            base_path.write_bytes(cv2.imencode('.png', base[..., ::-1])[1].tobytes())
            eye_path.write_bytes(cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))[1].tobytes())
            editor = ObjectEditor(base_path)
            editor.add('右眼', eye_path, xy=(2, 1), clip=clip, clip_xy=(2, 1))
            editor.add('眼睑', rgba, xy=(2, 1))
            editor.set_visible('眼睑', False)
            original = base.copy()
            original[1, 2:4] = rgba[0, :, :3]
            np.testing.assert_array_equal(editor.render(), original)
            archive = root / '可编辑画稿.bwo'
            editor.save(archive)
            self.assertTrue(archive.is_file())
            base_path.unlink()
            eye_path.unlink()
            restored = ObjectEditor.load(archive)

        self.assertEqual(tuple(restored.object_ids), ('右眼', '眼睑'))
        self.assertIsNone(restored.bbox('眼睑'))
        np.testing.assert_array_equal(restored.render(), original)
        restored.move('右眼', 1, 0)
        expected = base.copy()
        expected[1, 3] = rgba[0, 0, :3]
        np.testing.assert_array_equal(restored.render(), expected)
        np.testing.assert_array_equal(editor.render(), original)

    def test_add_mask_with_grayscale_and_alpha_clip_files(self):
        mask = np.array([[128, 255, 255, 0]], np.uint8)
        clip = np.array([[255, 0, 128, 255]], np.uint8)
        rgba = np.full((1, 4, 4), 255, np.uint8)
        rgba[..., 3] = clip
        with tempfile.TemporaryDirectory() as directory:
            for name, pixels in [('灰度', clip), ('透明度', rgba)]:
                with self.subTest(name=name):
                    path = Path(directory) / (name + '.png')
                    path.write_bytes(cv2.imencode('.png', pixels)[1].tobytes())
                    base = np.full((3, 6, 3), 24, np.uint8)
                    editor = ObjectEditor(base)
                    editor.add_mask('ink', mask, '#1878D8', xy=(1, 1), clip=path, clip_xy=(1, 1))
                    expected = base.copy()
                    expected[1, 1] = expected[1, 3] = [24, 72, 120]
                    np.testing.assert_array_equal(editor.render(), expected)
                    self.assertEqual(editor.bbox('ink'), (1, 1, 4, 2))

    def test_duplicate_id_leaves_state_intact(self):
        base = np.zeros((2, 3, 3), np.uint8)
        editor = ObjectEditor(base)
        editor.add('eye', np.full((1, 1, 3), 180, np.uint8))
        before = editor.render().copy()
        with self.assertRaises((ValueError, KeyError)):
            editor.add('eye', base)
        self.assertEqual(tuple(editor.object_ids), ('eye',))
        np.testing.assert_array_equal(editor.render(), before)

    def test_object_mask_is_canvas_sized_and_preserves_covered_pixels(self):
        base = np.zeros((4, 6, 3), np.uint8)
        tile = np.full((2, 3, 4), 255, np.uint8)
        tile[1, 1, 3] = 0
        clip = np.array([[0, 255], [255, 255]], np.uint8)
        editor = ObjectEditor(base)
        editor.add('iris', tile, xy=(2, 1), clip=clip, clip_xy=(3, 1))
        editor.add('eyelid', np.zeros((1, 1, 3), np.uint8), xy=(4, 1))
        expected = np.zeros(base.shape[:2], bool)
        expected[1:3, 4] = True
        self.assertEqual(editor.mask('iris').dtype, np.bool_)
        np.testing.assert_array_equal(editor.mask('iris'), expected)
        np.testing.assert_array_equal(editor.render()[1, 4], base[1, 4])
        editor.set_visible('iris', False)
        np.testing.assert_array_equal(editor.mask('iris'), np.zeros_like(expected))

    def test_inspect_distinguishes_old_new_coverage_and_the_gap_between_eyes(self):
        base = np.zeros((3, 11, 3), np.uint8)
        dot = np.full((1, 1, 3), 250, np.uint8)
        before = ObjectEditor(base)
        after = ObjectEditor(base)
        for editor in (before, after):
            editor.add('left', dot, xy=(1, 1))
            editor.add('right', dot, xy=(7, 1))
        after.move('left', 1, 0)
        after.move('right', 1, 0)
        after.add('stray', dot, xy=(4, 1))

        result = after.inspect(['left', 'right'], before=before, padding=0)
        self.assertEqual(result.metadata['changed_pixels'], 5)
        self.assertEqual(result.metadata['changed_outside_selection'], 1)
        self.assertEqual(result.metadata['selection_bbox'], [1, 1, 9, 2])
        self.assertEqual(result.metadata['crop_bbox'], [1, 1, 9, 2])
        raw = after.inspect(['left', 'right'], before=before.render(), padding=0)
        self.assertEqual(raw.metadata['changed_outside_selection'], 3)
        self.assertEqual(raw.metadata['selection_bbox'], [2, 1, 9, 2])
        current = after.inspect('left', padding=0)
        self.assertEqual(current.metadata['selection_bbox'], [2, 1, 3, 2])
        self.assertEqual(current.metadata['changed_pixels'], 0)


if __name__ == '__main__':
    unittest.main()

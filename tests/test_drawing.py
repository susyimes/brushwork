"""Regressions for source-level edits to the underdrawing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import Studio
from brushwork.drawing import Drawing


def face():
    drawing = Drawing()
    drawing.add('eye', dict(op='background', color='#e0ddc0', region='sclera'))
    drawing.add('iris', [
        dict(op='ellipse', center=[280, 260], rx=60, region='iris', detail=0.5,
             color={'radial': [280, 260, 60], 'colors': ['#102010', '#55704f']},
             direction={'around': [280, 260]}, name='iris_mask'),
        dict(op='ellipse', center=[270, 250], rx=5, color='#ffffff', region='iris'),
    ])
    drawing.add('eyelid', dict(op='rect', x=0, y=0, w=1000, h=248,
                              color='#cc9870', region='skin', detail=2.0,
                              direction={'angle': 15, 'spread': 10}))
    return drawing


class DrawingTests(unittest.TestCase):
    def test_move_restores_background_and_keeps_occluder(self):
        drawing = face()
        original = drawing.render(500, 300)
        drawing.move('iris', 200, 0)
        shifted = drawing.render(500, 300)
        for field in ('target', 'region', 'rmin', 'vx', 'vy'):
            prev, now = getattr(original, field), getattr(shifted, field)
            np.testing.assert_array_equal(now[:122], prev[:122])
            np.testing.assert_allclose(now[130:150, 224:250], prev[130:150, 124:150], atol=1e-6)
        self.assertEqual(shifted.name_of(shifted.region[135, 140]), 'sclera')
        self.assertEqual(shifted.name_of(shifted.region[135, 240]), 'iris')
        self.assertEqual(shifted.rmin[135, 140], 3)

    def test_replace_hide_and_roundtrip_at_new_resolution(self):
        drawing = face()
        drawing.move('iris', 200, 0)
        drawing.replace('iris', dict(op='ribbon', path=[[250, 260], [310, 260]], width=20,
                                     color='#111111', region='iris'))
        painted = Studio(250, 150, sketch=drawing, guide_scale=1, log=None)
        self.assertEqual(painted.sk.name_of(painted.sk.region[65, 120]), 'iris')
        drawing.set_visible('iris', False)
        hidden = drawing.render(250, 150)
        self.assertFalse(hidden.mask('iris').any())
        self.assertEqual(hidden.regions, painted.sk.regions)
        drawing.set_visible('iris', True)
        clone = Drawing(**json.loads(json.dumps(drawing.to_dict())))
        np.testing.assert_array_equal(painted.sk.preview(), clone.render(250, 150).preview())
        self.assertEqual(clone.render(500, 300).region[130, 240], hidden.rid('iris'))

    def test_hidden_named_mask_and_fixed_random_fields(self):
        drawing = face()
        drawing.add('reflection', dict(op='line', path=[[0, 270], [1000, 270]],
                                       width=6, color='#ffffff', on='iris_mask'))
        original = drawing.render(250, 150)
        drawing.set_visible('iris', False)
        hidden = drawing.render(250, 150)
        self.assertFalse(hidden.mask('iris_mask').any())
        np.testing.assert_array_equal(hidden.vx[:56], original.vx[:56])
        np.testing.assert_array_equal(hidden.vy[:56], original.vy[:56])

    def test_retained_source_is_not_mutated_by_callers_or_failed_edit(self):
        drawing = face()
        source = drawing.to_dict()
        with self.assertRaises(ValueError):
            drawing.replace('iris', {'op': 'unknown'})
        with self.assertRaises(ValueError):
            drawing.move('iris', float('nan'), 0)
        self.assertEqual(source, drawing.to_dict())
        source['objects'][0]['ops'][0]['color'] = '#000000'
        self.assertNotEqual(source, drawing.to_dict())

    def test_edit_across_sessions_preserves_canvas_and_invalidates_guide_cache(self):
        studio = Studio(160, 110, guide_scale=1, log=None)
        studio.set_drawing(face().to_dict())
        studio.strokes([dict(path=[[110, 80], [360, 260]], brush=6, color='#654321')])
        painted_before = studio.cv.col.copy()
        old_lab = studio.lab.copy()
        with tempfile.TemporaryDirectory() as tmp:
            studio.save(tmp)
            loaded = Studio.load(tmp, log=None)
        loaded.edit_drawing('iris', 'move', dx=180, dy=0)
        studio.edit_drawing('iris', 'move', dx=180, dy=0)
        self.assertFalse(np.array_equal(old_lab, studio.lab))
        np.testing.assert_array_equal(loaded.sk.preview(), studio.sk.preview())
        np.testing.assert_array_equal(loaded.cv.col, painted_before)
        # Replay the same source and edits at double resolution.
        large = Studio(320, 220, guide_scale=1, log=None)
        large.set_drawing(face().to_dict())
        large.edit_drawing('iris', 'move', dx=180, dy=0)
        self.assertEqual(large.sk.name_of(large.sk.region[86, 147]), 'iris')
        self.assertEqual(loaded.sk.name_of(loaded.sk.region[43, 73]), 'iris')

    def test_manual_guide_changes_never_disappear_silently(self):
        studio = Studio(120, 80, sketch=face(), log=None)
        studio.sk.ellipse((200, 300), 20, color='#ee1111')
        with self.assertRaisesRegex(ValueError, 'Direct sketch edits'):
            studio.edit_drawing('iris', 'hide')
        before = studio.sk.preview()
        with tempfile.TemporaryDirectory() as tmp:
            studio.save(tmp)
            loaded = Studio.load(tmp, log=None)
        np.testing.assert_array_equal(loaded.sk.preview(), before)
        with self.assertRaisesRegex(ValueError, 'no current editable underdrawing'):
            loaded.edit_drawing('iris', 'show')

    def test_visible_object_footprint_respects_occlusion_and_region_ownership(self):
        drawing = Drawing()
        # Matching colours are still distinct painting regions.
        drawing.add('paper', dict(op='background', color='#eeeeee', region='sclera'))
        drawing.add('painted iris', dict(op='ellipse', center=[400, 300], rx=100,
                                         color='#eeeeee', region='pigment'))
        drawing.add('upper lid', dict(op='rect', x=200, y=100, w=400, h=200,
                                      color='#342832', region='skin'))
        sketch = drawing.render(200, 120)
        source = sketch.drawing_source
        mask = sketch.object_mask('painted iris')
        self.assertEqual(mask.shape, (120, 200))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertFalse(mask[50, 80])   # upper lid covers it
        self.assertTrue(mask[70, 80])    # identical colour but now belongs to pigment
        self.assertEqual(source, sketch.drawing_source)
        np.testing.assert_array_equal(mask, drawing.object_mask('painted iris', 200, 120))
        drawing.set_visible('painted iris', False)
        self.assertFalse(drawing.object_mask('painted iris', 200, 120).any())
        with self.assertRaises(KeyError):
            sketch.object_mask('typo')

    def test_visible_object_footprint_includes_downstream_material_fields(self):
        drawing = Drawing()
        drawing.add('eye', dict(op='background', region='sclera', detail=3))
        drawing.add('construction', dict(op='rect', x=200, y=200, w=80, h=80, name='shine'))
        drawing.add('reflective surface', dict(op='sheen_field', mask='shine', amount=0.9))
        mask = drawing.object_mask('construction', 100, 100)
        self.assertTrue(mask[23, 23])
        self.assertFalse(mask[10, 10])


if __name__ == '__main__':
    unittest.main()

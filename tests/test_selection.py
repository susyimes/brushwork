"""Regression checks for pixel-clipped brush selections and their physical recordings."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import Studio, StrokeTape
from brushwork.cli import run_ops
from brushwork.engine import make_params


PAINT_FIELDS = ('col', 'sca', 'wet', 'hgt', 'thk', 'iri', 'ifd', 'stamp')


def studio(guide_scale=1):
    st = Studio(160, 120, seed=41, primer='#ede5d2', guide_scale=guide_scale, log=None)
    st.sk.background('#ede5d2', region='paper')
    st.sk.rect(300, 150, 350, 450, region='eye')
    return st


def stroke(path):
    return dict(path=path, brush=42, color='#38567c', style='impasto', sheen=.7)


class SelectionChecks(unittest.TestCase):
    def test_native_object_id_clips_separately_from_its_shared_region(self):
        st = Studio(160, 120, guide_scale=.5, log=None)
        def eye(id, x):
            return dict(id=id, ops=[dict(op='ellipse', center=[x, 350], rx=130,
                                        ry=85, region='skin', color='#563f46')])
        st.set_drawing(dict(objects=[eye('left', 300), eye('right', 700)]))
        selected = st.rmask({'object': 'left'}, canvas=True)
        run_ops(st, [dict(op='strokes', on={'object': 'left'}, items=[stroke([(0, 350), (990, 350)])])])
        self.assertTrue(st.cv.thk[selected].any())
        self.assertFalse(st.cv.thk[~selected].any())

    def test_two_point_crossing_matches_subdivided_line(self):
        sparse, dense = studio(), studio()
        xs = range(100, 901, 50)
        self.assertEqual(sparse.strokes([stroke([(100, 350), (900, 350)])], on='eye'), 1)
        result = run_ops(dense, [dict(op='strokes', items=[stroke([(x, 350) for x in xs])],
                                    on=['eye'])])
        self.assertEqual(result[0]['strokes'], 1)
        self.assertGreater(np.count_nonzero(sparse.cv.thk), 100)
        self.assertFalse(sparse.cv.thk[~sparse.rmask('eye')].any())
        np.testing.assert_array_equal(sparse.render(lit=False), dense.render(lit=False))

    def test_brush_fringe_reaches_inside_without_changing_any_field_outside(self):
        st = studio(guide_scale=.5)
        st.strokes([dict(path=[(50, 400), (950, 400)], brush=80, style='impasto',
                         color='#c27251', sheen=.5)])
        # The centre line stays outside the region, while half the brush overlaps it.
        before = {name: getattr(st.cv, name).copy() for name in PAINT_FIELDS}
        st.strokes([stroke([(289, 180), (289, 560)])], on='eye')
        mask = st.rmask('eye').repeat(2, 0).repeat(2, 1)
        self.assertGreater(np.count_nonzero(st.cv.thk[mask] != before['thk'][mask]), 10)
        for name in PAINT_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(st.cv, name)[~mask], before[name][~mask])

    def test_union_and_recorded_mask_outlive_the_sketch(self):
        st = studio()
        st.sk.rect(740, 150, 150, 450, name='outer')
        selected = st.sk.mask(['eye', 'outer']) > .5
        with st.capture_strokes() as tape:
            st.strokes([stroke([(50, 350), (965, 350)])], on=['eye', 'outer'])
        self.assertGreater(np.count_nonzero(st.cv.thk[st.rmask('eye')]), 10)
        self.assertGreater(np.count_nonzero(st.cv.thk[st.rmask('outer')]), 10)
        self.assertFalse(st.cv.thk[~selected].any())
        with tempfile.TemporaryDirectory() as temp:
            saved = Path(temp) / 'clipped.brushes'
            tape.save(saved)
            restored = StrokeTape.load(saved)
        # The destination has no 'outer' mask. Replay must use the mask saved with the batch.
        replayed = studio()
        restored.replay(replayed)
        for name in PAINT_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(replayed.cv, name), getattr(st.cv, name))
        np.testing.assert_array_equal(st.render(lit=False), replayed.render(lit=False))

    def test_guide_array_can_protect_a_hole(self):
        st = studio()
        selected = np.ones((st.GH, st.GW), bool)
        selected[50:70, 50:90] = False
        st.strokes([stroke([(30, 375), (970, 375)])], on=selected)
        self.assertFalse(st.cv.thk[~selected].any())
        self.assertTrue(st.cv.thk[60, :45].any())
        self.assertTrue(st.cv.thk[60, 95:].any())

    def test_mixed_resolution_union_preserves_one_pixel_stripes_and_holes(self):
        st, reference = studio(guide_scale=.5), studio(guide_scale=.5)
        stripes = np.zeros((st.H, st.W), bool)
        stripes[50:72, 110:150:2] = True
        stripes[60, 120] = False
        selection = st.rmask('eye').repeat(2, 0).repeat(2, 1) | stripes
        line = stroke([(30, 375), (970, 375)])
        st.strokes([line], on=['eye', stripes])
        reference.strokes([line], on=selection)
        self.assertGreater(st.cv.thk[60, 119:123].max(), 0)
        self.assertEqual(st.cv.thk[60, 120], 0)
        self.assertEqual(st.cv.thk[60, 121], 0)
        self.assertGreater(st.cv.thk[60, 78], 0)
        np.testing.assert_array_equal(st.render(lit=False), reference.render(lit=False))

    def test_cli_explains_how_to_select_with_replayable_json(self):
        st = studio()
        op = dict(op='strokes', items=[stroke([(30, 375), (970, 375)])],
                  on=[[True, False], [False, True]])
        with self.assertRaises(ValueError) as error:
            run_ops(st, [op])
        self.assertRegex(str(error.exception), r'(?i)region|named mask')
        self.assertRegex(str(error.exception), r'(?i)numpy|ndarray')
        self.assertEqual(st.cv.n_strokes, 0)

    def test_disjoint_stroke_reports_zero_and_leaves_canvas_alone(self):
        st = studio()
        before = {name: getattr(st.cv, name).copy() for name in PAINT_FIELDS}
        self.assertEqual(st.strokes([stroke([(20, 45), (960, 45)])], on='eye'), 0)
        self.assertEqual(st.cv.n_strokes, 0)
        for name in PAINT_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(st.cv, name), before[name])

    def test_blend_brush_loads_only_from_the_selected_wet_paint(self):
        a, b = studio(), studio()
        mask = a.rmask('eye')
        # The pool of wet paint inside is identical; only the approach to the eye differs.
        for st, outside in [(a, [.8, .05, .08]), (b, [.05, .35, .8])]:
            st.cv.col[:, 82:] = [.38, .23, .16]
            st.cv.col[~mask] = outside
            st.cv.rewet(np.ones(mask.shape, bool), .9)
        inside = a.cv.col[mask].copy()
        for st in (a, b):
            self.assertEqual(st.strokes([dict(path=[(40, 350), (960, 350)], brush=40,
                                              style='blend')], on='eye'), 1)
        self.assertTrue(np.any(a.cv.col[mask] != inside))
        for name in PAINT_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(a.cv, name)[mask], getattr(b.cv, name)[mask])

    def test_selecting_every_pixel_matches_unrestricted_brushes(self):
        a, b = studio(), studio()
        items = [stroke([(90, 400), (830, 480)]),
                 dict(path=[(900, 435), (110, 460)], brush=70, style='blend')]
        a.strokes(items)
        b.strokes(items, on=np.ones((b.GH, b.GW), bool))
        for name in PAINT_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(a.cv, name), getattr(b.cv, name))
        np.testing.assert_array_equal(a.render(lit=False), b.render(lit=False))

    def test_v1_unclipped_archive_remains_playable(self):
        original, replayed = studio(), studio()
        values = dict(points=np.array([[3, 42], [110, 42]], np.float32),
                      offsets=np.array([0, 2], np.int64),
                      settings=make_params(1, radius=6),
                      tubes=np.zeros((1, 4), np.int32),
                      weights=np.array([[1, 0, 0, 0]], np.float32),
                      order=np.array([0], np.int64), seeds=np.array([79], np.int64))
        original.cv.deposit(*values.values())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'legacy.npz'
            meta = json.dumps(dict(version=1, size=[original.W, original.H], events=[['paint']]))
            np.savez_compressed(path, meta=meta, **{'0_' + k: v for k, v in values.items()})
            StrokeTape.load(path).replay(replayed)
        self.assertTrue(replayed.cv.thk[~replayed.rmask('eye')].any())
        np.testing.assert_array_equal(original.render(lit=False), replayed.render(lit=False))


if __name__ == '__main__':
    unittest.main()

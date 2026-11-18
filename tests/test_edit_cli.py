"""Verify the JSON workflows an agent uses to edit and inspect a saved painting."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import Drawing, ObjectEditor, Studio
from brushwork.cli import main
from brushwork.studio import _imwrite


def call(*args, ok=True):
    output = io.StringIO()
    with redirect_stdout(output):
        result = main(list(map(str, args)))
    report = json.loads(output.getvalue())
    if (result == 0) != ok:
        raise AssertionError(report)
    return report


class EditCliTests(unittest.TestCase):
    def test_guide_preview_includes_interpolated_object_edges(self):
        drawing = Drawing()
        drawing.add('iris', dict(op='ellipse', center=[370, 340], rx=70, ry=50, color='#441122', region='eye'))
        old = Studio(200, 150, sketch=drawing, guide_scale=.5, log=None)
        drawing.move('iris', 95, 0)
        new = Studio(200, 150, sketch=drawing, guide_scale=.5, log=None)
        detail = new.inspect('iris', before=old, what='sketch').metadata
        self.assertGreater(detail['changed_pixels'], 0)
        self.assertEqual(detail['changed_outside_selection'], 0)

    def test_edit_batch_persists_relative_assets_and_preserves_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            doc, ops, base, tile = (p / name for name in ('picture.npz', 'ops.json', '画.png', '虹膜.png'))
            blank = np.zeros((24, 24, 4), np.uint8)
            _imwrite(base, blank)
            _imwrite(tile, np.full((3, 3, 4), (32, 64, 230, 178), np.uint8))
            ops.write_text(json.dumps(dict(op='add', id='eye', image=tile.name, xy=[3, 4])), encoding='utf-8')
            call('edit', doc, ops, '--base', base)
            ops.write_text(json.dumps(dict(op='move', id='eye', dx=6, dy=0)), encoding='utf-8')
            report = call('edit', doc, ops, '--focus', 'eye', '--preview', p/'view.png', '--out', p/'out.png')
            self.assertEqual(report['changed_pixels'], 18)
            self.assertEqual(report['selection_bbox'], [3, 4, 12, 7])
            self.assertEqual(report['changed_outside_selection'], 0)
            after = ObjectEditor.load(doc).render()
            np.testing.assert_array_equal(ObjectEditor(p/'out.png').render(), after)
            np.testing.assert_array_equal(after[4:7, 3:6], blank[4:7, 3:6])
            ops.write_text(json.dumps([dict(op='move', id='eye', dx=2, dy=1),
                                       dict(op='move', id='missing', dx=1, dy=0)]), encoding='utf-8')
            call('edit', doc, ops, ok=False)
            np.testing.assert_array_equal(ObjectEditor.load(doc).render(), after)

    def test_session_inspects_id_distinct_from_region_and_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            session, ops = p / 'session', p / 'do.json'
            call('new', session, '--width', '120', '--height', '90')
            circle = dict(op='ellipse', center=[360, 280], rx=65, color='#39495e', region='features')
            source = dict(objects=[dict(id='left-iris', ops=[circle])])
            ops.write_text(json.dumps(dict(op='set_drawing', source=source)), encoding='utf-8')
            call('do', session, ops)
            self.assertEqual(call('info', session)['objects'], ['left-iris'])
            ops.write_text(json.dumps(dict(op='edit_drawing', id='left-iris', action='move', dx=100, dy=15)),
                           encoding='utf-8')
            call('do', session, ops)
            detail = call('inspect', session, '--object', 'left-iris', '--what', 'sketch', '--out', p/'detail.png')
            self.assertEqual(detail['baseline'], 'previous_batch')
            self.assertGreater(detail['changed_pixels'], 0)
            self.assertEqual(detail['changed_outside_selection'], 0)
            self.assertLess(detail['crop_bbox'][2], 120)
            call('undo', session)
            self.assertEqual(call('info', session)['objects'], ['left-iris'])


if __name__ == '__main__':
    unittest.main()

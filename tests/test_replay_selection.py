"""Check that HTML replay masks address the strokes actually painted, in display coordinates."""
import base64
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork.engine import make_params
from brushwork.replay import Recorder, read


class ReplaySelectionTest(unittest.TestCase):
    def test_clipped_ranges_and_alpha(self):
        recorder = Recorder(100, 60, .1, frame_width=50)
        pts = np.array([[4, 20], [96, 20], [4, 40], [96, 40]], dtype=np.float32)
        offsets = np.array([0, 2, 4])
        params = make_params(2, radius=5)
        base = np.zeros((60, 100, 3), np.uint8)
        mask = np.zeros((60, 100), bool)
        mask[13:48, 21:59] = True
        mask[24:40, 30:50] = False
        recorder.keyframe('base', base)
        # Only the second stroke touched paint. Its data live at offset index 1.
        recorder.strokes(pts, offsets, params, [1], clip=mask)
        recorder.strokes(pts, offsets, params, [0], clip=mask.copy())
        recorder.strokes(pts, offsets, params, [0])
        mask[16:20, 24:28] = False
        recorder.strokes(pts, offsets, params, [1], clip=mask)
        mask[16:20, 24:28] = True
        recorder.strokes(pts, offsets, params, [1], clip=mask)
        recorder.keyframe('paint', 255 - base)
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp, 'replay.html')
            recorder.export(page)
            meta, _, data = read(page)

        self.assertEqual(meta['strokes'], 5)
        self.assertEqual(len(data['counts']), 5)
        self.assertEqual(meta['clipRanges'], [[0, 2, 0], [3, 4, 1], [4, 5, 0]])
        self.assertEqual(len(meta['clips']), 2)
        clip = meta['clips'][0]
        png = base64.b64decode(clip['src'].split(',')[1])
        tile = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        x, y = clip['x'], clip['y']
        h, w = tile.shape[:2]
        alpha = np.zeros((30, 50), np.uint8)
        alpha[y:y+h, x:x+w] = tile[:, :, 3]
        expected = cv2.resize(mask.astype(np.uint8) * 255, (50, 30), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(alpha, expected)


if __name__ == '__main__':
    unittest.main()

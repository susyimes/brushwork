"""An archive should keep one snapshot per distinct brush selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import StrokeTape, Studio
from brushwork.engine import make_params


WIDTH, HEIGHT = 97, 65  # deliberately not divisible by eight when packed


def brush():
    return dict(points=np.array([[2, 30], [95, 30]], np.float32),
                offsets=np.array([0, 2], np.int64), settings=make_params(1, radius=8),
                tubes=np.zeros((1, 4), np.int32), weights=np.array([[1, 0, 0, 0]], np.float32),
                order=np.array([0], np.int64), seeds=np.array([37], np.int64))


def canvas():
    return Studio(WIDTH, HEIGHT, seed=44, log=None)


class ClipStorageChecks(unittest.TestCase):
    def test_equal_content_shares_storage_and_mutation_creates_a_snapshot(self):
        tape = StrokeTape(WIDTH, HEIGHT)
        values = brush().values()
        mask = np.zeros((HEIGHT, WIDTH), bool)
        mask[:, :35] = True
        left = mask.copy()
        tape.batch(*values, clip=mask)
        tape.batch(*values, clip=np.asfortranarray(mask))
        mask[:] = ~mask
        right = mask.copy()
        tape.batch(*values, clip=mask)
        tape.batch(*values, clip=left.copy())
        mask[:] = False
        self.assertEqual(len(tape._clip_ids), 2)
        self.assertEqual(sum(map(len, tape._clip_ids)), 2 * ((WIDTH * HEIGHT + 7) // 8))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'shared.npz'
            tape.save(path)
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['meta']))
                self.assertEqual(meta['version'], 3)
                self.assertEqual(meta['events'], [['clipped_paint', i] for i in [0, 0, 1, 0]])
                self.assertEqual(set(k for k in archive.files if 'clip' in k), {'clip_0', 'clip_1'})
            loaded = StrokeTape.load(path)

        expected = canvas()
        for selected in (left, left, right, left):
            expected.cv.deposit(*values, clip=selected)
        self.assertTrue(expected.cv.thk[left].any() and expected.cv.thk[right].any())
        for source in (tape, loaded):
            replayed = canvas()
            source.replay(replayed)
            np.testing.assert_array_equal(replayed.cv.col, expected.cv.col)
            np.testing.assert_array_equal(replayed.cv.thk, expected.cv.thk)

    def test_legacy_v1_and_per_event_v2_replay(self):
        values = brush()
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                arrays = {'0_' + k: v for k, v in values.items()}
                mask = None
                events = [['paint']]
                if version == 2:
                    mask = np.zeros((HEIGHT, WIDTH), bool)
                    mask[:, :47] = True
                    arrays['0_clip'] = mask
                    events = [['clipped_paint']]
                path = Path(temp) / 'legacy.npz'
                arrays['meta'] = json.dumps(dict(version=version, size=[WIDTH, HEIGHT], events=events))
                np.savez_compressed(path, **arrays)
                restored = StrokeTape.load(path)
                expected, replayed = canvas(), canvas()
                expected.cv.deposit(*values.values(), clip=mask)
                restored.replay(replayed)
                np.testing.assert_array_equal(replayed.cv.thk, expected.cv.thk)
                restored.save(path)
                with np.load(path, allow_pickle=False) as archive:
                    saved = json.loads(str(archive['meta']))
                    self.assertEqual(saved['version'], 1 if version == 1 else 3)


if __name__ == '__main__':
    unittest.main()

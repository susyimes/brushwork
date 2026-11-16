"""Physical brush recordings. All path coordinates and radii here are canvas pixels."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .color import PIG_K
from .engine import NPAR, _COLS


class StrokeBatch:
    """Editable pose of one native batch. IDs, path boundaries, order and bristle seeds stay fixed."""
    def __init__(self, arrays, first_id):
        self.points, self.offsets, self.settings, self.tubes, self.weights, self.order, self.seeds = arrays
        self.ids = np.arange(first_id, first_id + len(self.seeds), dtype=np.int64)

    @property
    def centers(self):
        return .5 * (self.points[self.offsets[:-1]] + self.points[self.offsets[1:] - 1])

    def parameter(self, name):
        """Writable per-stroke values, e.g. batch.parameter('opacity')[:] *= gain."""
        return self.settings[:, _COLS[name]]

    def translate(self, dx, dy):
        """Translate rigid strokes by a scalar or one displacement per ID."""
        n = len(self.ids)
        shift = np.column_stack((np.broadcast_to(dx, (n,)), np.broadcast_to(dy, (n,))))
        self.points += np.repeat(shift, np.diff(self.offsets), axis=0)


_KEYS = ('points', 'offsets', 'settings', 'tubes', 'weights', 'order', 'seeds')
_DTYPES = (np.float32, np.int64, np.float32, np.int32, np.float32, np.int64, np.int64)


def _validate(arrays):
    pts, off, sp, ci, cw, order, seed = arrays
    n = len(seed)
    if (seed.shape != (n,) or off.shape != (n+1,) or off[0] != 0 or off[-1] != len(pts)
            or np.any(np.diff(off) < 2) or pts.shape != (len(pts), 2) or sp.shape != (n, NPAR)
            or ci.shape != (n, 4) or cw.shape != (n, 4) or order.ndim != 1
            or np.any(order < 0) or np.any(order >= n)
            or np.any(ci < 0) or np.any(ci >= len(PIG_K))
            or not all(np.isfinite(a).all() for a in (pts, sp, cw))):
        raise ValueError('Invalid geometry, pigment recipe or order in a recorded brush batch')


class StrokeTape:
    """Native paint batches interleaved with drying, rewetting and surface-tooth updates."""
    def __init__(self, width, height):
        self.size = (width, height)
        self.count = 0
        self._events = []

    def batch(self, *values):
        arrays = tuple(np.array(a, dtype=t, copy=True, order='C') for a, t in zip(values, _DTYPES))
        _validate(arrays)
        for a in arrays:
            a.flags.writeable = False
        self._events.append(('paint', arrays))
        self.count += len(arrays[-1])

    def dry(self, amount):
        self._events.append(('dry', float(amount)))

    def tooth(self):
        self._events.append(('tooth', None))

    def rewet(self, mask, amount):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.size[::-1]:
            raise ValueError('Rewet mask must match the recorded canvas')
        self._events.append(('rewet', (mask.copy(), float(amount))))

    def replay(self, studio, frame=None, motion=None):
        """Paint on the supplied physical state. motion(batch, frame) may change a working pose."""
        cv = studio.cv
        if (cv.W, cv.H) != self.size:
            raise ValueError('Replay on a canvas of the recorded size; resize the finished frame for export')
        if cv.stroke_tape is self:
            raise ValueError('A recording cannot replay into itself')
        first = 0
        for name, data in self._events:
            if name == 'dry':
                cv.dry(data)
            elif name == 'tooth':
                cv.update_tooth()
            elif name == 'rewet':
                cv.rewet(*data)
            else:
                if motion is None:
                    cv.deposit(*data)
                else:
                    pts, off, sp, ci, cw, order, seeds = data
                    pose = StrokeBatch((pts.copy(), off, sp.copy(), ci.copy(), cw.copy(), order, seeds), first)
                    motion(pose, frame)
                    # The kernel uses unchecked integer indexing. Validate edits before depositing.
                    posed = (pose.points, off, pose.settings, pose.tubes, pose.weights, order, seeds)
                    _validate(posed)
                    cv.deposit(*posed)
                first += len(data[-1])

    def save(self, path):
        """Write a versioned NPZ with numeric arrays; the underpainting is supplied at replay."""
        arrays, events = {}, []
        for i, (name, data) in enumerate(self._events):
            if name == 'paint':
                arrays.update({f'{i}_{key}': a for key, a in zip(_KEYS, data)})
                events.append([name])
            elif name == 'rewet':
                arrays[f'{i}_mask'] = data[0]
                events.append([name, data[1]])
            else:
                events.append([name, data])
        arrays['meta'] = json.dumps(dict(version=1, size=self.size, events=events))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
            temp = Path(fp.name)
        try:
            with temp.open('wb') as fp:
                np.savez_compressed(fp, **arrays)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as file:
            meta = json.loads(str(file['meta']))
            if meta['version'] != 1:
                raise ValueError('Unsupported stroke recording version')
            tape = cls(*meta['size'])
            for i, event in enumerate(meta['events']):
                name = event[0]
                if name == 'paint':
                    tape.batch(*(file[f'{i}_{key}'] for key in _KEYS))
                elif name == 'rewet':
                    tape.rewet(file[f'{i}_mask'], event[1])
                elif name == 'dry':
                    tape.dry(event[1])
                elif name == 'tooth':
                    tape.tooth()
                else:
                    raise ValueError(f'Unknown recording event {name!r}')
        return tape


@contextmanager
def capture_strokes(studio, seed=None):
    cv = studio.cv
    if cv.stroke_tape is not None:
        raise ValueError('Finish the current stroke recording before starting another')
    tape = StrokeTape(cv.W, cv.H)
    previous = studio.rng, cv.rng
    if seed is not None:
        studio.rng, cv.rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    cv.stroke_tape = tape
    try:
        yield tape
    finally:
        cv.stroke_tape = None
        studio.rng, cv.rng = previous

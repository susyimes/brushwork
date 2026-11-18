"""Editable underdrawings in the same 1000-unit coordinate space as Sketch.

Each object holds one or more JSON sketch operations, ordered back to front.
Pass a Drawing to Studio(sketch=drawing) to rasterise its current revision at
the requested guide resolution. An already painted Studio owns its snapshot.
"""
from copy import deepcopy
import json
import math

import numpy as np

from .sketch import Sketch


SHAPES = ('background', 'ellipse', 'rect', 'polygon', 'spline', 'ribbon')
OPERATIONS = SHAPES + ('line', 'tint', 'form_shadow', 'form_light', 'cast_shadow',
                       'direction', 'detail', 'sheen_field', 'region_props')


def _operations(ops):
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, (list, tuple)) or not ops:
        raise ValueError('an object needs at least one sketch operation')
    for op in ops:
        if not isinstance(op, dict) or op.get('op') not in OPERATIONS:
            raise ValueError('unknown sketch operation: %r' % (op,))
    # The retained source is also written into Studio sessions and CLI replays.
    # Normalise tuples to lists and report non-serialisable parameters on entry.
    return json.loads(json.dumps(ops, allow_nan=False))


def _offset(dx, dy):
    if not all(math.isfinite(v) for v in (dx, dy)):
        raise ValueError('drawing offsets must be finite')
    return [float(dx), float(dy)]


def _translated(op, dx, dy):
    """Shift absolute geometry, colour ramps and flow guides together.

    Radii, widths, shadow offsets and lighting vectors are relative lengths.
    """
    args = deepcopy(op)
    args.pop('op')
    if dx == dy == 0:
        return args

    def point(p):
        return [p[0] + dx, p[1] + dy, *p[2:]]

    def translate_keys(spec, points=(), paths=()):
        for key in points:
            if spec.get(key) is not None:
                spec[key] = point(spec[key])
        for key in paths:
            if spec.get(key) is not None:
                spec[key] = [point(p) for p in spec[key]]

    if op['op'] == 'rect':
        args['x'] += dx
        args['y'] += dy
    translate_keys(args, ('center', 'blob'), ('points', 'path', 'ramp'))
    color = args.get('color')
    if isinstance(color, dict):
        translate_keys(color, ('radial',), ('linear',))
    direction = args.get('spec') if op['op'] == 'direction' else args.get('direction')
    if isinstance(direction, dict):
        translate_keys(direction, ('around', 'radial', 'swirl'), ('along',))
    return args


class Drawing:
    """A retained source for a Sketch. Object IDs are independent of regions.

    move() translates every operation in the object, in canvas units.
    replace() installs new source operations at the same stack position and
    retains its existing offset and visibility. to_dict() contains everything
    needed to recreate the drawing with Drawing(**data).
    """
    def __init__(self, objects=(), *, seed=11, base='#8a8580'):
        self.seed = seed
        self.base = deepcopy(base)
        self._objects = {}
        for object_spec in objects:
            self.add(**object_spec)

    @property
    def object_ids(self):
        return tuple(self._objects)

    def add(self, id, ops, *, offset=(0, 0), visible=True):
        if not isinstance(id, str) or not id or id in self._objects:
            raise ValueError('expected a unique, nonempty object id: %r' % (id,))
        if not isinstance(visible, bool):
            raise TypeError('visible must be a boolean')
        self._objects[id] = dict(id=id, ops=_operations(ops), offset=_offset(*offset), visible=visible)

    def move(self, id, dx, dy):
        obj = self._objects[id]
        x, y = obj['offset']
        obj['offset'] = _offset(x + dx, y + dy)

    def replace(self, id, ops):
        self._objects[id]['ops'] = _operations(ops)

    def set_visible(self, id, visible):
        if not isinstance(visible, bool):
            raise TypeError('visible must be a boolean')
        self._objects[id]['visible'] = visible

    def to_dict(self):
        return deepcopy(dict(objects=list(self._objects.values()), seed=self.seed, base=self.base))

    def object_mask(self, id, width, height):
        """Return the object's visible contribution at the requested guide resolution."""
        return self.render(width, height).object_mask(id)

    def render(self, width, height):
        """Build a fresh native Sketch at guide-pixel dimensions width × height."""
        sketch = Sketch(width, height, seed=self.seed, base=self.base)
        # Reserve region IDs even for hidden shapes.
        for obj in self._objects.values():
            for op in obj['ops']:
                if op['op'] in SHAPES or op['op'] == 'region_props':
                    region = op.get('region', 'background' if op['op'] == 'background' else None)
                    if region is not None:
                        sketch.rid(region)
        empty = None
        for index, obj in enumerate(self._objects.values()):
            if not obj['visible']:
                # Later operations may clip to a hidden object's named mask.
                for op in obj['ops']:
                    name = op.get('name')
                    if name and op['op'] in SHAPES and name not in sketch.masks:
                        if empty is None:
                            empty = np.zeros((height, width), np.float32)
                        sketch.masks[name] = empty
                continue
            # Hiding or replacing earlier objects must not reroll this one's noise.
            sketch.rng = np.random.default_rng([self.seed, index])
            for op in obj['ops']:
                args = _translated(op, *obj['offset'])
                getattr(sketch, op['op'])(**args)
        sketch._attach_drawing(self.to_dict())
        return sketch

    __call__ = render

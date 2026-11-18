"""Editable, ordered paint objects on a finished RGB or RGBA image.

All coordinates are integer pixels in the original image. An object owns a local
tile and may have a separate, stationary canvas matte, e.g. an eyelid opening.
"""
from dataclasses import dataclass
import json
import os
from operator import index
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .animation import _alpha, _composite, _image
from .inspection import inspect_edit


def _decode(path):
    data = np.fromfile(path, dtype=np.uint8)
    pixels = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if pixels is None or pixels.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit image in {path}')
    return pixels


def _pixels(source):
    if isinstance(source, (str, os.PathLike)):
        raw = _decode(source)
        if raw.ndim == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        conversion = cv2.COLOR_BGRA2RGBA if raw.shape[2] == 4 else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(raw, conversion)
    return _image(source).copy()


def _matte(source):
    if isinstance(source, (str, os.PathLike)):
        source = _decode(source)
        if source.ndim == 3:
            source = source[..., 3] if source.shape[2] == 4 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if not np.size(source):
        raise ValueError('A matte must have at least one pixel')
    return _alpha(source)


def _xy(pair):
    x, y = pair
    return index(x), index(y)


@dataclass
class _Object:
    image: np.ndarray
    xy: tuple
    clip: np.ndarray | None
    clip_xy: tuple
    visible: bool = True

    def views(self, size):
        """Intersect the canvas, tile and stationary matte, then take matching views."""
        if not self.visible:
            return None
        x, y = self.xy
        h, w = self.image.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(size[0], x + w), min(size[1], y + h)
        clip = self.clip
        if clip is not None:
            cx, cy = self.clip_xy
            mh, mw = clip.shape
            x0, y0 = max(x0, cx), max(y0, cy)
            x1, y1 = min(x1, cx + mw), min(y1, cy + mh)
        if x1 <= x0 or y1 <= y0:
            return None
        if clip is not None:
            clip = clip[y0-cy:y1-cy, x0-cx:x1-cx]
        return (x0, y0, x1, y1), self.image[y0-y:y1-y, x0-x:x1-x], clip


def _coverage(tile, clip):
    if tile.shape[2] == 3:
        return clip
    alpha = tile[..., 3].astype(np.float32) / 255
    return alpha if clip is None else alpha * clip


def _over_rgba(dst, tile, clip):
    """Straight-alpha source-over, retaining transparent RGB where coverage is zero."""
    alpha = _coverage(tile, clip)
    if alpha is None:
        dst[..., :3] = tile
        dst[..., 3] = 255
        return
    result_alpha = alpha + dst[..., 3] * ((1.0 - alpha) / 255)
    weight = np.divide(alpha, result_alpha, out=np.zeros_like(alpha), where=result_alpha > 0)
    dst[..., :3] = np.rint(dst[..., :3] + (tile[..., :3].astype(np.float32) - dst[..., :3]) * weight[..., None])
    dst[..., 3] = np.rint(result_alpha * 255)


class ObjectEditor:
    """Keep named RGB(A) tiles over a reusable base. Earlier objects paint underneath later ones.

    Accepts an H x W uint8 RGB or RGBA base, a PNG path, or the result of Studio.render().
    When editing a flattened painting, prepare the exposed background and the moving
    feature separately. A flat file cannot supply the pixels originally under that feature.
    """

    def __init__(self, base):
        self._base = _pixels(base)
        self._objects = {}

    @property
    def size(self):
        """Canvas width and height in pixels."""
        return self._base.shape[1], self._base.shape[0]

    @property
    def object_ids(self):
        """Stable object IDs in drawing order, from bottom to top."""
        return tuple(self._objects)

    def add(self, id, image, xy=(0, 0), *, clip=None, clip_xy=(0, 0)):
        """Add an RGB(A) tile or image path. clip is a stationary canvas matte.

        Array mattes may be uint8 (0..255) or bool/float (0..1). An image path uses
        grayscale/brightness, or its alpha channel for an RGBA image.
        """
        if not isinstance(id, str) or not id.strip() or id in self._objects:
            raise ValueError(f'Object ID must be unique and nonempty: {id!r}')
        self._objects[id] = _Object(_pixels(image), _xy(xy), None if clip is None else _matte(clip), _xy(clip_xy))

    def add_mask(self, id, mask, color, xy=(0, 0), *, clip=None, clip_xy=(0, 0)):
        """Make a solid-color object from a brush/shape coverage mask and an 8-bit RGB triple or #rrggbb."""
        alpha = _matte(mask)
        if isinstance(color, str):
            if len(color) != 7 or not color.startswith('#'):
                raise ValueError('Use #rrggbb or three integer color components in 0..255')
            color = bytes.fromhex(color[1:])
            color = tuple(color)
        rgb = tuple(index(c) for c in color)
        if len(rgb) != 3 or any(not 0 <= c <= 255 for c in rgb):
            raise ValueError('Use #rrggbb or three integer color components in 0..255')
        tile = np.empty((*alpha.shape, 4), np.uint8)
        tile[..., :3] = rgb
        tile[..., 3] = np.rint(alpha * 255)
        self.add(id, tile, xy, clip=clip, clip_xy=clip_xy)

    def move(self, id, dx, dy):
        """Translate one object's tile by an integer pixel offset."""
        obj = self._objects[id]
        x, y = obj.xy
        obj.xy = (x + index(dx), y + index(dy))

    def replace(self, id, image, xy=None):
        """Replace one object's tile, preserving its order, visibility and fixed matte."""
        obj = self._objects[id]
        obj.image, obj.xy = _pixels(image), obj.xy if xy is None else _xy(xy)

    def set_visible(self, id, visible):
        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError('Object visibility must be true or false')
        self._objects[id].visible = bool(visible)

    def mask(self, id):
        """Return this visible object's alpha footprint as a full-canvas boolean array.

        The footprint honors the stationary matte and the canvas edges. Higher
        objects can occlude pixels in this footprint. A hidden object has an empty mask.
        """
        mask = np.zeros(self._base.shape[:2], bool)
        views = self._objects[id].views(self.size)
        if views is not None:
            (x0, y0, x1, y1), tile, clip = views
            coverage = _coverage(tile, clip)
            mask[y0:y1, x0:x1] = True if coverage is None else coverage > 0
        return mask

    def bbox(self, id):
        """Return half-open (left,top,right,bottom) coverage bounds, or None if empty.

        Bounds include pixels underneath higher objects; they exclude a hidden
        object's entire tile and any pixels outside the canvas or its own matte.
        """
        views = self._objects[id].views(self.size)
        if views is None:
            return None
        bounds, tile, clip = views
        coverage = _coverage(tile, clip)
        if coverage is None:
            return bounds
        rows = np.flatnonzero(coverage.any(axis=1))
        if not len(rows):
            return None
        cols = np.flatnonzero(coverage.any(axis=0))
        x, y = bounds[:2]
        return x + int(cols[0]), y + int(rows[0]), x + int(cols[-1]) + 1, y + int(rows[-1]) + 1

    def render(self):
        """Composite independent objects over a fresh base and return a uint8 RGB(A) array."""
        image = self._base.copy()
        for obj in self._objects.values():
            views = obj.views(self.size)
            if views is None:
                continue
            (x0, y0, x1, y1), tile, clip = views
            dst = image[y0:y1, x0:x1]
            if image.shape[2] == 3:
                _composite(dst, tile, (0, 0), clip)
            else:
                _over_rgba(dst, tile, clip)
        return image

    def inspect(self, id, *, before=None, padding=16, path=None):
        """Inspect one ID or several together, including pixels between disjoint objects in the preview.

        Against another ObjectEditor, the selected area is the union of the old
        and new masks for each ID. A before image has no object metadata, so only
        current masks are available. Omit before to view the current focus alone.
        """
        names = (id,) if isinstance(id, str) else tuple(id)
        if not names:
            raise ValueError('Select at least one object ID to inspect')
        documents = [self]
        if isinstance(before, ObjectEditor):
            if before.size != self.size:
                raise ValueError('Both object documents must have the same canvas size')
            documents.append(before)
        selected = np.zeros(self._base.shape[:2], bool)
        for name in names:
            found = False
            for document in documents:
                if name in document._objects:
                    selected |= document.mask(name)
                    found = True
            if not found:
                raise KeyError(name)
        after = self.render()
        if isinstance(before, ObjectEditor):
            before = before.render()
        return inspect_edit(after if before is None else before, after, mask=selected, padding=padding, path=path)

    def save(self, path):
        """Atomically save the base, named tiles and stationary mattes in a single NPZ archive."""
        assets = {'base': self._base}
        records = []
        for i, (id, obj) in enumerate(self._objects.items()):
            assets[f'tile_{i}'] = obj.image
            if obj.clip is not None:
                assets[f'clip_{i}'] = obj.clip
            records.append(dict(id=id, xy=obj.xy, clip_xy=obj.clip_xy, visible=obj.visible))
        assets['metadata'] = json.dumps(dict(format='brushwork-objects', version=1, objects=records), ensure_ascii=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = Path(tmp.name)
        try:
            with name.open('wb') as stream:
                np.savez_compressed(stream, **assets)
            os.replace(name, path)
        finally:
            name.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Open an editable archive written by save()."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data['metadata'].item())
            if meta.get('format') != 'brushwork-objects' or meta.get('version') != 1:
                raise ValueError('Unsupported Brushwork object document')
            editor = cls(data['base'])
            for i, row in enumerate(meta['objects']):
                id = row['id']
                editor.add(id, data[f'tile_{i}'], row['xy'], clip=data.get(f'clip_{i}'), clip_xy=row['clip_xy'])
                editor.set_visible(id, row['visible'])
        return editor

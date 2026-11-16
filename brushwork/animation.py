"""Looping painted layers, fresh drawings on exposures, and physically re-laid strokes."""
from dataclasses import dataclass
import math
from contextlib import suppress
from operator import index
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock

import cv2
import numpy as np

from .color import srgb_to_lin


@dataclass(frozen=True)
class Frame:
    index: int
    count: int
    fps: int

    @property
    def seconds(self):
        return self.index / self.fps

    @property
    def phase(self):
        return self.index / self.count


def _image(image):
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4) or min(a.shape[:2]) < 1:
        raise ValueError('A painted layer needs an H x W x 3/4 uint8 RGB or RGBA image')
    return a


def _alpha(mask):
    a = np.asarray(mask)
    if a.ndim != 2:
        raise ValueError('A matte must be a two-dimensional image')
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Use a uint8 matte or coverage in the range 0..1')
    return a.astype(np.float32)


def _origin(origin, frame):
    x, y = origin(frame) if callable(origin) else origin
    return index(x), index(y)


def _composite(canvas, tile, xy, clip=None):
    tile = _image(tile)
    h, w = tile.shape[:2]
    H, W = canvas.shape[:2]
    x, y = xy
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
    if clip is not None and clip.shape != (h, w):
        raise ValueError('Destination matte and painted tile must have the same size')
    if x1 <= x0 or y1 <= y0:
        return
    box = np.s_[y0-y:y1-y, x0-x:x1-x]
    dst = canvas[y0:y1, x0:x1]
    src = tile[box]
    coverage = clip[box] if clip is not None else None
    if src.shape[2] == 4:
        a = src[..., 3].astype(np.float32) / 255
        coverage = a if coverage is None else a * coverage
    if coverage is None:
        dst[:] = src[..., :3]
    else:
        dst[:] = np.rint(dst + (src[..., :3].astype(np.float32) - dst) * coverage[..., None])


class PaintedLayer:
    """An RGB(A) painting. warp(frame,x,y) returns inverse sampling offsets in local pixels."""
    def __init__(self, image, *, origin=(0, 0), warp=None, clip=None, mesh=1, edge='transparent'):
        self.image = _image(image).copy()
        self.origin, self.warp = origin, warp
        self.clip = None if clip is None else _alpha(clip)
        if edge not in ('transparent', 'reflect') or index(mesh) < 1:
            raise ValueError('Choose a positive mesh step and a transparent or reflect edge')
        self.border = cv2.BORDER_CONSTANT if edge == 'transparent' else cv2.BORDER_REFLECT_101
        if warp is not None:
            h, w = self.image.shape[:2]
            self.x = np.arange(w, dtype=np.float32)[None, :]
            self.y = np.arange(h, dtype=np.float32)[:, None]
            gw, gh = (w+mesh-1)//mesh, (h+mesh-1)//mesh
            self.gx = ((np.arange(gw, dtype=np.float32)+.5)*w/gw-.5)[None, :]
            self.gy = ((np.arange(gh, dtype=np.float32)+.5)*h/gh-.5)[:, None]
            self.pixels = self.image
            if self.image.shape[2] == 4 or edge == 'transparent':
                alpha = (self.image[..., 3:4] / np.float32(255) if self.image.shape[2] == 4
                         else np.ones((h, w, 1), np.float32))
                self.pixels = np.concatenate((self.image[..., :3]*alpha, 255*alpha), axis=2)

    def apply(self, canvas, frame):
        tile = self.image
        if self.warp is not None:
            dx, dy = self.warp(frame, self.gx, self.gy)
            h, w = tile.shape[:2]
            maps = []
            for delta, coord in ((dx, self.x), (dy, self.y)):
                field = np.broadcast_to(np.asarray(delta, np.float32), (self.gy.size, self.gx.size))
                if field.shape != (h, w):
                    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
                maps.append(np.ascontiguousarray(coord + field))
            tile = cv2.remap(self.pixels, *maps, cv2.INTER_LINEAR, borderMode=self.border)
            if tile.shape[2] == 4:
                # Interpolate premultiplied colour to avoid a fringe from transparent texels.
                tile[..., :3] *= 255 / np.maximum(tile[..., 3:4], 1e-6)
                tile = np.uint8(np.clip(np.rint(tile), 0, 255))
        _composite(canvas, tile, _origin(self.origin, frame), self.clip)


class RepaintLayer:
    """Start each exposed drawing from factory()'s Studio. Set hold=2 to draw on twos."""
    def __init__(self, factory, draw, *, origin=(0, 0), clip=None, hold=1, active=None):
        self.factory, self.draw = factory, draw
        self.origin, self.active, self.hold = origin, active, index(hold)
        if self.hold < 1:
            raise ValueError('An exposure must last at least one frame')
        self.clip = None if clip is None else _alpha(clip)
        self._cached = (None, None)

    def apply(self, canvas, frame):
        pose = Frame(frame.index - frame.index % self.hold, frame.count, frame.fps)
        if self.active is not None and not self.active(pose):
            return
        last, tile = self._cached
        if pose != last:
            studio = self.factory()
            result = self.draw(studio, pose)
            tile = _image(studio.render() if result is None else result).copy()
            self._cached = pose, tile
        _composite(canvas, tile, _origin(self.origin, frame), self.clip)


# Invert the highlight shoulder of Canvas.render when laying wet paint on a lit RGB plate.
_PLATE = srgb_to_lin(np.arange(256) / 255)
_WHITE = _PLATE > .82
_PLATE[_WHITE] = .82 + .18 * np.arctanh(np.minimum((_PLATE[_WHITE] - .82)/.18, .995))
_PLATE = _PLATE.astype(np.float32)


def paint_over(studio, rgb, draw, **light):
    """Apply the visible difference of a real brush pass to an already lit RGB underpainting."""
    rgb = _image(rgb)
    if rgb.shape != studio.cv.col.shape:
        raise ValueError('The RGB underpainting must match the Studio canvas')
    studio.cv.col[:] = _PLATE[rgb]
    light.setdefault('dither', False)
    before = studio.render(**light)
    draw(studio)
    after = studio.render(**light)
    return np.clip(rgb.astype(np.int16) + after - before, 0, 255).astype(np.uint8)


class StrokeLayer:
    """Restore the physical base for each frame and paint an optional moving pose of a StrokeTape."""
    def __init__(self, factory, tape, motion=None, *, origin=(0, 0), clip=None, over=False, light=None):
        self.studio = factory()
        cv = self.studio.cv
        if tape.size != (cv.W, cv.H):
            raise ValueError('Stroke tape and Studio dimensions must match')
        self.tape, self.motion, self.origin, self.over = tape, motion, origin, over
        self.clip = None if clip is None else _alpha(clip)
        self.light = dict(light or {})
        # Weave and grain describe the support. Only these material fields change under a brush.
        self.base = {n: getattr(cv, n).copy() for n in ('col','sca','wet','hgt','thk','iri','ifd','tooth')}
        self.counts = (cv.n_strokes, cv.n_trails, cv.n_pix)
        self._lock = Lock()

    def apply(self, canvas, frame):
        # Simultaneous preview/export requests use this same physical work surface.
        with self._lock:
            self._paint(canvas, frame)

    def _paint(self, canvas, frame):
        studio = self.studio
        cv = studio.cv
        for name, data in self.base.items():
            np.copyto(getattr(cv, name), data)
        cv.stamp.fill(0)
        cv._stamp_base = 0
        cv.n_strokes, cv.n_trails, cv.n_pix = self.counts
        x, y = _origin(self.origin, frame)
        def draw(st):
            self.tape.replay(st, frame, self.motion)
        if self.over:
            h, w = canvas.shape[:2]
            if 0 <= x <= w - cv.W and 0 <= y <= h - cv.H:
                plate = canvas[y:y+cv.H, x:x+cv.W]
            else:
                ix = np.clip(x + np.arange(cv.W), 0, w-1)
                iy = np.clip(y + np.arange(cv.H), 0, h-1)
                plate = canvas[iy[:, None], ix]
            tile = paint_over(studio, plate, draw, **self.light)
        else:
            draw(studio)
            tile = studio.render(**self.light)
        _composite(canvas, tile, (x, y), self.clip)


class Animation:
    """Ordered layer composition on an integer frame clock. Frame count wraps to frame zero."""
    def __init__(self, background, *, seconds=8, fps=30):
        self.background = _image(background).copy()
        if self.background.shape[2] != 3:
            raise ValueError('Use an RGB backdrop; upper layers may have alpha')
        self.fps = index(fps)
        n = seconds * self.fps
        if self.fps <= 0 or not math.isfinite(n) or n < 1 or not math.isclose(n, round(n), abs_tol=1e-8):
            raise ValueError('Duration and fps must give a positive whole number of frames')
        self.count = round(n)
        self.layers = {}

    def add(self, name, layer):
        if not name or name in self.layers:
            raise ValueError(f'Layer name must be unique and nonempty: {name!r}')
        self.layers[name] = layer
        return layer

    def frame(self, number, layers=None):
        frame = Frame(index(number) % self.count, self.count, self.fps)
        if isinstance(layers, str):
            layers = (layers,)
        names = set(self.layers if layers is None else layers)
        unknown = names.difference(self.layers)
        if unknown:
            raise ValueError(f'Unknown animation layers: {sorted(unknown)}')
        canvas = self.background.copy()
        for name, layer in self.layers.items():
            if name in names:
                layer.apply(canvas, frame)
        return canvas

    def write_mp4(self, path, *, layers=None, width=None):
        """Encode exactly one cycle. A failed paint or encode leaves the previous output intact."""
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise RuntimeError('MP4 export requires FFmpeg with libx264 on PATH')
        if layers is not None and not isinstance(layers, str):
            layers = tuple(layers)
        h, w = self.background.shape[:2]
        ow = w if width is None else index(width)
        oh = h if width is None else 2*round(h*ow/w/2)
        if min(ow, oh) < 2 or ow % 2 or oh % 2:
            raise ValueError('H.264 output needs positive even dimensions')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix='.brushwork-') as tmp:
            movie = Path(tmp)/'loop.mp4'
            cmd = [ffmpeg, '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{ow}x{oh}', '-r', str(self.fps), '-i', 'pipe:0', '-an',
                   '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p', '-c:v', 'libx264',
                   '-threads', '4', '-crf', '11', '-tune', 'grain', '-g', str(self.count), '-sc_threshold','0',
                   '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                   '-movflags', '+faststart', str(movie)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for i in range(self.count):
                    image = self.frame(i, layers)
                    if (ow, oh) != (w, h):
                        image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_AREA)
                    proc.stdin.write(memoryview(image))
                proc.stdin.close()
                if proc.wait():
                    raise RuntimeError('FFmpeg failed to encode the painting')
            finally:
                if proc.poll() is None:
                    proc.kill()
                # A buffered final write to a killed encoder can fail on Windows.
                # Keep the original drawing/encoding exception and reap the child.
                with suppress(BrokenPipeError):
                    proc.stdin.close()
                proc.wait()
            movie.replace(path)
        return path

"""The underdrawing: what the painter intends before any paint is laid.

A Sketch is a handful of fields over the canvas, all at "guide" resolution:

  target   the colour each place should end up (linear RGB)
  region   which named thing each place belongs to
  vx, vy   which way the brush should travel there
  rmin     the smallest brush that place deserves (units); small = fine work
  sheen    how much interference medium a place gets, and `film` its optical thickness

You fill them by drawing shapes, back to front, the way a picture is blocked out: each shape
can set the colour, the region, the brush direction and the fineness of the area it covers,
and a few modelling calls (tint, form_shadow, cast_shadow ...) turn flat shapes into form.

Coordinates are canvas units: the canvas is always 1000 units wide, x to the right, y down,
whatever its pixel size.  Colours are '#rrggbb' or (r, g, b) in 0-255.

Everything here takes plain lists, numbers and strings, so the same calls work from JSON.
From Python you can also reach into the arrays directly (sk.target, sk.region, sk.vx ...);
call sk.touch() afterwards so the painter notices.
"""
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import color as C
from .engine import value_noise
from .geom import Raster, catmull, ellipse_pts, resample, tangents


def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


class Sketch:
    def __init__(self, GW, GH, seed=11, base='#8a8580'):
        self.GW, self.GH = int(GW), int(GH)
        self.s = self.GW / 1000.0                    # guide pixels per unit
        self.WU, self.HU = 1000.0, self.GH / self.s
        self.ra = Raster(self.GW, self.GH, self.s)
        self.X, self.Y = self.ra.X, self.ra.Y
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.target = np.empty((self.GH, self.GW, 3), np.float32)
        self.target[:] = C.as_lin(base)
        self.region = np.zeros((self.GH, self.GW), np.uint8)
        self.vx = np.ones((self.GH, self.GW), np.float32)
        self.vy = np.zeros((self.GH, self.GW), np.float32)
        self.rmin = np.full((self.GH, self.GW), 3.0, np.float32)
        self.sheen = np.zeros((self.GH, self.GW), np.float32)
        self.film = np.full((self.GH, self.GW), 1100.0, np.float32)
        self.regions = {'canvas': dict(id=0)}        # name -> properties (id, style, brush, length, black)
        self.masks = {}                              # named masks, for later reference
        self.geo = SimpleNamespace()                 # free space for your own geometry
        self.version = 0

    # ------------------------------------------------------------------ bookkeeping
    def touch(self):
        """Tell the painter the fields changed (call after editing the arrays by hand)."""
        self.version += 1

    def rid(self, name, **props):
        """Id of a named region, registering it (and updating its properties) if needed."""
        if isinstance(name, (int, np.integer)):
            return int(name)
        if name not in self.regions:
            if len(self.regions) >= 255:
                raise ValueError('too many regions (255)')
            self.regions[name] = dict(id=len(self.regions))
        self.regions[name].update({k: v for k, v in props.items() if v is not None})
        return self.regions[name]['id']

    def props(self, region):
        """Properties of a region given by name or id."""
        if isinstance(region, str):
            return self.regions[region]
        for p in self.regions.values():
            if p['id'] == int(region):
                return p
        return dict(id=int(region))

    def name_of(self, rid):
        for n, p in self.regions.items():
            if p['id'] == int(rid):
                return n
        return str(rid)

    def mask(self, ref):
        """Resolve a mask reference to a float array in [0,1].

        ref may be an array, the name of a stored mask, the name of a region, a region id,
        {'region': name}, or a list of any of these (their union)."""
        if isinstance(ref, np.ndarray):
            return ref.astype(np.float32)
        if isinstance(ref, (list, tuple)):
            m = np.zeros((self.GH, self.GW), np.float32)
            for r in ref:
                m = np.maximum(m, self.mask(r))
            return m
        if isinstance(ref, dict):
            ref = ref['region']
            return (self.region == self.regions[ref]['id']).astype(np.float32)
        if isinstance(ref, (int, np.integer)):
            return (self.region == int(ref)).astype(np.float32)
        if ref in self.masks:
            return self.masks[ref]
        if ref in self.regions:
            return (self.region == self.regions[ref]['id']).astype(np.float32)
        raise KeyError('no mask or region named %r' % (ref,))

    # ------------------------------------------------------------------ colour and direction specs
    def color(self, spec):
        """A colour spec -> linear (3,) or a (GH, GW, 3) field.

        '#rrggbb' | (r, g, b) | {'linear': [[x0, y0], [x1, y1]], 'colors': [c0, c1, ...]}
        | {'radial': [cx, cy, r] or [cx, cy, rx, ry], 'colors': [inner, ..., outer]}
        | callable(X, Y) -> array.  Gradients are interpolated perceptually (OKLab)."""
        if isinstance(spec, dict):
            cols = np.stack([C.lin_to_oklab(C.as_lin(c)) for c in spec['colors']])
            if 'linear' in spec:
                (x0, y0), (x1, y1) = spec['linear']
                dx, dy = x1 - x0, y1 - y0
                t = ((self.X - x0) * dx + (self.Y - y0) * dy) / (dx * dx + dy * dy + 1e-9)
            else:
                v = spec['radial']
                rx = v[2]
                ry = v[3] if len(v) > 3 else rx
                t = np.sqrt(((self.X - v[0]) / rx) ** 2 + ((self.Y - v[1]) / ry) ** 2)
            t = np.clip(t, 0, 1)
            n = len(cols) - 1
            seg = np.minimum((t * n).astype(np.int32), n - 1)
            f = (t * n - seg)[..., None]
            lab = cols[seg] + (cols[seg + 1] - cols[seg]) * f
            return np.clip(C.oklab_to_lin(lab), 0, 1).astype(np.float32)
        if callable(spec):
            return np.asarray(spec(self.X, self.Y), np.float32)
        return C.as_lin(spec)

    def _direction(self, spec, m):
        """A direction spec -> (fx, fy), scalars or fields.

        number: angle in degrees (0 = left to right, 90 = top to bottom)
        {'angle': a, 'spread': degrees, 'scale': units}   an angle that wanders
        {'around': [x, y]}    circling a point           {'radial': [x, y]}  away from a point
        {'swirl': [x, y], 'pitch': degrees}               circling and spiralling outward
        {'along': [[x, y], ...]}                          parallel to the nearest part of a curve
        {'contour': true}                                 following the shape's own outline
        callable(X, Y) -> (vx, vy), or a pair of arrays."""
        X, Y = self.X, self.Y
        if isinstance(spec, (int, float)):
            a = math.radians(spec)
            return math.cos(a), math.sin(a)
        if callable(spec):
            fx, fy = spec(X, Y)
            return np.asarray(fx, np.float32), np.asarray(fy, np.float32)
        if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[0], np.ndarray):
            return spec[0].astype(np.float32), spec[1].astype(np.float32)
        if not isinstance(spec, dict):
            raise ValueError('bad direction spec %r' % (spec,))
        if 'angle' in spec:
            n = value_noise(self.GW, self.GH, spec.get('scale', 160) * self.s, self.rng, octaves=2)
            a = math.radians(spec['angle']) + (n - 0.5) * 2 * math.radians(spec.get('spread', 25))
            return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
        for key in ('around', 'radial', 'swirl'):
            if key in spec:
                cx, cy = spec[key]
                dx, dy = X - cx, Y - cy
                n = np.sqrt(dx * dx + dy * dy) + 1e-6
                if key == 'radial':
                    return dx / n, dy / n
                tx, ty = -dy / n, dx / n
                if key == 'swirl':
                    p = math.radians(spec.get('pitch', 20))
                    return tx * math.cos(p) + dx / n * math.sin(p), ty * math.cos(p) + dy / n * math.sin(p)
                return tx, ty
        if 'along' in spec:
            pl = np.asarray(spec['along'], np.float64)
            if spec.get('smooth', True) and len(pl) >= 3:
                pl = catmull(pl, n=10)
            pl = resample(pl, step=2.0)
            tn = tangents(pl)
            sel = m > 0.5
            fx = np.ones((self.GH, self.GW), np.float32)
            fy = np.zeros((self.GH, self.GW), np.float32)
            if sel.any():
                _, k = cKDTree(pl).query(np.stack([X[sel], Y[sel]], 1), workers=-1)
                fx[sel] = tn[k, 0]
                fy[sel] = tn[k, 1]
            return fx, fy
        if 'contour' in spec:
            area = max(float((m > 0.5).sum()) / self.s ** 2, 1.0)
            g = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.12 * math.sqrt(area) * self.s))
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            n = np.sqrt(gx * gx + gy * gy) + 1e-9
            return -gy / n, gx / n
        raise ValueError('bad direction spec %r' % (spec,))

    def _weight(self, m, blob=None, ramp=None, noise=None):
        w = m
        if blob is not None:
            cx, cy, rx = blob[0], blob[1], blob[2]
            ry = blob[3] if len(blob) > 3 else rx
            rot = math.radians(blob[4]) if len(blob) > 4 else 0.0
            dx, dy = self.X - cx, self.Y - cy
            u = dx * math.cos(rot) + dy * math.sin(rot)
            v = -dx * math.sin(rot) + dy * math.cos(rot)
            w = w * np.exp(-((u / rx) ** 2 + (v / ry) ** 2))
        if ramp is not None:
            (x0, y0), (x1, y1) = ramp
            dx, dy = x1 - x0, y1 - y0
            w = w * _sstep(0, 1, ((self.X - x0) * dx + (self.Y - y0) * dy) / (dx * dx + dy * dy + 1e-9))
        if noise is not None:
            w = w * _sstep(0.35, 0.7, value_noise(self.GW, self.GH, noise * self.s, self.rng, octaves=2))
        return w

    # ------------------------------------------------------------------ shapes
    def _apply(self, m, color=None, region=None, direction=None, detail=None, soft=0.0, opacity=1.0,
               name=None, **props):
        hard = m > 0.5
        if color is not None:
            a = (self.ra.blur(m, soft) if soft > 0 else m) * opacity
            col = self.color(color)
            self.target *= (1.0 - a)[..., None]
            self.target += a[..., None] * col
        if region is not None:
            self.region[hard] = self.rid(region, **props)
        if direction is not None:
            self.direction(m, direction)
        if detail is not None:
            self.rmin[hard] = float(detail)
        if name:
            self.masks[name] = m
        self.version += 1
        return m

    def background(self, color=None, region='background', direction=None, detail=None, **props):
        """Cover the whole canvas: the first thing to lay in."""
        m = np.ones((self.GH, self.GW), np.float32)
        return self._apply(m, color, region, direction, detail, **props)

    def polygon(self, points, **kw):
        """A straight-edged shape through the points.  kw: color, region, direction, detail,
        soft (edge blur, units), opacity, name (store the mask), plus region properties
        (style, brush, length, vary, black)."""
        return self._apply(self.ra.poly(np.asarray(points, np.float64)), **kw)

    def spline(self, points, closed=True, **kw):
        """A smooth shape through the points (centripetal Catmull-Rom)."""
        return self._apply(self.ra.poly(catmull(np.asarray(points, np.float64), closed=closed, n=12)), **kw)

    def ellipse(self, center, rx, ry=None, rot=0.0, **kw):
        ry = rx if ry is None else ry
        return self._apply(self.ra.poly(ellipse_pts(center[0], center[1], rx, ry, math.radians(rot), n=96)), **kw)

    def rect(self, x, y, w, h, **kw):
        return self._apply(self.ra.poly(np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], np.float64)), **kw)

    def ribbon(self, path, width, smooth=True, **kw):
        """A band along a path.  width: a number, or (start, end), or one value per point."""
        pl = np.asarray(path, np.float64)
        if smooth and len(pl) >= 3:
            pl = catmull(pl, n=10)
        pl = resample(pl, step=2.0)
        w = np.asarray(width, np.float64)
        if w.ndim == 1 and len(w) == 2:
            w = np.linspace(w[0], w[1], len(pl))
        elif w.ndim == 1:
            w = np.interp(np.linspace(0, 1, len(pl)), np.linspace(0, 1, len(w)), w)
        return self._apply(self.ra.stroke(pl, w), **kw)

    # ------------------------------------------------------------------ modelling
    def tint(self, mask, color, amount=1.0, blob=None, ramp=None, noise=None, soft=0.0):
        """Mix the intended colour toward `color` inside a mask.

        blob=(cx, cy, rx, ry[, rot]) limits it to a soft spot (a highlight, a blush);
        ramp=((x0, y0), (x1, y1)) fades it in from the first point to the second;
        noise=scale breaks it into patches."""
        m = self.mask(mask)
        w = self._weight(self.ra.blur(m, soft) if soft > 0 else m, blob, ramp, noise)
        k = np.clip(w * amount, 0, 1)[..., None]
        self.target += (self.color(color) - self.target) * k
        self.version += 1

    def _side(self, m, light, width, toward_light):
        L = np.asarray(light, np.float64)
        L = L / (np.linalg.norm(L) + 1e-9)
        d = L * width * (-1.0 if toward_light else 1.0)
        return m * (1.0 - self.ra.shift(m, d[0], d[1]))

    def form_shadow(self, mask, color, light=(-1, -0.5), width=10.0, amount=0.6, blur=None):
        """Shade the side of a shape that faces away from the light.  `light` points from the
        shape toward the light (default: upper left)."""
        m = self.mask(mask)
        w = self.ra.blur(self._side(m, light, width, False), width * 0.35 if blur is None else blur) * m
        self.target += (self.color(color) - self.target) * np.clip(w * amount, 0, 1)[..., None]
        self.version += 1

    def form_light(self, mask, color, light=(-1, -0.5), width=8.0, amount=0.6, blur=None):
        """Lighten the side of a shape that faces the light."""
        m = self.mask(mask)
        w = self.ra.blur(self._side(m, light, width, True), width * 0.35 if blur is None else blur) * m
        self.target += (self.color(color) - self.target) * np.clip(w * amount, 0, 1)[..., None]
        self.version += 1

    def cast_shadow(self, caster, onto, color, offset=(6, 10), blur=2.0, amount=0.75):
        """The shadow one shape throws on others: the caster's outline, displaced by `offset`."""
        mc = self.mask(caster)
        w = self.ra.blur(self.ra.shift(mc, offset[0], offset[1]), blur) * self.mask(onto) * (1.0 - mc)
        self.target += (self.color(color) - self.target) * np.clip(w * amount, 0, 1)[..., None]
        self.version += 1

    def line(self, path, width, color, amount=1.0, on=None, smooth=True):
        """A drawn line in the underdrawing (the fine brushes will find it when refining)."""
        pl = np.asarray(path, np.float64)
        if smooth and len(pl) >= 3:
            pl = catmull(pl, n=10)
        m = self.ra.stroke(resample(pl, step=1.0), width)
        if on is not None:
            m = m * self.mask(on)
        self.target += (self.color(color) - self.target) * np.clip(m * amount, 0, 1)[..., None]
        self.version += 1
        return m

    # ------------------------------------------------------------------ fields
    def direction(self, mask, spec):
        """Set which way the brush travels inside a mask (see _direction for the specs)."""
        m = self.mask(mask)
        fx, fy = self._direction(spec, m)
        sel = m > 0.5
        if np.ndim(fx) == 0:
            self.vx[sel] = fx
            self.vy[sel] = fy
        else:
            n = np.sqrt(fx * fx + fy * fy) + 1e-9
            self.vx[sel] = (fx / n)[sel]
            self.vy[sel] = (fy / n)[sel]
        self.version += 1

    def detail(self, mask, value):
        """Set the smallest brush (units) a place deserves: about 0.75 for eyes and lettering,
        1.2-1.8 for faces and small objects, 3 for cloth, 4-6 for backgrounds."""
        self.rmin[self.mask(mask) > 0.5] = float(value)
        self.version += 1

    def sheen_field(self, mask, amount=0.6, hue=None, film=None, noise=None):
        """Where interference medium should go (0..1) and its colour: `hue` is a name from
        brushwork.SHEEN_HUES ('magenta', 'blue', 'teal', 'gold' ...) or `film` an optical
        thickness in nm.  noise=scale breaks the amount into streaks."""
        m = self.mask(mask)
        sel = m > 0.5
        a = self._weight(m, noise=noise) * amount if noise else m * amount
        self.sheen[sel] = np.clip(a, 0, 1)[sel]
        if hue is not None or film is not None:
            self.film[sel] = C.film_for(hue) if film is None else float(film)
        self.version += 1

    def region_props(self, region, **props):
        """How a region likes to be painted: style ('body', 'smooth', 'impasto', 'hair' ...),
        brush (largest brush radius, units), length ((min, max) stroke length, units),
        vary (0 flat and well mixed .. 0.4 default .. 1 broken colour),
        black=True (lay it in chromatic black with whatever sheen the sketch asks for)."""
        self.rid(region, **props)
        self.version += 1

    # ------------------------------------------------------------------ looking at it
    def preview(self):
        """The intended colours as an 8-bit sRGB image."""
        return (C.lin_to_srgb(np.clip(self.target, 0, 1)) * 255 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        meta = dict(GW=self.GW, GH=self.GH, seed=self.seed, version=self.version, regions=self.regions,
                    rng=self.rng.bit_generator.state)
        arrays = dict(target=self.target, region=self.region, vx=self.vx, vy=self.vy, rmin=self.rmin,
                      sheen=self.sheen, film=self.film)
        for k, v in self.masks.items():
            arrays['mask:' + k] = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        sk = cls(meta['GW'], meta['GH'], seed=meta['seed'])
        for k in ('target', 'region', 'vx', 'vy', 'rmin', 'sheen', 'film'):
            setattr(sk, k, z[k])
        for k in z.files:
            if k.startswith('mask:'):
                sk.masks[k[5:]] = z[k].astype(np.float32) / 255.0
        sk.regions = meta['regions']
        sk.version = meta['version']
        sk.rng.bit_generator.state = meta['rng']
        return sk

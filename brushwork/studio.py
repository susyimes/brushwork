"""The studio: a canvas, a palette, an underdrawing, and the passes a painter makes.

Works the way an oil painter does: toned ground, block-in from big brushes to small, cutting
one shape in against another, stepping back to compare canvas with intention and correcting
where it is furthest off, then explicit strokes for the drawing and the thick lights.

Sizes and positions are canvas units (the canvas is 1000 units wide); the same script paints
a 900-pixel study or a 3200-pixel picture.  Nothing here knows what the picture is: that is
the Sketch (sketch.py) plus whatever strokes you add yourself.
"""
import json
import math
import os
import time

import cv2
import numpy as np

from . import color as C
from .engine import (_COLS, DEFAULTS, NPAR, P_FILM, P_IRI, Canvas, Strokes, make_params,
                     trace_paths, value_noise)
from .geom import arclen, catmull, ellipse_pts, normals, resample, tangents
from .sketch import Sketch

# Brush "styles": how the paint is loaded and how the brush is handled.  Any of these knobs
# can be overridden per call or per stroke.
#   opacity  hiding power of the stroke          thick   impasto height
#   dry      how soon the stroke breaks up on the canvas tooth (dry-brush)
#   pickup   how much wet paint the bristles lift and drag along (wet-into-wet)
#   unmix    how unevenly the colour is mixed on the brush (streaks)
#   furrow / ridge / blob   bristle furrows, squeezed-out edges, the lump where the brush lifts
#   cap / taper / ragged / wobble / skip   shape of the ends and steadiness of the hand
STYLE = dict(
    wash=dict(opacity=.78, thick=.35, dry=.62, pickup=.0, unmix=.5, furrow=.5, ridge=.2, blob=.1,
              cap=.6, taper=.2, skip=.45, wobble=1.2, ragged=.9),
    impasto=dict(opacity=.94, thick=2.0, dry=.45, pickup=.035, unmix=.9, furrow=.65, ridge=.9, blob=.6,
                 cap=.8, taper=.35, skip=.35, wobble=1.0, ragged=.6),
    body=dict(opacity=.95, thick=1.3, dry=.38, pickup=.05, unmix=.35, furrow=.55, ridge=.6, blob=.4,
              cap=.85, taper=.35, skip=.25, wobble=.9, ragged=.5),
    smooth=dict(opacity=.93, thick=.42, dry=.20, pickup=.10, unmix=.15, furrow=.35, ridge=.3, blob=.2,
                cap=.9, taper=.4, skip=.12, wobble=.7, ragged=.35),
    hair=dict(opacity=.95, thick=1.25, dry=.36, pickup=.04, unmix=.3, furrow=.6, ridge=.7, blob=.35,
              cap=.9, taper=.6, skip=.25, wobble=.8, ragged=.5),
    line=dict(opacity=1., thick=.9, dry=.12, pickup=.0, unmix=.12, furrow=.3, ridge=.35, blob=.15,
              cap=1., taper=.75, skip=.08, wobble=.3, ragged=.2),
    dab=dict(opacity=1., thick=3.0, dry=.0, pickup=.0, unmix=.1, furrow=.35, ridge=1.2, blob=1.2,
             cap=1., taper=.5, skip=.0, wobble=.3, ragged=.2),
    glaze=dict(opacity=.12, thick=.6, dry=.25, pickup=.0, unmix=.4, furrow=.7, ridge=.5, blob=.3,
               cap=.8, taper=.4, skip=.3, wobble=1., ragged=.6),
    blend=dict(opacity=.55, thick=.4, dry=.0, pickup=.25, unmix=0., furrow=.25, ridge=.1, blob=.0,
               cap=.9, taper=.4, skip=.0, wobble=.6, ragged=.3, mode=1.),
)

FAMILIES = ['violet', 'indigo', 'blue', 'teal', 'green', 'warm', 'red', 'wine']
COARSE = ((2.6, 0.050, (10, 44)), (1.6, 0.042, (7, 30)), (1.0, 0.036, (4, 18)))
FINE = ((1.6, 0.040, (6, 26)), (1.0, 0.032, (4, 16)), (0.75, 0.030, (3, 12)))
BROAD = ((6.0, 0.055, (24, 90)), (4.0, 0.050, (16, 64)))      # for regions too coarse for COARSE
PASSES = dict(broad=BROAD, coarse=COARSE, fine=FINE)


def _black_families():
    idx = np.zeros((len(FAMILIES), 4), np.int32)
    w = np.zeros((len(FAMILIES), 4), np.float32)
    for k, n in enumerate(FAMILIES):
        idx[k], w[k] = C.recipe(**C.CHROMATIC_BLACKS[n])
    return idx, w


def _imwrite(path, rgb, quality=92):
    """Save an RGB uint8 image; works with non-ASCII paths."""
    ext = os.path.splitext(path)[1].lower() or '.png'
    args = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(rgb[:, :, ::-1]), args)
    if not ok:
        raise IOError('could not encode ' + path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return path


class Studio:
    def __init__(self, width=1200, height=None, seed=1, sketch=None, guide_scale=None,
                 primer='#e2dbcc', log=print):
        """width, height: canvas size in pixels (height defaults to 5/4 of the width).
        sketch: a Sketch, or a function (GW, GH) -> sketch-like object; default an empty Sketch.
        The underdrawing lives at guide resolution: full size up to 1200 px wide, half above."""
        self.W = int(width)
        self.H = int(height) if height else self.W * 5 // 4
        self.S = self.W / 1000.0
        if guide_scale is None:
            guide_scale = 1.0 if self.W <= 1200 else 0.5
        self.guide_scale = guide_scale
        self.GW = int(round(self.W * guide_scale))
        self.GH = int(round(self.H * guide_scale))
        self.seed = seed
        self.primer = primer if isinstance(primer, str) else [int(v) for v in primer]
        self.log = log or (lambda *a: None)
        self.t0 = time.time()
        if sketch is None:
            sketch = Sketch(self.GW, self.GH)
        elif callable(sketch):
            sketch = sketch(self.GW, self.GH)
        self.sk = sketch
        self.gs = self.GW / self.W
        self.s = self.GW / 1000.0
        self.rng = np.random.default_rng(seed)
        self.book = C.RecipeBook()
        self.cv = Canvas(self.W, self.H, ground=C.as_lin(primer), pitch=max(3.0, 2.9 * self.S), seed=seed + 1)
        self.tk = self.S ** 0.5
        self._lab = None
        self._lab_ver = None
        self._blur = {}
        self.fam_idx, self.fam_w = _black_families()
        self.lut = C.thinfilm_lut()
        self.stages = []
        self.frame_cb = None
        self.thk0 = np.zeros((self.H, self.W), np.float32)
        self.view_dir = None
        self._looks = 0
        self.say('studio ready: %dx%d canvas, %dx%d underdrawing' % (self.W, self.H, self.GW, self.GH))

    @property
    def HU(self):
        """Canvas height in units (the width is always 1000)."""
        return 1000.0 * self.H / self.W

    @property
    def lab(self):
        v = getattr(self.sk, 'version', 0)
        if self._lab is None or self._lab_ver != v:
            self._lab = C.lin_to_oklab(self.sk.target)
            self._lab_ver = v
            self._blur = {}
        return self._lab

    # ------------------------------------------------------------------ utilities
    def say(self, msg):
        self.log('[%6.1fs] %s' % (time.time() - self.t0, msg))

    def gidx(self, pts_px):
        g = (np.asarray(pts_px) * self.gs).astype(np.int32)
        return np.clip(g[:, 1], 0, self.GH - 1), np.clip(g[:, 0], 0, self.GW - 1)

    def uidx(self, pts_units):
        g = (np.asarray(pts_units) * self.s).astype(np.int32)
        return np.clip(g[:, 1], 0, self.GH - 1), np.clip(g[:, 0], 0, self.GW - 1)

    def grid_seeds(self, mask, spacing_units, jitter=1.0, margin=0.0):
        sp = max(1.0, spacing_units * self.S)
        mg = margin * self.S
        nx = int((self.W + 2 * mg) / sp) + 2
        ny = int((self.H + 2 * mg) / sp) + 2
        gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
        pts = np.stack([gx.ravel(), gy.ravel()], 1) * sp - mg
        pts += (self.rng.random(pts.shape, dtype=np.float32) - 0.5) * sp * jitter
        ok = (pts[:, 0] >= -mg) & (pts[:, 1] >= -mg) & (pts[:, 0] < self.W - 1 + mg) & (pts[:, 1] < self.H - 1 + mg)
        pts = pts[ok]
        iy, ix = self.gidx(pts)
        return pts[mask[iy, ix]]

    def target_blur(self, key, mask, sigma_units):
        """The intended colours, blurred inside a mask only (so a big brush takes the average
        of its own region and nothing from the neighbours)."""
        sg = sigma_units * self.s
        if sg < 0.7:
            return self.sk.target
        k = (key, round(sg, 1), getattr(self.sk, 'version', 0), int(mask.sum()))
        if k not in self._blur:
            m = mask.astype(np.float32)
            num = cv2.GaussianBlur(self.sk.target * m[..., None], (0, 0), sg)
            den = cv2.GaussianBlur(m, (0, 0), sg)
            tb = num / np.maximum(den, 1e-4)[..., None]
            bad = den < 1e-3
            tb[bad] = self.sk.target[bad]
            if len(self._blur) > 6:
                self._blur.pop(next(iter(self._blur)))
            self._blur[k] = tb.astype(np.float32)
        return self._blur[k]

    def params(self, n, style, R_px, **over):
        d = dict(STYLE[style])
        d.update(over)
        d['thick'] = np.asarray(d['thick']) * self.tk
        return make_params(n, radius=R_px, **d)

    def black_recipes(self, film, calm=None, swap=0.12):
        """Chromatic-black recipe per stroke, its undertone following the local sheen hue."""
        n = len(film)
        V, I, B, T, G, Wm, Rd, Wn = range(8)
        base = np.select([film < 580, film < 640, film < 700, film < 780, film < 880, film < 960, film < 1070,
                          film < 1135, film < 1185, film < 1225],
                         [V, I, B, T, G, Wm, Wn, V, I, B], default=T)
        sw = self.rng.random(n) < swap
        base = np.where(sw, self.rng.choice([V, I, B, T, Wn, Rd], n), base)
        if calm is not None:
            base = np.where(calm, self.rng.choice([V, I, B], n), base)
        return self.fam_idx[base].copy(), self.fam_w[base].copy()

    def jit(self, col, dl=0.03, dc=0.012):
        lab = C.lin_to_oklab(np.asarray(col, np.float32))
        lab = lab + self.rng.normal(0, 1, 3) * np.array([dl, dc, dc])
        return np.clip(C.oklab_to_lin(lab), 0.0006, 1.0)

    def albedo_lab(self):
        a = cv2.resize(self.cv.col, (self.GW, self.GH), interpolation=cv2.INTER_AREA)
        return C.lin_to_oklab(a)

    def error_map(self, sigma_units=0.0):
        cur = self.albedo_lab()
        d = cur - self.lab
        e = np.sqrt((d * d * np.array([1.0, 1.6, 1.6], np.float32)).sum(-1))
        if sigma_units > 0:
            e = cv2.GaussianBlur(e, (0, 0), max(0.5, sigma_units * self.s))
        return e

    def contour_field(self, mask, sigma_units=2.5):
        """Direction field that runs along the edge of a shape (for cutting in)."""
        g = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma_units * self.s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        n = np.sqrt(gx * gx + gy * gy)
        weak = n < 1e-5
        fx = np.where(weak, self.sk.vx, -gy / (n + 1e-9)).astype(np.float32)
        fy = np.where(weak, self.sk.vy, gx / (n + 1e-9)).astype(np.float32)
        return fx, fy

    def stage(self, name):
        self.cv.dry(1.0)
        self.cv.update_tooth()
        self.stages.append((name, self.cv.n_strokes))
        self.say('%-28s strokes so far %8d' % (name, self.cv.n_strokes))
        if self.frame_cb:
            self.frame_cb(name)

    # ------------------------------------------------------------------ the two stroke layers everything else is built on
    def fill(self, key, mask, R, length, cover=1.5, style='body', blur=0.6, jit=(0.012, 0.010),
             use_region=True, edge_tol=0.10, recipe_fn=None, weights=None, thresh=None, load=1.6,
             inertia=0.6, rvar=(0.8, 1.15), field=None, iri_fn=None, spacing=None, margin=0.0,
             overshoot=0, ties=True, region_map=None, **over):
        """A layer of strokes that follow the direction field, coloured from the underdrawing."""
        S = self.S
        Lbar = 0.5 * (length[0] + length[1]) * 0.7
        if spacing is None:
            spacing = math.sqrt(2 * R * Lbar / cover)
        pts = self.grid_seeds(mask, spacing, margin=margin)
        if weights is not None and len(pts):
            iy, ix = self.gidx(pts)
            pts = pts[weights[iy, ix] > thresh]
        N = len(pts)
        if N == 0:
            return 0
        iy, ix = self.gidx(pts)
        if recipe_fn is None:
            tb = self.target_blur(key, mask, R * blur)
            lab = C.lin_to_oklab(tb[iy, ix])
            lab += self.rng.normal(0, 1, (N, 3)).astype(np.float32) * np.array([jit[0], jit[1], jit[1]], np.float32)
            col = np.clip(C.oklab_to_lin(lab), 0.0006, 1.0)
            idx, w, _ = self.book.lookup(col, self.rng if ties else None)
        else:
            idx, w = recipe_fn(pts, iy, ix)
        Rpx = (R * S * self.rng.uniform(rvar[0], rvar[1], N)).astype(np.float32)
        Lpx = self.rng.uniform(length[0], length[1], N) * S
        step = np.maximum(1.5, Rpx * 0.45).astype(np.float32)
        nst = np.maximum(2, Lpx / step).astype(np.int32)
        nb = (nst * self.rng.random(N)).astype(np.int32)
        nf = (nst - nb).astype(np.int32)
        ov = np.full(N, overshoot, np.int32)
        M = int(nst.max()) + 4
        fvx, fvy = (self.sk.vx, self.sk.vy) if field is None else field
        sp = self.params(N, style, Rpx, loadlen=Lpx * load, **over)
        if iri_fn is not None:
            a, f = iri_fn(iy, ix)
            sp[:, P_IRI] = a
            sp[:, P_FILM] = f
        st = Strokes()
        CH = 60000
        for c0 in range(0, N, CH):
            c1 = min(N, c0 + CH)
            out = np.zeros((c1 - c0, M, 2), np.float32)
            nout = np.zeros(c1 - c0, np.int32)
            trace_paths(np.ascontiguousarray(pts[c0:c1]), fvx, fvy, self.gs, nf[c0:c1], nb[c0:c1],
                        step[c0:c1], inertia, self.sk.region if region_map is None else region_map,
                        self.lab, edge_tol, use_region,
                        float(margin * S), ov[c0:c1], out, nout)
            short = np.flatnonzero(nout < 2)          # nowhere to go: still leave a touch of paint
            if len(short):
                d = np.stack([fvx[iy[c0:c1][short], ix[c0:c1][short]], fvy[iy[c0:c1][short], ix[c0:c1][short]]], 1)
                h = (Rpx[c0:c1][short] * 0.5)[:, None]
                out[short, 0] = pts[c0:c1][short] - d * h
                out[short, 1] = pts[c0:c1][short] + d * h
                nout[short] = 2
            st.add(out, nout, sp[c0:c1], idx[c0:c1], w[c0:c1])
        self.cv.paint(st)
        return N

    def lines(self, items, style='line', shuffle=False):
        """Explicit strokes.  item: dict(pl=(n,2) units, r=units, rgb=linear | rec=(idx,w), ...)."""
        if not items:
            return 0
        S = self.S
        pls, keys, rgbs, recs, which = [], [], [], [], []
        for it in items:
            pl = np.asarray(it['pl'], np.float64) * S
            if len(pl) < 2:
                continue
            L = arclen(pl)[-1]
            rpx = it['r'] * S
            if L < 0.6:
                d = np.array([max(0.8, rpx * 0.6), 0.0])
                pl = np.stack([pl[0] - d, pl[0] + d])
                L = 2 * d[0]
            pl = resample(pl, step=max(1.0, rpx * 0.4))
            pls.append(pl.astype(np.float32))
            st = dict(STYLE[it.get('style', style)])
            st.update({k: v for k, v in it.items() if k in _COLS})
            st['radius'] = rpx
            st['thick'] = st['thick'] * self.tk
            if 'loadlen' not in it:
                st['loadlen'] = L * it.get('load', 3.0)
            keys.append(st)
            if 'rec' in it:
                recs.append(it['rec'])
                which.append(-1)
            else:
                which.append(len(rgbs))
                rgbs.append(np.asarray(it['rgb'], np.float32))
        n = len(pls)
        if n == 0:
            return 0
        sp = np.empty((n, NPAR), np.float32)
        for k, v in DEFAULTS.items():
            sp[:, _COLS[k]] = v
        for i, st in enumerate(keys):
            for k, v in st.items():
                sp[i, _COLS[k]] = v
        idx = np.zeros((n, 4), np.int32)
        w = np.zeros((n, 4), np.float32)
        if rgbs:
            li, lw, _ = self.book.lookup(np.clip(np.stack(rgbs), 0.0006, 1.0), self.rng)
        ri = 0
        for i, wh in enumerate(which):
            if wh < 0:
                idx[i], w[i] = recs[ri]
                ri += 1
            else:
                idx[i], w[i] = li[wh], lw[wh]
        stx = Strokes()
        stx.add(pls, None, sp, idx, w)
        self.cv.paint(stx, shuffle=shuffle)
        return n

    def runs(self, pl, regions, bridge=2, min_len=0.0):
        """Split a polyline (units) into the runs that lie on the given regions.  Gaps of a
        point or two are bridged and runs shorter than min_len (units) are dropped, so a line
        that skims a boundary does not break into a string of beads."""
        pl = np.asarray(pl)
        iy, ix = self.uidx(pl)
        ok = np.isin(self.sk.region[iy, ix], regions)
        n = len(ok)
        i = 0
        while i < n:
            if not ok[i]:
                j = i
                while j < n and not ok[j]:
                    j += 1
                if 0 < i and j < n and j - i <= bridge:
                    ok[i:j] = True
                i = j
            else:
                i += 1
        out, start = [], None
        for i, o in enumerate(ok):
            if o and start is None:
                start = i
            if (not o or i == n - 1) and start is not None:
                end = i + 1 if o else i
                if end - start >= 2:
                    seg = pl[start:end]
                    if min_len <= 0 or arclen(seg)[-1] >= min_len:
                        out.append(seg)
                start = None
        return out

    def on_regions(self, items, regions):
        out = []
        for it in items:
            for run in self.runs(it['pl'], regions):
                d = dict(it)
                d['pl'] = run
                out.append(d)
        return out

    def film_rgb(self, film):
        return self.lut[int(np.clip(film / 1800.0 * 1023, 0, 1023))]

    def uncovered(self, mask, thr=0.22):
        """Where, inside mask, the toned ground is still showing (guide-res bool)."""
        added = cv2.resize(self.cv.thk - self.thk0, (self.GW, self.GH), interpolation=cv2.INTER_AREA)
        return mask & (added < thr * self.tk)

    def sheen_glaze(self, mask, key, sizes=((9, (70, 220), 0.9), (5, (50, 160), 0.8)), strength=1.0, thresh=0.34,
                    scumble=True):
        """Interference medium over the dry blacks: first dragged on as a glaze, then scumbled
        with a nearly dry brush so the colour breaks on the ridges of the strokes underneath."""
        sc = self.sk
        wts = sc.sheen + 0.25 * self.rng.random(sc.sheen.shape, dtype=np.float32)

        def rec_fn(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix])

        for k, (R, length, cover) in enumerate(sizes):
            def iri_fn(iy, ix):
                a = np.clip(sc.sheen[iy, ix] * self.rng.uniform(0.6, 1.25, len(iy)) * strength, 0, 1)
                f = sc.film[iy, ix] + self.rng.normal(0, 22, len(iy))
                gold = self.rng.random(len(iy)) < 0.06          # the odd touch of gold among the violets
                return a, np.where(gold, self.rng.uniform(860, 930, len(iy)), f)

            if k == 0 or not scumble:
                self.fill(key, mask, R, length, cover=cover, style='glaze', recipe_fn=rec_fn, edge_tol=9.0,
                          iri_fn=iri_fn, weights=wts, thresh=thresh, filmvar=30, load=0.9, filmload=0.12)
            else:
                self.fill(key, mask, R, length, cover=cover, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                          iri_fn=iri_fn, weights=wts, thresh=thresh + 0.06, filmvar=30, load=0.55, filmload=0.0,
                          dry=0.85, opacity=0.8, unmix=1.1, thick=1.5, skip=0.5)

    # ------------------------------------------------------------------ region references
    def rids(self, regions=None):
        """Region names / ids / a list of them -> list of ids.  None means every region."""
        sk = self.sk
        if regions is None:
            if hasattr(sk, 'regions'):
                return sorted(p['id'] for p in sk.regions.values())
            return [int(v) for v in np.unique(sk.region)]
        if isinstance(regions, (str, int, np.integer)):
            regions = [regions]
        return [int(r) if isinstance(r, (int, np.integer)) else sk.regions[r]['id'] for r in regions]

    def rmask(self, regions=None):
        """Bool mask (guide resolution) of the given regions.  Also takes a mask array, or the
        name of a mask stored in the sketch."""
        if isinstance(regions, np.ndarray):
            return regions if regions.dtype == bool else regions > 0.5
        if isinstance(regions, str) and regions not in getattr(self.sk, 'regions', {}) \
                and regions in getattr(self.sk, 'masks', {}):
            return self.sk.masks[regions] > 0.5
        return np.isin(self.sk.region, self.rids(regions))

    def _is_black(self, rid):
        return bool(hasattr(self.sk, 'props') and self.sk.props(rid).get('black'))

    def _edge_margin(self, m, lmax):
        """Strokes of a region that reaches the canvas edge must be able to start off-canvas."""
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            return float(min(0.5 * lmax, 200.0))
        return 0.0

    def _region_spec(self, rid):
        sk = self.sk
        m = sk.region == rid
        if not m.any():
            return None
        p = sk.props(rid) if hasattr(sk, 'props') else {}
        r0 = float(np.median(sk.rmin[m]))
        area = float(m.sum()) / self.s ** 2
        big = p.get('brush') or min(max(3.2 * r0, 1.2), 16.0, 0.12 * math.sqrt(area) + 0.6)
        ln = p.get('length') or (3.5 * big, 12.0 * big)
        style = p.get('style') or ('smooth' if r0 <= 1.3 else 'body')
        vary = float(p.get('vary', 0.4))
        return rid, float(big), (float(ln[0]), float(ln[1])), style, vary >= 0.75, vary

    # ------------------------------------------------------------------ the passes
    def tone_ground(self, color=None, pigments=None, brush=26, length=(160, 420),
                    passes=((0.9, 1.7, 0.92, 0.40), (-0.7, 1.6, 0.95, 0.25)), margin=200.0):
        """Tone the white primer: a thin colour scrubbed on in two crossing passes.

        Give `color` ('#rrggbb') or `pigments` ({'umber': .4, 'violet': .6}).  A mid-value
        ground that is roughly the average of the picture is the usual choice: whatever the
        later strokes leave uncovered will show this colour, not white."""
        rng = self.rng
        mask = np.ones((self.GH, self.GW), bool)
        n1 = value_noise(self.GW, self.GH, 260 * self.s, rng, octaves=2)
        if pigments is None and color is None:
            pigments = dict(umber=.5, ochre=.2, white=.3)
        if pigments is not None:
            rec = C.recipe(**pigments)
        else:
            li, lw, _ = self.book.lookup(C.as_lin(color)[None], None)
            rec = (li[0], lw[0])
        for base, cover, op, dry in passes:
            ang = base + (n1 - 0.5) * 1.6
            self.fill('all', mask, brush, length, cover=cover, style='wash', use_region=False, edge_tol=9.0,
                      recipe_fn=lambda p, iy, ix: (np.tile(rec[0], (len(p), 1)), np.tile(rec[1], (len(p), 1))),
                      field=(np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)), load=1.8,
                      opacity=op, dry=dry, margin=margin)
        self.thk0 = self.cv.thk.copy()
        self.stage('toned ground')

    def underpaint(self, regions=None, brush=3.6, length=(8, 26), cover=2.8, edge=0.22, dry=True):
        """A soft first coat over several regions taken together.  Its strokes run straight
        across the boundaries between them, so no ground is left showing in the seams; the
        block-in that follows sharpens the edges.  A stroke stops where the intended colour
        changes by more than `edge` (OKLab), so it crosses thin lines and soft transitions but
        does not drag one object's colour far into its neighbour."""
        fig = self.rmask(regions)
        merged = fig.astype(np.uint8)
        n = self.fill('figure', fig, brush, length, cover=cover, style='body', edge_tol=edge, blur=0.45,
                      region_map=merged, ties=False, jit=(0.004, 0.003), margin=8.0, thick=0.7, pickup=0.03)
        if dry:
            self.cv.dry(1.0)
        return n

    def block_in(self, regions=None, specs=None, margin='auto', label='block-in', dry=True):
        """Lay each region in with two sizes of brush, in the order given (back to front).

        regions: names or ids (default: every region, in the order the sketch introduced them).
        specs: instead of regions, a list of (region, brush, (min_len, max_len), style, ties)
        to choose the brushes yourself; `ties` lets the same colour be mixed from different
        tubes from stroke to stroke.  Otherwise each region uses its properties from the
        sketch (style, brush, length, vary): vary 0 lays flat, well-mixed colour, the default
        0.4 is quietly alive, 1 is broken colour in the impressionist way.  Regions marked
        black=True are laid in chromatic black with the sheen the sketch asks for."""
        sk = self.sk
        if specs is None:
            specs = [s for s in (self._region_spec(r) for r in self.rids(regions)) if s]
        total = 0
        for spec in specs:
            reg, r, ln, style, ties = spec[:5]
            rid = self.rids([reg])[0]
            m = sk.region == rid
            if not m.any():
                continue
            if self._is_black(rid):
                total += self._block_black(rid, m, r, ln)
                continue
            mg = self._edge_margin(m, ln[1]) if margin == 'auto' else float(margin)
            over = {}
            if len(spec) > 5:                          # liveliness asked for by the sketch
                vary = spec[5]
                jit = (0.004 + 0.010 * vary, 0.003 + 0.008 * vary)
                over['unmix'] = STYLE[style]['unmix'] * (0.3 + 0.9 * vary)
            else:
                jit = (0.012, 0.010) if ties else (0.005, 0.004)
            key = 'r%d' % rid
            total += self.fill(key, m, r, ln, cover=2.0, style=style, edge_tol=0.16, blur=0.55, overshoot=1,
                               ties=ties, jit=jit, margin=mg, **over)
            total += self.fill(key, m, r * 0.52, (ln[0] * 0.6, ln[1] * 0.6), cover=1.8, style=style, edge_tol=0.11,
                               blur=0.4, overshoot=1, ties=ties, jit=jit, margin=mg, **over)
        if dry:
            self.stage(label)
        return total

    def _block_black(self, rid, m, r, ln):
        """A region of chromatic black: long loaded strokes mixed from complementary darks,
        carrying whatever interference medium the sketch's sheen field asks for."""
        sk = self.sk
        mg = self._edge_margin(m, ln[1])

        def rec_fn(p, iy, ix):
            return self.black_recipes(sk.film[iy, ix])

        def iri_fn(k, pw):
            return lambda iy, ix: (np.clip(k * sk.sheen[iy, ix] ** pw * self.rng.uniform(0.7, 1.1, len(iy)), 0, 1),
                                   sk.film[iy, ix])

        key = 'k%d' % rid
        n = self.fill(key, m, r, ln, cover=2.3, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                      iri_fn=iri_fn(0.25, 1.5), unmix=0.6, filmvar=30, opacity=0.98, dry=0.3, overshoot=1, load=2.0,
                      filmload=0.1, margin=mg)
        n += self.fill(key, m, r * 0.58, (ln[0] * 0.62, ln[1] * 0.62), cover=1.3, style='impasto', recipe_fn=rec_fn,
                       edge_tol=9.0, iri_fn=iri_fn(0.5, 1.6), unmix=0.9, pickup=0.05, filmvar=40, opacity=0.96,
                       overshoot=1, filmload=0.1, margin=mg * 0.6)
        for R in (4.0, 2.0, 1.2):                      # close what the big brushes could not reach
            gap = self.uncovered(m)
            if gap.sum() > 30:
                gap = cv2.dilate(gap.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                n += self.fill(key, gap & m, R, (4 * R, 14 * R), cover=3.0, style='impasto', recipe_fn=rec_fn,
                               edge_tol=9.0, unmix=0.6, opacity=0.98, dry=0.15, overshoot=1, thick=1.4, margin=10.0)
        return n

    def cut_in(self, region, against=None, brushes=((3.2, 13.0), (1.7, 6.0), (1.0, 3.2)), dry=True):
        """Redraw an edge from one side: small strokes of `region`'s own colour laid along its
        boundary with `against` (default: everything else).  This is how a background is used
        to trim a silhouette after the figure has been brushed in loosely."""
        sk = self.sk
        rid = self.rids([region])[0]
        own = sk.region == rid
        other = (self.rmask(against) if against is not None else ~own) & ~own
        if not own.any() or not other.any():
            return 0
        dist = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3) / self.s
        fld = self.contour_field(other, 2.5)
        black = self._is_black(rid)
        vary = float(sk.props(rid).get('vary', 0.4)) if hasattr(sk, 'props') else 0.4
        style = (sk.props(rid).get('style') if hasattr(sk, 'props') else None) or 'body'
        total = 0
        for k, (R, d) in enumerate(brushes):
            m = own & (dist > 0.75 * R) & (dist < d)
            cover = 2.5 if k == len(brushes) - 1 else 1.3 + 0.6 * k / max(1, len(brushes) - 1)
            if black:
                total += self.fill('cut%d' % rid, m, R, (12, 46), cover=2.5, style='impasto', edge_tol=9.0,
                                   recipe_fn=lambda p, iy, ix: self.black_recipes(sk.film[iy, ix]), field=fld,
                                   unmix=0.6, thick=1.3, opacity=0.98, dry=0.18, inertia=0.5)
            else:
                total += self.fill('cut%d' % rid, m, R, (10, 40), cover=cover, style=style, edge_tol=0.10, blur=0.3,
                                   field=fld, opacity=0.98, dry=0.15, inertia=0.5, ties=vary >= 0.75,
                                   jit=(0.004 + 0.010 * vary, 0.003 + 0.008 * vary),
                                   unmix=STYLE[style]['unmix'] * (0.3 + 0.9 * vary))
        if dry:
            self.stage('cutting in')
        return total

    def mend(self, regions=None, brush=1.1, max_n=60000):
        """Look over the given regions at full resolution for pinholes where the ground still
        shows between strokes, and touch each one in with a small dab of the right colour."""
        sk = self.sk
        regs = self.rids(regions)
        reg_full = cv2.resize(sk.region, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        hole = ((self.cv.thk - self.thk0) < 0.10 * self.tk) & np.isin(reg_full, regs)
        ys, xs = np.nonzero(hole)
        if len(ys) == 0:
            return 0
        sel = self.rng.permutation(len(ys))[:max_n]
        pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32) + 0.5
        iy, ix = self.gidx(pts)
        idx, w, _ = self.book.lookup(np.clip(sk.target[iy, ix], 0.0006, 1.0), None)
        d = np.stack([sk.vx[iy, ix], sk.vy[iy, ix]], 1)
        Rpx = np.full(len(pts), brush * self.S, np.float32)
        out = np.zeros((len(pts), 2, 2), np.float32)
        out[:, 0] = pts - d * Rpx[:, None] * 0.6
        out[:, 1] = pts + d * Rpx[:, None] * 0.6
        sp = self.params(len(pts), 'smooth', Rpx, loadlen=1e6, dry=0.0, opacity=0.97, thick=0.5, skip=0.0)
        st = Strokes()
        st.add(out, np.full(len(pts), 2), sp, idx, w)
        self.cv.paint(st)
        self.say('   mended %d pinholes' % len(pts))
        return len(pts)

    def refine(self, regions=None, passes=COARSE, label='refinement', inset=0.7, dry=True):
        """Step back, compare the canvas with the sketch, and correct the places that are
        furthest off, with a smaller brush each pass.

        passes: (brush, threshold, (min_len, max_len)) each; a place is repainted when its
        colour error (OKLab distance) exceeds the threshold, and only where the sketch allows
        a brush that small (rmin).  passes may also be 'broad', 'coarse' or 'fine': broad
        reaches the big-brush regions (walls, skies), coarse then fine finish everything the
        sketch marks as detailed.  A stroke is only started at least inset x brush away from
        the border of its region, so a broad brush does not slop over into the neighbour; the
        finer passes then work up to the edge."""
        sk = self.sk
        if isinstance(passes, str):
            passes = PASSES[passes]
        fig = np.isin(sk.region, self.rids(regions))
        if inset > 0:
            r = sk.region
            edge = np.zeros(r.shape, bool)
            edge[:, 1:] |= r[:, 1:] != r[:, :-1]
            edge[1:, :] |= r[1:, :] != r[:-1, :]
            from_edge = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3) / self.s
        total = 0
        for R, th, ln in passes:
            err = self.error_map(R * 0.35)
            m = fig & (sk.rmin <= R * 1.05)
            if inset > 0:
                m &= from_edge >= inset * R
            n = self.fill('fig', m, R, ln, cover=2.2, style='smooth', edge_tol=0.07, blur=0.25, weights=err,
                          thresh=th, jit=(0.004, 0.003), pickup=0.05, thick=0.6, ties=False)
            self.cv.dry(0.6)
            self.say('   refine R=%.2f -> %d strokes' % (R, n))
            total += n
        if dry:
            self.stage(label)
        return total

    def soften(self, regions, brush=3.0, length=(14, 46), cover=1.3, rewet=0.6, dry=False):
        """Drag a clean brush over a region to lose hard edges inside it.  A clean brush only
        moves wet paint, so the region is first brought back to `rewet` wetness (as if oiled
        out); pass rewet=0 to work only on what is still wet."""
        m = self.rmask(regions)
        if rewet > 0:
            full = cv2.resize(m.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
            self.cv.wet[full] = np.maximum(self.cv.wet[full], rewet)
        n = self.fill('soften', m, brush, length, cover=cover, style='blend', edge_tol=0.05, use_region=True)
        if dry:
            self.stage('softening')
        return n

    def strokes(self, items, on=None, shuffle=False, dry=False):
        """Explicit strokes: the drawing, the accents, anything you want placed exactly.

        Each item is a dict:
          path     [[x, y], ...] in units (two points make a dab or a straight stroke)
          brush    radius in units (default 1.0)
          color    '#rrggbb' or (r, g, b)   | pigments {'alizarin': .5, 'white': .5}
                                            | black 'violet' (a chromatic black family)
          style    'line' (default), 'dab', 'hair', 'body', 'impasto', 'smooth', 'glaze', 'wash'
          smooth   true to round the path into a curve through its points
          sheen    0..1 interference medium, with hue 'magenta' (see SHEEN_HUES) or film nm
          plus any style knob to override: opacity, thick, taper, dry, pickup, unmix ...
        on: restrict every stroke to these regions (it is cut where it leaves them)."""
        norm = []
        for it in items:
            d = {k: v for k, v in it.items() if k in _COLS}
            pl = np.asarray(it['path'], np.float64)
            if it.get('smooth') and len(pl) >= 3:
                pl = catmull(pl, n=10)
            d['pl'] = pl
            d['r'] = float(it.get('brush', 1.0))
            d['style'] = it.get('style', 'line')
            if 'load' in it:
                d['load'] = it['load']
            if 'pigments' in it:
                d['rec'] = C.recipe(**it['pigments'])
            elif 'black' in it:
                d['rec'] = C.recipe(**C.CHROMATIC_BLACKS[it['black']])
            else:
                d['rgb'] = C.as_lin(it.get('color', '#202020'))
            if 'sheen' in it:
                d['iri'] = float(it['sheen'])
                d['film'] = float(it['film']) if 'film' in it else C.film_for(it.get('hue', 'purple'))
            norm.append(d)
        if on is not None:
            norm = self.on_regions(norm, self.rids(on))
        n = self.lines(norm, shuffle=shuffle)
        if dry:
            self.stage('strokes')
        return n

    def dabs(self, n, regions=None, brush=(0.4, 1.6), colors=('#ffffff',), stretch=1.2, angle=None, style='dab',
             seed=None, **over):
        """Scatter n short touches over the given regions (or a named mask): stars, petals,
        glints on water, texture.  brush is a (min, max) radius, small ones far more common;
        stretch is the length of each touch in brush radii; angle (degrees) lays them all one
        way, give or take a little, instead of every which way."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        ys, xs = np.nonzero(self.rmask(regions))
        if len(ys) == 0:
            return 0
        pick = rng.integers(0, len(ys), int(n))
        items = []
        for k in pick:
            p = np.array([(xs[k] + rng.random()) / self.s, (ys[k] + rng.random()) / self.s])
            r = brush[0] + (brush[1] - brush[0]) * rng.random() ** 3
            a = rng.uniform(0, math.pi) if angle is None else math.radians(angle + rng.uniform(-8, 8))
            d = np.array([math.cos(a), math.sin(a)]) * r * stretch * 0.5
            items.append(dict(path=[p - d, p + d], brush=r, color=colors[rng.integers(len(colors))], style=style,
                              **over))
        return self.strokes(items, shuffle=True)

    def glaze(self, regions, color, opacity=0.25, brush=6.0, length=None, cover=1.3, dry=True):
        """A transparent film of colour over what is already there (it multiplies, so it can
        only darken and tint): deepen a shadow, warm a passage, unify an area."""
        m = self.rmask(regions)
        li, lw, _ = self.book.lookup(C.as_lin(color)[None], None)
        n = self.fill('glaze', m, brush, length or (5 * brush, 16 * brush), cover=cover, style='glaze', edge_tol=9.0,
                      recipe_fn=lambda p, iy, ix: (np.tile(li[0], (len(p), 1)), np.tile(lw[0], (len(p), 1))),
                      opacity=opacity, mult=1.0, thick=0.25, dry=0.1, pickup=0.0, unmix=0.0)
        if dry:
            self.stage('glaze')
        return n

    def sheen(self, regions, strength=1.0, thresh=0.34, sizes=((9, (70, 220), 0.9), (5, (50, 160), 0.8)),
              scumble=True, dry=True):
        """Interference medium over dry paint, following the sketch's sheen field (set it with
        sketch.sheen_field): the colour lives only on the lit ridges of the strokes."""
        self.sheen_glaze(self.rmask(regions), 'sheen', sizes=sizes, strength=strength, thresh=thresh, scumble=scumble)
        if dry:
            self.stage('sheen')

    def dry(self, amount=1.0):
        """Let the paint dry (fully by default).  Wet paint gets dragged and mixed by later
        strokes; dry paint is simply covered."""
        self.cv.dry(amount)
        self.cv.update_tooth()

    def paint(self, ground=None, rounds=2):
        """The whole routine for a finished underdrawing: tone the ground, underpaint, block in
        back to front, mend pinholes, then refine against the sketch.  Add your own strokes
        afterwards for the drawing and the accents."""
        self.tone_ground(color=ground)
        normal = [r for r in self.rids() if not self._is_black(r)]
        self.underpaint(normal)
        self.block_in()
        self.mend()
        self.refine(normal, passes=BROAD, label='broad correction')
        self.refine(normal, passes=COARSE)
        if rounds > 1:
            self.refine(normal, passes=FINE, label='fine refinement')
        self.mend()

    # ------------------------------------------------------------------ looking
    def look(self, path=None, width=1200, crop=None, zoom=1.0, lit=True, what='canvas', max_side=2400):
        """Save a view to look at, and return its path.

        crop=(x, y, w, h) in units picks a detail, shown at painted resolution times `zoom`;
        without crop the whole canvas is scaled to `width` pixels.
        what: 'canvas', 'sketch' (the underdrawing), 'both' (side by side), or 'error'
        (bright where the canvas departs from the sketch).  lit=False shows the bare paint
        colour without the relief lighting."""
        views = []
        if what in ('canvas', 'both'):
            views.append(self.cv.render(lit=lit))
        if what in ('sketch', 'both'):
            sk8 = (C.lin_to_srgb(np.clip(self.sk.target, 0, 1)) * 255 + 0.5).astype(np.uint8)
            views.append(cv2.resize(sk8, (self.W, self.H), interpolation=cv2.INTER_LINEAR))
        if what == 'error':
            e = np.clip(self.error_map(1.0) / 0.25, 0, 1)
            heat = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
            views.append(cv2.resize(heat, (self.W, self.H), interpolation=cv2.INTER_LINEAR))
        if not views:
            raise ValueError("what must be 'canvas', 'sketch', 'both' or 'error'")
        out = []
        for im in views:
            if crop is not None:
                x, y, w, h = [float(v) * self.S for v in crop]
                x0, y0 = max(0, int(x)), max(0, int(y))
                im = im[y0:min(self.H, int(y + h)), x0:min(self.W, int(x + w))]
                f = float(zoom)
            else:
                f = float(width) / self.W
            f = min(f, max_side / max(im.shape[0], im.shape[1]))
            if abs(f - 1.0) > 1e-3:
                im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            out.append(im)
        img = out[0] if len(out) == 1 else np.concatenate(out, 1)
        if path is None:
            self._looks += 1
            path = os.path.join(self.view_dir or '.', 'look_%03d.jpg' % self._looks)
        return _imwrite(path, img)

    def save_image(self, path, lit=True):
        """Write the finished painting at full resolution (PNG or JPG by extension)."""
        return _imwrite(path, self.cv.render(lit=lit), quality=95)

    def stats(self):
        return dict(width=self.W, height=self.H, seed=self.seed, brushstrokes=int(self.cv.n_strokes),
                    bristle_trails=int(self.cv.n_trails), pixels_touched=int(self.cv.n_pix),
                    stages=[dict(name=n, strokes_after=int(k)) for n, k in self.stages])

    # ------------------------------------------------------------------ keeping a painting between sittings
    def save(self, d):
        """Write the whole state (canvas, underdrawing, random streams) to a directory."""
        os.makedirs(d, exist_ok=True)
        cv = self.cv
        np.savez(os.path.join(d, 'canvas.npz'), col=cv.col, sca=cv.sca, wet=cv.wet, hgt=cv.hgt, thk=cv.thk,
                 iri=cv.iri, ifd=cv.ifd, tooth=cv.tooth, thk0=self.thk0)
        self.sk.save(os.path.join(d, 'sketch.npz'))
        meta = dict(format=1, width=self.W, height=self.H, seed=self.seed, guide_scale=self.guide_scale,
                    primer=self.primer, n_strokes=int(cv.n_strokes), n_trails=int(cv.n_trails),
                    n_pix=int(cv.n_pix), stages=self.stages, looks=self._looks,
                    rng=self.rng.bit_generator.state, canvas_rng=cv.rng.bit_generator.state)
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, d, log=print):
        """Pick a saved painting up exactly where it was left."""
        with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        sk = Sketch.load(os.path.join(d, 'sketch.npz'))
        st = cls(meta['width'], meta['height'], seed=meta['seed'], sketch=sk, guide_scale=meta['guide_scale'],
                 primer=meta['primer'], log=log)
        z = np.load(os.path.join(d, 'canvas.npz'))
        cv = st.cv
        for k in ('col', 'sca', 'wet', 'hgt', 'thk', 'iri', 'ifd', 'tooth'):
            setattr(cv, k, z[k])
        st.thk0 = z['thk0']
        cv.stamp[:] = 0
        cv._stamp_base = 0
        cv.n_strokes, cv.n_trails, cv.n_pix = meta['n_strokes'], meta['n_trails'], meta['n_pix']
        st.stages = [tuple(x) for x in meta['stages']]
        st._looks = meta.get('looks', 0)
        st.rng.bit_generator.state = meta['rng']
        cv.rng.bit_generator.state = meta['canvas_rng']
        return st

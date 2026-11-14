"""五彩斑斓的黑 — Astra in Obsidian Gothic in front of an eclipsed moon.

    python paint.py                 # 3200 x 4000 canvas -> out/astra_iridescent_black.png
    python paint.py --width 1000    # quick study
    python paint.py --frames        # also save a strip of the painting in progress

The picture the toolkit was first built for, kept as its most involved example.  The
underdrawing is hand-written in scene.py; this file is the painting of it.  It uses the
studio's ready-made passes where they fit (tone_ground, underpaint, block_in, mend, refine,
soften, sheen_glaze) and drops down to `fill` and `lines` for everything particular to this
picture: the swirl of the night, the satin folds, the hair strand by strand, the eyes.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2
import numpy as np

import scene as SC
from brushwork import color as C
from brushwork.geom import catmull, ellipse_pts, normals, resample, tangents
from brushwork.studio import FINE, Studio, _imwrite

# the nine official Astra iris colours
IRIS_NINE = ['#8a8f9c', '#589772', '#4a9dd3', '#d478b8', '#dc7953', '#9169d3', '#2db1ae', '#cca73a', '#c9495d']

LIGHT2D = np.array([-0.894, -0.447])
FIGURE = [SC.HAIR_BACK, SC.SKIN, SC.HAIR, SC.LACE, SC.BOW, SC.EYE, SC.PIN, SC.PLACKET, SC.BUTTON]
MENDABLE = [SC.HAIR_BACK, SC.SKIN, SC.HAIR, SC.LACE, SC.BOW, SC.EYE, SC.PLACKET, SC.DRESS]


class AstraPainter(Studio):
    def __init__(self, W, seed=3, log=print):
        super().__init__(W, W * 5 // 4, seed=seed, sketch=SC.build, log=log,
                         palette='classic')                 # the thirteen tubes this picture was painted with
        self.sc = self.sk

    def figure_blockin(self):
        specs = [(SC.HAIR_BACK, 9.0, (40, 150), 'body', True), (SC.SKIN, 7.0, (24, 80), 'smooth', False),
                 (SC.LACE, 3.2, (10, 34), 'body', True), (SC.PLACKET, 3.6, (14, 60), 'body', True),
                 (SC.BOW, 4.0, (12, 44), 'body', True), (SC.HAIR, 8.0, (40, 170), 'hair', True),
                 (SC.PIN, 2.2, (8, 30), 'body', True), (SC.EYE, 2.0, (5, 16), 'smooth', False),
                 (SC.BUTTON, 1.5, (3, 8), 'body', True)]
        self.underpaint([s[0] for s in specs], edge=9.0)
        self.block_in(specs=specs, margin=0.0, label='figure block-in')

    def background(self):
        """The night: long loaded strokes of chromatic black swept round the eclipse."""
        sc = self.sc
        bg = (sc.region == SC.BG)
        disc = sc.masks['disc'] > 0.5

        def rec_fn(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix], calm=disc[iy, ix] & (self.rng.random(len(p)) < 0.85))

        def iri_fn(k, pw=1.0):
            return lambda iy, ix: (np.clip(k * sc.sheen[iy, ix] ** pw * self.rng.uniform(0.7, 1.1, len(iy)), 0, 1),
                                   sc.film[iy, ix])

        self.fill('bg', bg, 17, (220, 560), cover=2.3, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn(0.22), unmix=0.55, inertia=0.72, filmvar=30, opacity=0.98, dry=0.30, margin=280.0,
                  load=2.2, overshoot=1, filmload=0.1)
        self.fill('bg', bg, 10, (140, 380), cover=1.5, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn(0.5, 1.3), unmix=0.8, pickup=0.05, inertia=0.68, filmvar=40, opacity=0.96, dry=0.36,
                  margin=190.0, load=1.9, overshoot=1, filmload=0.1)
        self.fill('bg', bg, 5.5, (70, 220), cover=0.7, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn(0.6, 1.5), unmix=1.0, pickup=0.06, inertia=0.6, filmvar=40, opacity=0.95, dry=0.4,
                  margin=110.0, load=1.6, overshoot=1, filmload=0.1, thick=1.7)
        for R in (6.0, 3.0, 1.6):                  # close whatever the big brushes could not reach
            gap = self.uncovered(bg)
            if gap.sum() > 30:
                gap = cv2.dilate(gap.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                self.fill('bg', gap & bg, R, (4 * R, 14 * R), cover=3.0, style='impasto', recipe_fn=rec_fn,
                          edge_tol=9.0, unmix=0.6, opacity=0.98, dry=0.15, overshoot=1, thick=1.4)
        self.stage('background block-in')

    def cut_silhouette(self):
        """Redraw the silhouette from the outside: small strokes of the night laid along the
        figure's edge, and of the dress along the lace and skin."""
        sc = self.sc
        s = self.s
        fig = sc.masks['figure'] > 0.5
        bg = sc.region == SC.BG
        disc = sc.masks['disc'] > 0.5
        dist = cv2.distanceTransform((~fig).astype(np.uint8), cv2.DIST_L2, 3) / s
        fld = self.contour_field(fig, 2.5)

        def rec_bg(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix], calm=disc[iy, ix])

        for R, d in ((3.2, 13.0), (1.7, 6.0), (1.0, 3.2)):
            m = bg & (dist > 0.75 * R) & (dist < d)
            self.fill('bg', m, R, (12, 46), cover=2.5, style='impasto', recipe_fn=rec_bg, edge_tol=9.0, field=fld,
                      unmix=0.6, thick=1.3, opacity=0.98, dry=0.18, inertia=0.5)
        dr = sc.region == SC.DRESS
        other = fig & ~dr
        dist2 = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3) / s
        fld2 = self.contour_field(other, 2.0)

        def rec_dr(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix])

        for R, d in ((1.8, 6.0), (1.0, 3.0)):
            m = dr & (dist2 > 0.75 * R) & (dist2 < d)
            self.fill('dress', m, R, (8, 34), cover=2.5, style='impasto', recipe_fn=rec_dr, edge_tol=9.0, field=fld2,
                      unmix=0.6, thick=1.2, opacity=0.98, dry=0.18, inertia=0.5)
        self.stage('cutting in the silhouette')

    def halo(self):
        """The lit crescent of the eclipse: long thin arcs, colour turning round the ring."""
        rng = self.rng
        c0 = SC.HALO_C
        R0 = SC.HALO_R
        dlt = SC.HALO_IN_C - c0
        IR = SC.HALO_IN_R
        white = np.ones(3, np.float32)

        def r_in(a):
            dx, dy = np.cos(a), np.sin(a)
            dd = dx * dlt[0] + dy * dlt[1]
            return dd + np.sqrt(np.maximum(dd * dd - dlt @ dlt + IR * IR, 0))

        def film_at(a):
            return 600.0 + 520.0 * (((a / (2 * math.pi)) + 0.62) % 1.0)

        def spectral(a):
            return self.film_rgb(film_at(a))

        def fade(a):          # the crescent sinks into shadow as it comes down toward the shoulders
            return 1.0 - 0.8 * float(SC.sstep(0.25, 0.8, math.sin(a)))

        items = []
        for _ in range(620):                      # soft coloured breath outside the rim
            a0 = rng.uniform(0, 2 * math.pi)
            a = np.linspace(a0, a0 + rng.uniform(0.18, 0.55), 24)
            am = float(a.mean())
            rad = R0 + rng.uniform(2, 40) + rng.normal(0, 2)
            th = np.clip(R0 - r_in(am), 0, 60) / 42.0
            pl = c0 + np.stack([np.cos(a), np.sin(a)], 1) * rad
            fall = math.exp(-(rad - R0) / 22.0)
            col = spectral(am) * (0.04 + 0.30 * th * fall) * rng.uniform(0.6, 1.3) * fade(am)
            items.append(dict(pl=pl, r=rng.uniform(3.5, 8.0), rgb=col + 0.004, style='glaze', opacity=0.55,
                              iri=(0.35 + 0.6 * th) * fade(am), film=float(film_at(am)), filmvar=40, thick=0.9))
        self.lines(self.on_regions(items, [SC.BG]), shuffle=True)
        items = []
        for _ in range(520):                      # the body of the crescent: broad loaded arcs
            a0 = rng.uniform(0, 2 * math.pi)
            a = np.linspace(a0, a0 + rng.uniform(0.16, 0.50), 28)
            am = float(a.mean())
            ri = r_in(a)
            th = R0 - ri
            ok = th > 2.5
            if ok.sum() < 5:
                continue
            f = rng.uniform(0.15, 0.85)
            pl = (c0 + np.stack([np.cos(a), np.sin(a)], 1) * (ri + f * th)[:, None])[ok]
            thn = float(np.clip(th[ok].mean() / 42.0, 0, 1))
            col = C.mix_lab(spectral(am) * 0.75 + 0.02, white * 0.95, 0.10 + 0.35 * thn) * (0.25 + 0.75 * fade(am))
            items.append(dict(pl=pl, r=rng.uniform(1.8, 3.6) * (0.5 + 0.7 * thn), rgb=col, style='impasto',
                              thick=2.2, iri=0.7, film=float(film_at(am)), filmvar=40, unmix=0.6, opacity=0.97,
                              pickup=0.03, dry=0.3))
        self.lines(self.on_regions(items, [SC.BG]), shuffle=True)
        items = []
        for _ in range(1700):                     # the lit edge: thin arcs, hottest where it is thickest
            a0 = rng.uniform(0, 2 * math.pi)
            a = np.linspace(a0, a0 + rng.uniform(0.10, 0.60), 28)
            am = float(a.mean())
            ri = r_in(a)
            th = R0 - ri
            ok = th > 0.9
            if th.max() < 1.2 or ok.sum() < 4:
                continue
            f = rng.uniform(0.06, 0.94)
            pl = (c0 + np.stack([np.cos(a), np.sin(a)], 1) * (ri + f * th)[:, None])[ok]
            thn = float(np.clip(th.mean() / 42.0, 0, 1))
            core = math.exp(-((f - 0.62) / 0.3) ** 2)
            fd = fade(am)
            col = C.mix_lab(spectral(am) * 0.9 + 0.02, white * 0.97, np.clip((0.22 + 0.68 * core * thn) * fd, 0, 0.92))
            col = col * (0.3 + 0.7 * fd)
            items.append(dict(pl=pl, r=rng.uniform(0.7, 1.9) * (0.6 + 0.6 * thn), rgb=col, style='hair',
                              thick=1.6 + 1.2 * core, iri=0.8, film=float(film_at(am)), filmvar=40,
                              unmix=0.5, opacity=0.97))
        for k in range(60):                       # a hair-thin closed rim all the way round
            a0 = k / 60 * 2 * math.pi + rng.uniform(-0.05, 0.05)
            a = np.linspace(a0, a0 + rng.uniform(0.14, 0.26), 20)
            am = float(a.mean())
            pl = c0 + np.stack([np.cos(a), np.sin(a)], 1) * (R0 + rng.uniform(-0.8, 0.8))
            thn = float(np.clip((R0 - r_in(am)) / 42.0, 0, 1))
            col = C.mix_lab(spectral(am) * (0.40 + 0.5 * thn), white, 0.18 + 0.5 * thn) * (0.35 + 0.65 * fade(am))
            items.append(dict(pl=pl, r=rng.uniform(0.5, 0.9), rgb=col, style='line', iri=0.9,
                              film=float(film_at(am)), thick=1.4))
        self.lines(self.on_regions(items, [SC.BG]), shuffle=True)
        self.stage('eclipse crescent')

    def feathers(self):
        rng = self.rng
        for f in self.sc.geo.feathers:
            base_film = rng.uniform(1060, 1100)
            items = []
            for b in f.barbs:                      # the dark vane, barb by barb
                fam = rng.choice(['indigo', 'violet', 'blue', 'teal'])
                items.append(dict(pl=b['pl'], r=f.wmax * rng.uniform(0.05, 0.075), style='hair', thick=1.3,
                                  rec=C.recipe(**C.CHROMATIC_BLACKS[fam]), unmix=0.8, taper=0.8))
            self.lines(items, shuffle=True)
            items = []
            for b in f.barbs:                      # structural colour running down the vane
                if rng.random() < 0.2:
                    continue
                film = base_film + 170.0 * b['t'] + (25 if b['side'] > 0 else -20) + rng.normal(0, 14)
                lift = rng.uniform(0.16, 0.62) * (0.35 + 0.65 * math.sin(math.pi * b['t']))
                items.append(dict(pl=b['pl'], r=f.wmax * rng.uniform(0.024, 0.042), style='hair', thick=1.6,
                                  rgb=self.film_rgb(film) * lift + 0.006, iri=1.0, film=float(film), filmvar=25,
                                  opacity=0.92, taper=0.85, unmix=0.5, filmload=0.2))
            items.append(dict(pl=f.rachis, r=f.wmax * 0.028, rgb=C.rgb8(206, 206, 230), style='line', thick=1.6,
                              taper=0.9, iri=0.5, film=float(base_film + 100)))
            self.lines(items, shuffle=False)
        self.stage('feathers')

    def stars(self):
        rng = self.rng
        sc = self.sc
        items = []
        white = C.rgb8(255, 253, 244)
        tints = [white, white, white] + [C.mix_lab(C.rgb8(h), white, 0.55) for h in IRIS_NINE[1:]]
        n = 0
        while n < 760:
            p = np.array([rng.uniform(8, 992), rng.uniform(8, 1242)])
            iy, ix = self.uidx(p[None])
            if sc.region[iy[0], ix[0]] != SC.BG:
                continue
            rr = math.hypot(*(p - SC.HALO_C))
            if rr < SC.HALO_R and rng.random() < 0.72:
                continue
            n += 1
            r = 0.30 + 1.7 * rng.random() ** 3.6
            a = rng.uniform(0, math.pi)
            d = np.array([math.cos(a), math.sin(a)]) * r * 0.6
            items.append(dict(pl=np.stack([p - d, p + d]), r=r, rgb=tints[rng.integers(len(tints))], style='dab',
                              thick=2.2 + 1.6 * rng.random(), iri=0.6, film=float(rng.uniform(600, 1100))))
        placed = 0
        while placed < 6:                           # a few four-point glints
            p = np.array([rng.uniform(40, 960), rng.uniform(40, 1000)])
            iy, ix = self.uidx(p[None])
            if sc.region[iy[0], ix[0]] != SC.BG or math.hypot(*(p - SC.HALO_C)) < SC.HALO_R * 1.05:
                continue
            placed += 1
            L = rng.uniform(4.5, 10)
            col = tints[rng.integers(len(tints))]
            a0 = rng.uniform(-0.12, 0.12)
            for a, k in ((a0, 1.0), (a0 + math.pi / 2, 1.3)):
                d = np.array([math.cos(a), math.sin(a)]) * L * k
                items.append(dict(pl=np.stack([p - d, p, p + d]), r=0.42, rgb=col, style='line', taper=0.97, thick=1.3))
            items.append(dict(pl=np.stack([p - [0.4, 0], p + [0.4, 0]]), r=1.0, rgb=tints[0], style='dab'))
        self.lines(items, shuffle=True)
        self.stage('stars')

    def dress(self):
        sc = self.sc
        m = sc.region == SC.DRESS

        def rec_fn(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix])

        def iri_fn(k, pw):
            return lambda iy, ix: (np.clip(k * sc.sheen[iy, ix] ** pw * self.rng.uniform(0.7, 1.1, len(iy)), 0, 1),
                                   sc.film[iy, ix])

        self.fill('dress', m, 9.5, (80, 240), cover=2.3, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn(0.25, 1.5), unmix=0.6, filmvar=30, opacity=0.98, dry=0.3, overshoot=1, load=2.0,
                  filmload=0.1, margin=120.0)
        self.fill('dress', m, 5.5, (50, 150), cover=1.3, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn(0.5, 1.6), unmix=0.9, pickup=0.05, filmvar=40, opacity=0.96, overshoot=1,
                  filmload=0.1, margin=75.0)
        for R in (4.0, 2.0, 1.2):
            gap = self.uncovered(m)
            if gap.sum() > 30:
                gap = cv2.dilate(gap.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                self.fill('dress', gap & m, R, (4 * R, 14 * R), cover=3.0, style='impasto', recipe_fn=rec_fn,
                          edge_tol=9.0, unmix=0.6, opacity=0.98, dry=0.15, overshoot=1, thick=1.4, margin=10.0)
        self.stage('dress block-in')

    def dress_sheen(self):
        """Satin: each gather a broad dim band, a brighter band, then a thin wet core of light."""
        rng = self.rng
        sc = self.sc
        m = sc.region == SC.DRESS
        self.sheen_glaze(m, 'dress', sizes=((6.5, (60, 190), 0.9), (4.0, (40, 130), 0.7)), strength=1.1, thresh=0.5)
        white = np.ones(3, np.float32)
        wide, mid, core, darks, rim = [], [], [], [], []

        def fold(pl, film, lift, nv):
            fr = self.film_rgb(film)
            wide.append(dict(pl=pl, r=rng.uniform(6.0, 9.0), rgb=fr * 0.13 * lift / 0.3 + 0.004, style='impasto',
                             thick=1.5, iri=0.8, film=float(film), filmvar=25, filmload=0.15, taper=0.9, opacity=0.75,
                             unmix=0.5, dry=0.5, pickup=0.03))
            mid.append(dict(pl=pl + nv * rng.uniform(-0.8, 0.8), r=rng.uniform(3.0, 4.6), rgb=fr * lift * 1.25 + 0.005,
                            style='impasto', thick=2.0, iri=1.0, film=float(film), filmvar=22, filmload=0.2, taper=0.92,
                            opacity=0.92, unmix=0.5, dry=0.35))
            c0 = int(len(pl) * rng.uniform(0.12, 0.3))
            c1 = int(len(pl) * rng.uniform(0.6, 0.85))
            core.append(dict(pl=pl[c0:c1] - nv * rng.uniform(0.3, 1.1), r=rng.uniform(0.8, 1.3),
                             rgb=C.mix_lab(fr * 0.8, white, 0.25 + 0.5 * lift), style='hair', thick=2.0, iri=1.0,
                             film=float(film), taper=0.95, opacity=0.95))

        for puff, seam, sgn in ((sc.geo.puff[0], SC.bd((-1.50, 2.12)), -1), (sc.geo.puff[1], SC.bd((1.50, 2.12)), 1)):
            n = len(puff)
            fr_ = np.linspace(0.17, 0.72, 11) + rng.uniform(-0.012, 0.012, 11)
            for k, f in enumerate(fr_):           # gathers fanning from the shoulder seam
                tgt = puff[int(f * n)]
                v = tgt - seam
                L = np.linalg.norm(v)
                nv = np.array([-v[1], v[0]]) / L
                bend = (0.10 + 0.09 * math.sin(k * 1.7)) * sgn
                pl = catmull([seam + v * rng.uniform(0.10, 0.2), seam + v * 0.55 + nv * L * bend,
                              seam + v * rng.uniform(0.88, 0.97)], n=12)
                film = float(SC.film_path(0.085 * k + 0.13 * (sgn > 0)))
                if k % 2 == 0:
                    lit = 0.55 + 0.45 * max(0.0, -(nv @ LIGHT2D) * sgn)
                    fold(pl, film, rng.uniform(0.20, 0.42) * lit, nv * (1 if (nv @ LIGHT2D) < 0 else -1))
                else:
                    darks.append(dict(pl=pl, r=rng.uniform(2.0, 3.6), style='impasto', thick=1.5, taper=0.85,
                                      rec=C.recipe(**C.CHROMATIC_BLACKS[rng.choice(['violet', 'indigo', 'blue'])]),
                                      unmix=0.5, opacity=0.95))
            # one continuous lit rim round the top and outside of the sleeve, hue sliding along it
            edge = np.vstack([puff, puff[:1]])[int(0.03 * n): int(0.60 * n)]
            for q in range(7):
                a, b = int(len(edge) * q / 7), int(len(edge) * (q + 1) / 7) + 2
                film = float(SC.film_path(0.11 * q + 0.3 * (sgn > 0)))
                rim.append(dict(pl=edge[a:b], r=0.85, rgb=C.mix_lab(self.film_rgb(film) * 0.5, white, 0.12), style='hair',
                                thick=1.8, iri=1.0, film=float(film), filmvar=30, taper=0.4, opacity=0.95))
        for sx in (-1, 1):                          # bodice: falling folds and the princess seam
            for k, (x0, x1) in enumerate(((0.44, 0.36), (0.82, 0.64), (1.20, 1.04))):
                pl = SC.bd(catmull(np.array([(x0, 3.14), ((x0 + x1) / 2 + 0.05, 3.7), (x1, 4.3), (x1 - 0.02, 5.02)]) * [sx, 1], n=10))
                film = float(SC.film_path(0.21 * k + 0.15 * (sx > 0) + 0.2))
                fold(pl, film, rng.uniform(0.14, 0.30), np.array([-1.0, 0.0]))
            seam = SC.bd(catmull(np.array([(0.78, 3.05), (0.70, 3.6), (0.56, 4.2), (0.52, 5.02)]) * [sx, 1], n=10))
            rim.append(dict(pl=seam, r=0.55, rgb=C.rgb8(58, 60, 110), style='hair', thick=1.5, iri=0.9,
                            film=float(SC.film_path(rng.random() * 0.5)), filmvar=30, taper=0.6))
        self.lines(self.on_regions(darks, [SC.DRESS]), shuffle=True)
        self.lines(self.on_regions(wide, [SC.DRESS]), shuffle=True)
        self.lines(self.on_regions(mid, [SC.DRESS]), shuffle=True)
        self.cv.dry(1.0)
        self.lines(self.on_regions(core, [SC.DRESS]), shuffle=True)
        self.lines(self.on_regions(rim, [SC.DRESS, SC.BG]), shuffle=False)
        self.stage('dress: folds and sheen')

    def hair_strands(self):
        rng = self.rng
        sc = self.sc
        P = SC.P
        back, front, edges, lights = [], [], [], []
        hc = SC.hd((0.0, -0.62))
        for L in sc.geo.locks:
            cl, wd = L['center'], L['width']
            n = len(cl)
            nn = normals(cl)
            is_back = L['layer'] == 'back'
            wmax = float(wd.max())
            ns = int(wmax / (2.0 if is_back else 0.72)) + 2
            lit_sign = np.sign(nn[n // 2] @ LIGHT2D)
            for _ in range(ns):
                o = rng.uniform(-0.47, 0.47)
                i0 = int(rng.uniform(0.0, 0.5) * (n - 1))
                i1 = int(rng.uniform(0.62, 1.0) * (n - 1))
                if i1 - i0 < 4:
                    continue
                seg = cl[i0:i1 + 1] + nn[i0:i1 + 1] * (o * wd[i0:i1 + 1])[:, None]
                lit = o * lit_sign / 0.47
                # take the strand's colour from the underdrawing where the strand is actually hair
                # (not from the hairpin or the face it happens to pass behind)
                iy, ix = self.uidx(seg)
                own = np.flatnonzero(sc.region[iy, ix] == (SC.HAIR_BACK if is_back else SC.HAIR))
                if len(own) < 3:
                    continue
                iy, ix = iy[own[[len(own) // 2]]], ix[own[[len(own) // 2]]]
                base = sc.target[iy[0], ix[0]]
                if is_back:
                    col = self.jit(base, 0.04, 0.012)
                    if rng.random() < 0.18:
                        col = C.mix_lab(P['hair_deep'], P['hair_shadow'], rng.uniform(0, 0.6))
                    for run in self.runs(seg, [SC.HAIR_BACK], min_len=5.0):
                        back.append(dict(pl=run, r=rng.uniform(1.1, 2.3), rgb=col, style='hair'))
                    continue
                u = rng.random()
                col = self.jit(base, 0.025, 0.010)
                if lit > 0.2 and u < 0.30:
                    col = P['white'] * 0.985
                elif lit < -0.15 and u < 0.42:
                    v = rng.random()
                    if v < 0.34:
                        col = C.mix_lab(P['hair_shadow'], P['hair_deep'], rng.uniform(0, 0.5))
                    elif v < 0.62:
                        col = C.mix_lab(P['hair_cyan'], P['hair_shadow'], rng.uniform(0.1, 0.5))
                    else:
                        col = C.mix_lab(P['hair_pink'], P['hair_shadow'], rng.uniform(0.1, 0.5))
                xl, yl = sc.Xl[iy[0], ix[0]], sc.Yl[iy[0], ix[0]]
                if rng.random() < 0.34:               # light of the eclipse caught in the outer hair
                    if xl < -1.2:
                        col = C.mix_lab(col, C.rgb8(150, 206, 242), rng.uniform(0.25, 0.6))
                    elif xl > 1.2:
                        col = C.mix_lab(col, C.rgb8(250, 232, 168), rng.uniform(0.25, 0.55))
                    elif yl < -1.5:
                        col = C.mix_lab(col, C.rgb8(196, 240, 226), rng.uniform(0.2, 0.5))
                for run in self.runs(seg, [SC.HAIR], min_len=5.0):
                    front.append(dict(pl=run, r=0.45 + 1.25 * rng.random() ** 1.6, rgb=col, style='hair',
                                      thick=1.1 + 0.9 * max(lit, 0)))
                # where the strand crosses the highlight band, a loaded touch of white
                rho = np.hypot(*(seg - hc).T) / SC.U
                iy, ix = self.uidx(seg)
                ang = math.atan2(*(seg[len(seg) // 2] - hc)[::-1])
                rho0 = 0.95 + 0.055 * math.sin(ang * 9.0) + rng.uniform(-0.03, 0.03)     # a zigzag, not a hoop
                inb = (np.abs(rho - rho0) < rng.uniform(0.025, 0.075)) & (sc.Yl[iy, ix] < -0.5) \
                    & (sc.region[iy, ix] == SC.HAIR)
                if inb.sum() >= 3 and lit > -0.1 and rng.random() < 0.34:
                    k = np.flatnonzero(inb)
                    lights.append(dict(pl=seg[k[0]:k[-1] + 1], r=rng.uniform(0.3, 0.6), rgb=P['white'], style='hair',
                                       thick=1.9, taper=0.9, opacity=0.95))
            if not is_back:
                # line work: the shadowed edge of each lock, drawn with a rigger
                for sgn in (-1.0, 1.0):
                    shadow_side = sgn * lit_sign < 0
                    if not shadow_side and rng.random() < 0.55:
                        continue
                    i0 = int(rng.uniform(0.08, 0.3) * (n - 1))
                    seg = cl[i0:] + nn[i0:] * (sgn * 0.5 * wd[i0:])[:, None]
                    col = C.mix_lab(P['hair_deep'], P['line'], rng.uniform(0.15, 0.5)) if shadow_side \
                        else C.mix_lab(P['hair_shadow'], P['hair_mid'], 0.4)
                    for run in self.runs(seg, [SC.HAIR, SC.SKIN, SC.HAIR_BACK], min_len=8.0):
                        edges.append(dict(pl=run, r=rng.uniform(0.38, 0.62), rgb=col, style='line',
                                          opacity=0.9 if shadow_side else 0.7))
        self.lines(back, shuffle=True)
        self.cv.dry(1.0)
        self.lines(front, shuffle=True)
        self.cv.dry(1.0)
        self.lines(edges, shuffle=True)
        self.lines(lights, shuffle=True)
        # flyaway hairs leaving the silhouette
        fly = []
        cap = sc.geo.cap
        top = cap[: int(len(cap) * 0.58)]
        tn = tangents(top)
        for _ in range(46):
            k = rng.integers(2, len(top) - 2)
            p0 = top[k]
            outv = np.array([tn[k][1], -tn[k][0]])
            if outv @ (p0 - SC.hd((0, -0.5))) < 0:
                outv = -outv
            ln = rng.uniform(14, 52)
            dirv = tn[k] * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0) + outv * rng.uniform(0.15, 0.6)
            dirv /= np.linalg.norm(dirv)
            p1 = p0 + dirv * ln * 0.5 + outv * ln * 0.10
            p2 = p0 + dirv * ln + np.array([0, 1.0]) * ln * rng.uniform(0.05, 0.3)
            fly.append(dict(pl=catmull([p0 - dirv * 6, p0, p1, p2], n=8), r=rng.uniform(0.28, 0.5),
                            rgb=C.mix_lab(P['hair_light'], P['hair_shadow'], rng.uniform(0, 0.5)), style='line',
                            opacity=0.85, taper=0.9))
        self.lines(fly)
        self.stage('hair, strand by strand')

    def _inside_opening(self, ey, pts):
        if not hasattr(ey, '_op'):
            ey._op = self.sc.ra.poly(ey.opening) > 0.5
        iy, ix = self.uidx(pts)
        return ey._op[iy, ix]

    def face_lines(self):
        rng = self.rng
        sc = self.sc
        g = sc.geo
        P = SC.P
        U = SC.U
        ct, st = math.cos(SC.TILT), math.sin(SC.TILT)
        down = np.array([-st, ct])
        items = []
        soft = C.mix_lab(P['skin_deep'], P['line'], 0.45)
        for run in self.runs(g.jaw, [SC.SKIN]):
            items.append(dict(pl=run, r=0.5, rgb=soft, opacity=0.85))
        items.append(dict(pl=g.nose, r=0.5, rgb=C.mix_lab(P['skin_deep'], P['line'], 0.3), opacity=0.8))
        items.append(dict(pl=g.mouth, r=0.62, rgb=P['line'], taper=0.85))
        mid = g.mouth[len(g.mouth) // 4: -len(g.mouth) // 4]
        items.append(dict(pl=mid + [0, 0.25], r=0.55, rgb=C.mix_lab(P['line'], P['lash'], 0.4), taper=0.9))
        items.append(dict(pl=g.lip_low, r=0.9, rgb=P['lip'], opacity=0.45, style='smooth'))
        for sx in (-1, 1):                                    # blush hatching
            for k in range(3):
                c = SC.hd((sx * (0.55 + 0.07 * k), 0.47))
                d = np.array([0.045, -0.085]) * U
                items.append(dict(pl=np.stack([c - d, c + d]), r=0.42, rgb=C.rgb8(240, 140, 150), opacity=0.5, taper=0.9))
        for sx in (-1, 1):                                    # the chest turns away under the lace
            for k, x in enumerate((0.80, 0.70, 0.60)):
                pl = SC.bd(catmull([(x * sx, 2.12 + 0.04 * k), ((x - 0.05) * sx, 2.5), ((x - 0.08) * sx, 2.86)], n=8))
                items.append(dict(pl=pl, r=4.6, rgb=P['skin_shadow'], opacity=0.17 - 0.04 * k, style='smooth', taper=0.6,
                                  ragged=0.9, wobble=1.4))
            pl = SC.bd(catmull([(0.14 * sx, 2.16), (0.42 * sx, 2.10), (0.74 * sx, 2.15)], n=8))
            items.append(dict(pl=pl, r=1.6, rgb=C.mix_lab(P['skin_light'], P['white'], 0.6), opacity=0.5, style='smooth'))
        items.append(dict(pl=SC.bd([(-0.62, 2.84), (0.0, 2.88), (0.62, 2.84)]), r=3.6, rgb=P['skin_shadow'], opacity=0.2,
                          style='smooth', taper=0.6))
        for sx in (-1, 1):                                    # collar bones, sides of the neck
            cb = SC.bd(catmull(np.array([(0.10, 2.08), (0.40, 2.02), (0.72, 2.06), (0.93, 2.14)]) * [sx, 1], n=8))
            items.append(dict(pl=cb, r=0.7, rgb=P['skin_shadow'], opacity=0.75, style='smooth'))
            nk = np.vstack([SC.hd([(0.29 * sx, 1.02)]), SC.bd([(0.31 * sx, 1.50), (0.36 * sx, 1.78)])])
            for run in self.runs(resample(nk, n=24), [SC.SKIN]):
                items.append(dict(pl=run, r=0.5, rgb=soft, opacity=0.8))
        self.lines(items)

        def spiral(c, rx, ry, turns=3.0, n=70):
            t = np.linspace(0.06, 1.0, n)
            a = t * turns * 2 * math.pi
            x = np.cos(a) * rx * t
            y = np.sin(a) * ry * t
            return c + np.stack([x * ct - y * st, x * st + y * ct], 1)

        items = []
        for ey in g.eyes:
            # the iris: a soft shadow under the lid, fine fibres, flecks in the nine official colours
            sh = ey.upper + down * 0.045 * U
            loc = (sh - ey.iris_c) @ np.array([[ct, -st], [st, ct]])
            ins = (loc[:, 0] / ey.iris_rx) ** 2 + (loc[:, 1] / ey.iris_ry) ** 2 < 0.9
            if ins.sum() >= 3:
                k = np.flatnonzero(ins)
                items.append(dict(pl=sh[k[0]:k[-1] + 1], r=0.03 * U, rgb=C.mix_lab(P['iris_top'], P['pupil'], 0.55),
                                  opacity=0.55, style='smooth', taper=0.3))
            for k in range(46):
                a = rng.uniform(-0.3, math.pi + 0.3)
                r0, r1 = rng.uniform(0.46, 0.60), rng.uniform(0.74, 0.90)
                e = np.array([[math.cos(a) * ey.iris_rx * r, math.sin(a) * ey.iris_ry * r] for r in (r0, r1)])
                pl = ey.iris_c + np.stack([e[:, 0] * ct - e[:, 1] * st, e[:, 0] * st + e[:, 1] * ct], 1)
                if not self._inside_opening(ey, pl).all():
                    continue
                if rng.random() < 0.38:
                    col = C.mix_lab(C.rgb8(IRIS_NINE[rng.integers(1, 9)]), P['iris_bot'], rng.uniform(0.3, 0.6))
                else:
                    col = C.mix_lab(P['iris_bot'], P['white'], rng.uniform(0, 0.55))
                items.append(dict(pl=pl, r=0.0042 * U, rgb=col, opacity=0.6, taper=0.85))
            ring = ellipse_pts(ey.iris_c[0], ey.iris_c[1], ey.iris_rx * 0.96, ey.iris_ry * 0.96, SC.TILT, n=90)
            inside = self._inside_opening(ey, ring)
            start = None
            for i, o in enumerate(inside):
                if o and start is None:
                    start = i
                if (not o or i == len(inside) - 1) and start is not None:
                    if i - start >= 3:
                        items.append(dict(pl=ring[start:i], r=0.0055 * U, rgb=P['iris_ring'], taper=0.3, opacity=0.9))
                    start = None
            items.append(dict(pl=spiral(ey.pupil_c, ey.pupil_rx * 0.86, ey.pupil_ry * 0.86), r=0.2 * ey.pupil_ry,
                              rgb=P['pupil'], taper=0.15, cap=0.9, thick=0.8))
            # the lash band, laid solid, then its two edges drawn
            for f in (0.5, 0.08, 0.25, 0.42, 0.58, 0.75, 0.92):
                pl = ey.upper * (1 - f) + ey.lash_top * f
                items.append(dict(pl=pl, r=0.019 * U, rgb=C.mix_lab(P['lash'], P['lash_tip'], 0.12 * f), thick=1.2,
                                  taper=0.85))
            items.append(dict(pl=(ey.upper * 0.45 + ey.lash_top * 0.55)[42:], r=0.02 * U, rgb=P['lash_tip'],
                              opacity=0.75, taper=0.9))
            items.append(dict(pl=ey.upper * 0.98 + ey.lash_top * 0.02, r=0.0055 * U, rgb=P['lash'], taper=0.6))
            items.append(dict(pl=ey.upper * 0.03 + ey.lash_top * 0.97, r=0.0055 * U, rgb=P['lash'], taper=0.7))
            for sp_ in ey.spikes:
                items.append(dict(pl=sp_, r=0.010 * U, rgb=P['lash'], taper=0.97, cap=1.0))
            items.append(dict(pl=ey.lower_line, r=0.006 * U, rgb=C.mix_lab(P['lash'], P['line'], 0.3), taper=0.9))
            items.append(dict(pl=ey.crease, r=0.0052 * U, rgb=C.mix_lab(P['line'], P['skin_deep'], 0.35),
                              opacity=0.8, taper=0.9))
        self.lines(items)
        items = []
        for ey in g.eyes:
            a = SC.TILT - 0.5
            ax = np.array([math.cos(a), math.sin(a)])
            pv = np.array([-ax[1], ax[0]])
            for off in (-0.55, -0.18, 0.18, 0.55):          # the big catch-light, an oval of thick white
                half = ey.hl1_r[0] * math.sqrt(1 - off * off) * 0.78
                c = ey.hl1 + pv * off * ey.hl1_r[1]
                items.append(dict(pl=np.stack([c - ax * half, c + ax * half]), r=ey.hl1_r[1] * 0.36, rgb=P['white'],
                                  style='dab', thick=2.2, blob=0.5))
            items.append(dict(pl=np.stack([ey.hl2 - [0.4, 0], ey.hl2 + [0.4, 0]]), r=ey.hl2_r, rgb=P['white'],
                              style='dab', thick=2.0, blob=0.6))
            for run in self.runs(ey.brow, [SC.SKIN]):       # brows, almost lost under the fringe
                items.append(dict(pl=run, r=0.0085 * U, rgb=P['brow'], taper=0.7))
            for run in self.runs(ey.brow, [SC.HAIR]):
                items.append(dict(pl=run, r=0.007 * U, rgb=C.mix_lab(P['brow'], P['hair_mid'], 0.3), opacity=0.4,
                                  taper=0.7))
        self.lines(items)
        self.stage('face and eyes')

    def trimmings(self):
        rng = self.rng
        sc = self.sc
        g = sc.geo
        P = SC.P
        U = SC.U
        items = []
        eyelet = C.rgb8(132, 112, 124)
        for band in g.lace_bands:                                  # lace: eyelets, a raised rim on every scallop
            cen = band['center']
            for c, r in band['scallops']:
                c = np.asarray(c)
                iy, ix = self.uidx(c[None])
                if sc.region[iy[0], ix[0]] != SC.LACE:
                    continue
                k = int(np.argmin(np.hypot(*(band['edge'] - c).T)))
                o = band['out'][k]
                a_mid = math.atan2(o[1], o[0])
                items.append(dict(pl=ellipse_pts(c[0], c[1], r * 0.92, r * 0.92, n=12, a0=a_mid - 1.25, a1=a_mid + 1.25),
                                  r=0.5, rgb=P['white'], style='hair', thick=2.0, opacity=0.95, taper=0.5))
                items.append(dict(pl=np.stack([c - [0.25, 0], c + [0.25, 0]]), r=r * 0.27, rgb=eyelet, style='dab',
                                  thick=0.5, blob=0.2))
                q = c - o * r * 1.25                               # a second, smaller row of holes further in
                items.append(dict(pl=np.stack([q - [0.2, 0], q + [0.2, 0]]), r=r * 0.15, rgb=eyelet, style='dab',
                                  thick=0.4, blob=0.2, opacity=0.85))
            for k in range(3, len(cen) - 8, 8):                    # stitches along the far edge of the band
                o = band['out'][k]
                p0 = cen[k] - o * band['width'] * 0.36
                for run in self.runs(np.stack([p0, cen[k + 4] - o * band['width'] * 0.36]), [SC.LACE]):
                    items.append(dict(pl=run, r=0.34, rgb=P['lace_shadow'], opacity=0.8, taper=0.5))
        for pe in g.placket_edges:
            for c, r in pe['scallops']:
                c = np.asarray(c)
                iy, ix = self.uidx(c[None])
                if sc.region[iy[0], ix[0]] != SC.PLACKET:
                    continue
                a_mid = 0.0 if pe['sgn'] > 0 else math.pi
                items.append(dict(pl=ellipse_pts(c[0], c[1], r * 0.9, r * 0.9, n=10, a0=a_mid - 1.2, a1=a_mid + 1.2),
                                  r=0.42, rgb=P['white'], style='hair', thick=1.9, opacity=0.95, taper=0.5))
                items.append(dict(pl=np.stack([c - [0.22, 0], c + [0.22, 0]]), r=r * 0.24, rgb=eyelet, style='dab',
                                  thick=0.5, blob=0.2))
        for run in self.runs(SC.bd([(-0.03, 3.25), (-0.02, 5.02)]), [SC.PLACKET]):
            items.append(dict(pl=run, r=0.9, rgb=P['white'], style='hair', opacity=0.8, thick=1.6))
        self.lines(items)

        items = []
        for c in g.buttons:                                        # buttons like chips of black opal
            r = g.button_r
            for k in (0.75, 0.35):
                items.append(dict(pl=ellipse_pts(c[0], c[1], r * k, r * k, n=22), r=r * 0.3, style='hair', thick=2.0,
                                  rec=C.recipe(**C.CHROMATIC_BLACKS['indigo']), iri=1.0,
                                  film=float(rng.uniform(600, 1100)), filmvar=120, unmix=1.0))
            items.append(dict(pl=np.stack([c - [r * 0.45, r * 0.4], c - [r * 0.2, r * 0.5]]), r=0.5, rgb=P['white'],
                              style='dab', thick=2.4))
        bow = g.bow                                                # bow: folds, lights, a drawn edge
        for lp in bow.loops:
            far = lp[np.argmax(np.hypot(*(lp - bow.knot).T))]
            side = np.sign(far[0] - bow.knot[0])
            items.append(dict(pl=np.vstack([lp, lp[:2]]), r=0.45, rgb=P['bow_dark'], opacity=0.9))
            for k in range(5):
                tgt = lp[rng.integers(len(lp) // 5, len(lp) * 4 // 5)]
                mid = (bow.knot + tgt) / 2 + rng.normal(0, 1.2, 2)
                items.append(dict(pl=catmull([bow.knot + [side * 6, 0], mid, bow.knot * 0.25 + tgt * 0.75], n=8),
                                  r=rng.uniform(0.5, 1.0), rgb=C.mix_lab(P['bow_dark'], P['bow_mid'], rng.uniform(0, 0.4)),
                                  taper=0.85, style='hair'))
            for k in range(4):
                c = bow.knot + np.array([side * U * rng.uniform(0.26, 0.5), -U * rng.uniform(0.08, 0.17)])
                d = np.array([side * U * 0.09, U * rng.uniform(-0.02, 0.03)])
                items.append(dict(pl=catmull([c - d, c + [0, -1.2], c + d], n=6), r=rng.uniform(0.8, 1.4),
                                  rgb=C.mix_lab(P['bow_light'], P['white'], rng.uniform(0, 0.35)), style='dab', thick=2.0,
                                  taper=0.8, blob=0.4))
        for tl in bow.tails:
            items.append(dict(pl=np.vstack([tl, tl[:1]]), r=0.42, rgb=P['bow_dark'], opacity=0.9))
            items.append(dict(pl=np.stack([tl[0] * 0.5 + tl[4] * 0.5, tl[2]]), r=0.8,
                              rgb=C.mix_lab(P['bow_mid'], P['bow_light'], 0.5), style='hair', taper=0.8))
        kr = bow.knot_r
        items.append(dict(pl=ellipse_pts(bow.knot[0], bow.knot[1], kr[0] * 0.9, kr[1] * 0.9, n=24), r=0.5, rgb=P['bow_dark']))
        items.append(dict(pl=np.stack([bow.knot + [-kr[0] * 0.3, -kr[1] * 0.5], bow.knot + [-kr[0] * 0.25, kr[1] * 0.3]]),
                          r=1.1, rgb=P['bow_light'], style='dab', thick=2.0))
        self.lines(items)

        pin = g.pin                                                # crescent hairpin of rainbow obsidian
        dlt = pin.ic - pin.c
        a_mid = math.atan2(-dlt[1], -dlt[0])
        items = []
        for f in (0.2, 0.5, 0.8, 0.35, 0.65):
            a = np.linspace(a_mid - 2.2, a_mid + 2.2, 60)
            dx, dy = np.cos(a), np.sin(a)
            dd = dx * dlt[0] + dy * dlt[1]
            rin = dd + np.sqrt(np.maximum(dd * dd - dlt @ dlt + pin.ir ** 2, 0))
            th = pin.r - rin
            ok = th > 0.25 * th.max()
            pl = (pin.c + np.stack([dx, dy], 1) * (rin + f * th)[:, None])[ok]
            items.append(dict(pl=pl, r=float(th.max()) * 0.17, style='hair', thick=1.8, taper=0.9,
                              rec=C.recipe(**C.CHROMATIC_BLACKS[rng.choice(['indigo', 'violet', 'blue'])]),
                              iri=1.0, film=float(620 + 420 * f), filmvar=90, filmload=0.6, unmix=0.9))
        a = np.linspace(a_mid + 0.3, a_mid + 1.5, 16)
        items.append(dict(pl=pin.c + np.stack([np.cos(a), np.sin(a)], 1) * pin.r * 0.9, r=0.55, rgb=P['white'],
                          style='dab', thick=2.4, taper=0.9, blob=0.3))
        self.lines(items)
        self.stage('lace, bow, hairpin')

    def finishing(self):
        """A few small loaded touches of sheen in the darks: glints, not texture."""
        sc = self.sc
        dark = np.isin(sc.region, [SC.BG, SC.DRESS])

        def rec_fn(p, iy, ix):
            return self.black_recipes(sc.film[iy, ix])

        def iri_fn(iy, ix):
            return (np.clip(sc.sheen[iy, ix] * self.rng.uniform(0.6, 1.3, len(iy)), 0, 1),
                    sc.film[iy, ix] + self.rng.normal(0, 30, len(iy)))

        w = sc.sheen + 0.3 * self.rng.random(sc.sheen.shape, dtype=np.float32)
        self.fill('dark', dark, 1.6, (14, 60), cover=0.45, style='impasto', recipe_fn=rec_fn, edge_tol=9.0,
                  iri_fn=iri_fn, unmix=0.9, thick=1.6, filmvar=22, filmload=0.0, weights=w, thresh=0.66)
        self.stage('finishing touches')

    # ------------------------------------------------------------------ the whole picture
    def run(self):
        self.tone_ground(pigments=dict(alizarin=.24, umber=.36, violet=.40))
        self.background()
        self.sheen_glaze(self.sc.region == SC.BG, 'bg')
        self.stage('background sheen')
        self.feathers()
        self.dress()
        self.figure_blockin()
        self.cut_silhouette()
        self.mend(MENDABLE)
        self.refine(FIGURE, inset=0.0)
        self.soften([SC.SKIN], rewet=0.0)
        self.cv.dry(1.0)
        self.refine(FIGURE, passes=FINE, inset=0.0)
        self.mend(MENDABLE)
        self.dress_sheen()
        self.hair_strands()
        self.trimmings()
        self.face_lines()
        self.finishing()
        self.halo()
        self.stars()
        return self.cv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--width', type=int, default=3200, help='canvas width in pixels (height is 5/4 of it)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'astra_iridescent_black'),
                    help='output path without extension')
    ap.add_argument('--frames', action='store_true', help='also save a strip of the painting in progress')
    a = ap.parse_args()

    t0 = time.time()
    p = AstraPainter(a.width, seed=a.seed)
    frames = []
    if a.frames:
        p.frame_cb = lambda name: frames.append(cv2.resize(p.cv.render(), (520, 650), interpolation=cv2.INTER_AREA))
    p.run()
    cv = p.cv
    img = cv.render()
    _imwrite(a.out + '.png', img)
    pw = min(1600, cv.W)
    _imwrite(a.out + '_%d.jpg' % pw, cv2.resize(img, (pw, pw * 5 // 4), interpolation=cv2.INTER_AREA), 93)
    _imwrite(a.out + '_unlit.jpg', cv.render(lit=False))
    stats = dict(title='五彩斑斓的黑', **p.stats(), seconds=round(time.time() - t0, 1))
    with open(a.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    if frames:
        pick = frames if len(frames) <= 8 else [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
        _imwrite(a.out + '_process.jpg', np.concatenate(pick, 1), 90)
    print('done: %d brushstrokes, %d bristle trails, %.1fs -> %s.png'
          % (cv.n_strokes, cv.n_trails, time.time() - t0, a.out))


if __name__ == '__main__':
    main()
